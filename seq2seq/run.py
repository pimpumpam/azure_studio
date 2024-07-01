import os
import sys
import time
import yaml
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.optim as optim

from src.dataLoader import DataLoad, convert_dtype
from src.preprocessing import MissingValueHandler, MultiColumnScaler, TrainTestSpliter, LogScaler, Scaler, S1APDataset, SlidingWindow, gen_time_covariate
from src.model import Model, negative_log_loss, negative_log_likelihood, mse_loss, mae_loss, median_abs_loss, mape_loss
from src.train import train
from src.evaluate import evaluate
from src.utils import TQDM_BAR_FORMAT, combination_hyperparams, denormalization_pred, find_item


SEED = 8332
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
CONFIG_DIR = os.path.join(ROOT, 'seq2seq', 'configs')
ARTIFACT_DIR = os.path.join(ROOT, 'seq2seq', 'artifacts')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--auth', type=str, default='auth.yaml', help="NDAP 접속 config 파일명")
    parser.add_argument('--data', type=str, default='s1ap.yaml', help="데이터 config 파일명")
    parser.add_argument('--model' , type=str, default='seq2seq_lstm.yaml', help='모델 아키텍처 config 파일명')
    parser.add_argument('--hyp' , type=str, default='hyp.yaml', help="하이퍼파라미터 config 파일명")
    parser.add_argument('--epoch', type=int, default=30, help="모델 학습 반복 횟수")
    parser.add_argument('--station', type=str, default='NPKG31401S', help="분석 대상 관측소 명")
    parser.add_argument('--tunning_params', action='store_true', help="Hyperparameter 튜닝 여부")
    args = parser.parse_args()
    
    # configs
    with open(os.path.join(CONFIG_DIR, 'data', args.data)) as y:
        DATA_CONFIG = yaml.load(y, Loader=yaml.FullLoader)
        
    with open(os.path.join(CONFIG_DIR, 'models', args.model)) as y:
        MODEL_CONFIG = yaml.load(y, Loader=yaml.FullLoader)
        
    with open(os.path.join(CONFIG_DIR, 'hyps', args.hyp)) as y:
        HYP_CONFIG = yaml.load(y, Loader=yaml.FullLoader)
        
    
    # globals
    COVARIATE_TIMEUNIT = ['month', 'dayofweek', 'hour']
    INPUT_FEATURE = ['rx_byte'] # 'rx_byte', 'month_cov', 'dayofweek_cov', 'hour_cov'
    TARGET_FEATURE = 'rx_byte'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FRAMEWORK = find_item(DATA_CONFIG, 'framework')
    
    
    # Load data
    print("[INFO] Load Data")
    loader = DataLoad(framework=find_item(DATA_CONFIG, 'framework'))
    data = loader.load_from_local(path=find_item(DATA_CONFIG, 'path'), 
                                  ext=find_item(DATA_CONFIG, 'extension')
                                 )
    
    data = convert_dtype(data=data,
                         dtype_dict=find_item(DATA_CONFIG, 'dtype')
                        )
        
    data = data[data['ru'] == args.station].reset_index(drop=True)
    print(f"Size of Data: {np.shape(data)}")
    # Preprocessing
    # generate covariate
    # print("[INFO] Generate Covariate")
    # for t in COVARIATE_TIMEUNIT:
    #     data = gen_time_covariate(data, 'five_minute_interval', timeunit=t)
    
    # split data
    print("[INFO] Split Data")
    spliter = TrainTestSpliter(FRAMEWORK)
    train_data, test_data = spliter.split_by_time(data,
                                                  'five_minute_interval',
                                                  find_item(DATA_CONFIG, 'train_start_time'),
                                                  find_item(DATA_CONFIG, 'train_end_time'),
                                                  find_item(DATA_CONFIG, 'test_start_time'),
                                                  find_item(DATA_CONFIG, 'test_end_time'))
    
    if len(train_data)==0 | len(test_data)==0:
        print(f"Station: {args.station} Out of Range")
        sys.exit()
    
    
    # log scaling
    print("[INFO] Log Scaling")
    log_scaler = LogScaler(FRAMEWORK)
    train_data = log_scaler.transform(train_data, INPUT_FEATURE)
    test_data = log_scaler.transform(test_data, INPUT_FEATURE)
    
    
    # scaling
    print("[INFO] Scaling")
    if FRAMEWORK == "pyspark":
        # train data scaling
        train_scaler = Scaler(find_item(HYP_CONFIG, 'scaler'))
        train_data = train_scaler.fit_transform(train_data, INPUT_FEATURE)
        train_data = train_scaler.denseVector_to_float(train_data, INPUT_FEATURE)

        # test data scaling
        test_scaler = Scaler(find_item(HYP_CONFIG, 'scaler'))
        test_data = test_scaler.fit_transform(test_data, INPUT_FEATURE)
        test_data = test_scaler.denseVector_to_float(test_data, INPUT_FEATURE)
    
    elif FRAMEWORK == 'pandas':
        # train_data = train_data.toPandas()
        # test_data = test_data.toPandas()

        # train data scaling
        train_scaler = MultiColumnScaler(find_item(HYP_CONFIG, 'scaler'))
        train_data = train_scaler.fit_transform(train_data, INPUT_FEATURE)

        # test data scaling
        test_scaler = MultiColumnScaler(find_item(HYP_CONFIG, 'scaler'))
        test_data = test_scaler.fit_transform(test_data, INPUT_FEATURE)
    
    
    if args.tunning_params:
        
        RESULT_MEAN = {}
        hyp_list = combination_hyperparams(HYP_CONFIG)
        
        for idx, hyp in enumerate(hyp_list):
            
            EXP_ID = f"exp_{idx+1}"
            RESULT_MEAN[EXP_ID] = {}
            
            # sliding window
            if FRAMEWORK == "pyspark":
                train_feature = [col+"_Scaled" for col in INPUT_FEATURE]
                target_feature = TARGET_FEATURE + "_Scaled"
                train_data = train_data.sort('five_minute_interval')
                test_data = test_data.sort('five_minute_interval')

            elif FRAMEWORK == 'pandas':
                train_feature = INPUT_FEATURE
                target_feature = TARGET_FEATURE

                train_data.sort_values('five_minute_interval', inplace=True)
                test_data.sort_values('five_minute_interval', inplace=True)
    
            slider = SlidingWindow(FRAMEWORK)    
            train_window = slider.apply_window(train_data,
                                               training_col=train_feature,
                                               target_col=target_feature,
                                               input_window_size=hyp['input_sequence_length'],
                                               label_window_size=hyp['output_sequence_length'])

            test_window = slider.apply_window(test_data,
                                              training_col=train_feature,
                                              target_col=target_feature,
                                              input_window_size=hyp['input_sequence_length'],
                                              label_window_size=hyp['output_sequence_length'])
    
            # data set
            train_dataset = S1APDataset(train_window)
            test_dataset = S1APDataset(test_window)

            # model
            model = Model(MODEL_CONFIG)
            criterion = negative_log_likelihood
            optimizer = getattr(optim, hyp['optimizer'])(params=model.parameters(),
                                                         lr=hyp['learning_rate'])
            
            print(f"[INFO] Experiment {idx+1}/{len(hyp_list)} is Running ...")
            print("[INFO] Train Model")
            TIC = time.time()
            train(dataset=train_dataset,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  batch_size=hyp['batch_size'],
                  num_epoch=args.epoch)
            TOC = time.time()
            train_time = TOC - TIC
            
            print("[INFO] Evaluate Model")
            TIC = time.time()
            pred_mean, pred_std, ground_truth = evaluate(dataset=test_dataset,
                                                         model=model,
                                                         batch_size=hyp['batch_size'])
            TOC = time.time()
            inference_time = TOC - TIC
            
            
            pred_mean = np.vstack(pred_mean)
            pred_std = np.vstack(pred_std)
            ground_truth = np.vstack(ground_truth)
            
            # denorm scaler
            pred_mean = denormalization_pred(pred_mean,
                                             test_scaler,
                                             framework=find_item(DATA_CONFIG, 'framework'),
                                             column=TARGET_FEATURE)
            
            ground_truth = denormalization_pred(ground_truth,
                                                test_scaler,
                                                framework=find_item(DATA_CONFIG, 'framework'),
                                                column=TARGET_FEATURE)
            
            # denorm log scaler
            pred_mean = np.exp(pred_mean)
            ground_truth = np.exp(ground_truth)
            
            
            loss = {
                'MSE': round(float(mse_loss(pred_mean, ground_truth)), 2),
                'RMSE': round(float(np.sqrt(mse_loss(pred_mean, ground_truth))), 2),
                'MAE': round(float(mae_loss(pred_mean, ground_truth)), 2),
                'MAPE': round(float(mape_loss(pred_mean, ground_truth)), 2),
                # 'MEDIAN_ABS': round(float(median_abs_loss(pred_mean, ground_truth)), 2)
            }
            
            run_time = {'Train': round(train_time, 2), 'Inference': round(inference_time, 2)}
            
            RESULT_MEAN[EXP_ID]['hyp'] = hyp
            RESULT_MEAN[EXP_ID]['loss'] = loss
            RESULT_MEAN[EXP_ID]['time'] = run_time

            
        with open(os.path.join(ARTIFACT_DIR, f'{args.station}_240617_exp_1.yaml'), 'w') as outfile:
            yaml.dump(RESULT_MEAN, outfile, default_flow_style=False)
            
            
    else:
        # sliding window
        print("[INFO] Split by Window")
        if FRAMEWORK == "pyspark":
            train_feature = [col+"_Scaled" for col in INPUT_FEATURE]
            target_feature = TARGET_FEATURE + "_Scaled"

            train_data = train_data.sort('five_minute_interval')
            test_data = test_data.sort('five_minute_interval')

        elif FRAMEWORK == 'pandas':
            train_feature = INPUT_FEATURE
            target_feature = TARGET_FEATURE

            train_data.sort_values('five_minute_interval', inplace=True)
            test_data.sort_values('five_minute_interval', inplace=True)
            
        slider = SlidingWindow(FRAMEWORK)    
        train_window = slider.apply_window(train_data,
                                           training_col=train_feature,
                                           target_col=target_feature,
                                           input_window_size=find_item(HYP_CONFIG, 'input_sequence_length'),
                                           label_window_size=find_item(HYP_CONFIG, 'input_sequence_length'))

        test_window = slider.apply_window(test_data,
                                          training_col=train_feature,
                                          target_col=target_feature,
                                          input_window_size=find_item(HYP_CONFIG, 'input_sequence_length'),
                                           label_window_size=find_item(HYP_CONFIG, 'input_sequence_length'))
        
        # dataset
        train_dataset = S1APDataset(train_window)
        test_dataset = S1APDataset(test_window)
        
        # model
        model = Model(MODEL_CONFIG)
        criterion = negative_log_likelihood
        optimizer = getattr(optim, find_item(HYP_CONFIG, 'optimizer'))(params=model.parameters(),
                                                                       lr=find_item(HYP_CONFIG, 'learning_rate'))
        
        print("Train Model")
        train(dataset=train_dataset,
              model=model,
              criterion=criterion,
              optimizer = optimizer,
              batch_size = find_item(HYP_CONFIG, 'batch_size'),
              num_epoch = 2)
        
        print("Evaluate Model")
        pred_mean, pred_std, ground_truth = evaluate(dataset=test_dataset,
                                                     model=model,
                                                     batch_size=find_item(HYP_CONFIG, 'batch_size'))
        
        
        pred_mean = np.vstack(pred_mean)
        pred_std = np.vstack(pred_std)
        ground_truth = np.vstack(ground_truth)
            
        pred_mean = denormalization_pred(pred_mean,
                                         test_scaler,
                                         framework=find_item(DATA_CONFIG, 'framework'),
                                         column=TARGET_FEATURE)

        ground_truth = denormalization_pred(ground_truth,
                                            test_scaler,
                                            framework=find_item(DATA_CONFIG, 'framework'),
                                            column=TARGET_FEATURE)
            
            
        pred_mean = np.exp(pred_mean)
        ground_truth = np.exp(ground_truth)
            
        print(int(mse_loss(pred_mean, ground_truth)))

