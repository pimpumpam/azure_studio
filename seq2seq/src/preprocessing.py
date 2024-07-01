import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from datetime import datetime

import sklearn
import sklearn.preprocessing

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType, DoubleType
from pyspark.sql.functions import col, min, max, udf, isnan, isnull, mean, stddev, hour, month, dayofweek

import torch
from torch.utils.data import Dataset

from src.utils import find_item
from src.dataLoader import DataLoad, convert_dtype

# Global variable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
CONFIG_DIR = os.path.join(ROOT, 'toby', 'configs')


class MissingValueHandler():
            
    def count_missing_value(self, data):
        """
        각 컬럼 별 결측치 개수 파악
        
        parameter
        ----------
        data(pyspark.sql.dataframe.DataFrame): 결측치 개수를 파악할 데이터
        
        return
        ----------
        counter(dict): 컬럼 별 결측치 개수가 key와 value로 구성된 dictionary
        
        """
        counter = {}
        
        for column in data.columns:
            try:
                cnt = data.filter(isnull(col(column)) | isnan(col(column))).count()
            except:
                cnt = data.filter(isnull(col(column))).count()

            counter[column] = cnt
            
        return counter
            
        
    def fillna_numeric_column(self, data, fill_value=0):
        """
        Numeric 컬럼의 결측치를 채워 넣음
        
        parameter
        ----------
        data(pyspark.sql.dataframe.DataFrame): 결측치를 채워 넣을 데이터
        fill_value(str, int, float): 결측치를 채워 넣을 값.
            - str 형태로 주어진 경우는 각 방법론에 맞는 값으로 채워 넣음
                'mean': 컬럼의 평균 값으로 결측치 대체
            - int, float 형태로 주어진 경우는 해당 값으로 채워 넣음.
            
        return
        ----------
        filled_data(pyspark.sql.dataframe.DataFrame): 결측치 처리가 완료 된 데이터
        
        """

        for column in data.columns:
            if data.schema[column].dataType.simpleString() in ['float', 'double', 'int']:
                if isinstance(fill_value, (int, float)):
                    filled_data = data.fillna({column: fill_value})
                    
                elif fill_value == 'mean':
                    mean_value = data.select(mean(col(column))).collect()[0][0]
                    filled_data = data.na.fill({column: mean_value})
                
        return filled_data
    
    
    def fillna_character_column(self, data, fill_value):
        """
        String, Timestamp 컬럼의 결측치를 채워 넣음
        
        parameter
        ----------
        data(pyspark.sql.dataframe.DataFrame): 결측치를 채워 넣을 데이터
        fill_value(str): 결측치를 채워 넣을 값.
            
        return
        ----------
        filled_data(pyspark.sql.dataframe.DataFrame): 결측치 처리가 완료 된 데이터
        
        """
        
        for column in data.columns:
            if data.schema[column].dataType.simpleString() in ['string', 'timestamp']:
                filled_data = data.fillna({column: fill_value})
                
        return filled_data
    

class Scaler():
    
    def __init__(self, scaler_kind):
        """
        initializer
        
        parameter
        ----------
        scaler_kind(str): pyspark.ml.feature에서 지원하는 scaling 종류. MinMaxScaler, StandardScaler 등 사용 가능.
        
        """
        self.scaler = getattr(pyspark.ml.feature, scaler_kind)
        

    def _get_column_min_max_value(self, data, columns):
        
        min_dict = {}
        max_dict = {}
        
        for column in columns:
            min_val = data.agg(min(column).alias(column)).collect()[0][0]
            max_val = data.agg(max(column).alias(column)).collect()[0][0]
            
            min_dict[column] = min_val
            max_dict[column] = max_val
            
        return min_dict, max_dict
    
    def _get_column_avg_std_value(self, data, columns):
        
        avg_dict = {}
        std_dict = {}
        
        for column in columns:
            avg_val = data.agg(mean(column).alias(column)).collect()[0][0]
            std_val = data.agg(stddev(column).alias(column)).collect()[0][0]
            
            avg_dict[column] = avg_val
            std_dict[column] = std_val
            
        return avg_dict, std_dict
    
    
    def fit_transform(self, data, columns):
        """
        Pyspark DataFrame에 scaler 적용
        
        parameter
        ----------
        data(pyspark.sql.dataframe.DataFrame): Scaler를 적용할 data
        columns(list): Scaler를 적용할 컬럼으로 구성된 list
        
        return
        ----------
        scaled_data(pyspark.sql.dataframe.DataFrame): Scaling이 된 data
        
        """
        
        assert isinstance(columns, list), "Type of argument \"columns\" must be the list"
        
        if self.scaler.__name__ == 'MinMaxScaler':
            self.min_dict, self.max_dict = self._get_column_min_max_value(data, columns)
        
        elif self.scaler.__name__ == 'StandardScaler':
            self.avg_dict, self.std_dict = self._get_column_avg_std_value(data, columns)
        
        
        assemblers = [VectorAssembler(inputCols=[col], outputCol=col+"_Vec") for col in columns]
        scalers = [self.scaler(inputCol=col+"_Vec", outputCol=col+"_Scaled") for col in columns]
        
        pipeline = Pipeline(stages=assemblers+scalers)
        
        self.scaled_model = pipeline.fit(data)
        scaled_data = self.scaled_model.transform(data)
        
        return scaled_data
    
    def denseVector_to_float(self, data, columns):
        
        def converter(dense_vec):
            return float(dense_vec[0])
        
        float_converter = udf(converter, FloatType())
        
        denseVec_cols = [col+"_Scaled" for col in columns]
        for col in denseVec_cols:
            data = data.withColumn(col, float_converter(data[col]))
            
        return data
    
    def inverse_transform(self, data, columns):
        
        if self.scaler.__name__ == "MinMaxScaler":
            for column in columns:
                new_column = column.replace("_Scaled", "") if "_Scaled" in column else column
                data = data.withColumn(new_column+"_InvScaled",
                                       col(column)*(self.max_dict[new_column]-self.min_dict[new_column])+self.min_dict[new_column])
            
            return data
            
        elif self.scaler.__name__ == "StandardScaler":
            for column in columns:
                new_column = column.replace("_Scaled", "") if "_Scaled" in column else column
                data = data.withColumn(new_column+"_InvScaled",
                                       (col(column)*self.std_dict[new_column])+self.avg_dict[new_column])
            
            return data
    
    
class MultiColumnScaler():
    
    def __init__(self, scaler_kind):
        self.scaler_kind = scaler_kind
        self.scaler = getattr(sklearn.preprocessing, scaler_kind)()
        
    def transform(self, df, columns, inplace=False):
        if isinstance(df, pyspark.sql.dataframe.DataFrame):
            df = df.toPandas()
        
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = self.scaler.transform(df[columns])
            
            return df
        
        else:
            df[columns] = self.scaler.transform(df[columns])
            
    def fit_transform(self, df, columns, inplace=False):
        if isinstance(df, pyspark.sql.dataframe.DataFrame):
            df = df.toPandas()
            
        if not isinstance(columns, list):
            columns = [columns]
        
        if not inplace:
            df = df.copy()
            df[columns] = self.scaler.fit_transform(df[columns])
            
            return df
        
        else:
            df[columns] = self.scaler.tif_transform(df[columns])
            
    def inverse_transform(self, df, columns, inplace=False):
        if isinstance(df, pyspark.sql.dataframe.DataFrame):
            df = df.toPandas()
            
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = self.scaler.inverse_transform(df[columns])
            
            return df
        
        else:
            df[columns] = self.scaler.inverse_transform(df[columns])
    
    
class S1APDataset(Dataset):
    def __init__(self, data):
        """
        initializer
        
        parameter
        ----------
        data(list, numpy.ndarray): 모델의 input과 label 데이터가 쌍으로 구성된 iterable한 데이터
        
        return
        ----------
        
        """
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_feature, label_feature = self.data[idx]
        sample = {'inputs': torch.tensor(input_feature, dtype=torch.float32),
                  'labels': torch.tensor(label_feature, dtype=torch.float32)}
        
        return sample
    

    
class SlidingWindow():
    def __init__(self, framework):
        self.framework = framework
        
    def _pyspark_slider(self, data, training_col, target_col, input_window_size, label_window_size):
        """
        Pyspark DataFrame에 Window를 돌려가며 데이터 추출

        parameter
        ----------
        data(pyspark.DataFrame)
        training_col(list): 학습에 사용할 column 명
        target_col(str): label 정보가 되는 column 명
        input_window_size(int): 모델의 input 데이터에 적용할 window 크기
        label_window_size(int): 모델의 label 데이터에 적용할 window 크기

        return
        ----------
        result(list): Window 크기로 분할 된 데이터 셋 

        """
        result = []
        rdd = data.rdd.zipWithIndex()
        index_rows = rdd.collect()

        column_index = {col: i for i, col in enumerate(rdd.first()[0].__fields__)}
        input_index = [column_index[col] for col in training_col]
        label_index = column_index[target_col]

        for i in range(len(index_rows)-input_window_size-(label_window_size+1)+2):
            input_feature = [
                [row[0][index] for index in input_index] for row in index_rows[i:i+input_window_size]
            ]

            label_feature = [
                index_rows[i+input_window_size-1+j][0][label_index] for j in range(label_window_size+1)
            ]

            result.append((input_feature, label_feature))
        
        return result
    
    def _pandas_slider(self, data, training_col, target_col, input_window_size, label_window_size):
        """
        Pyspark DataFrame에 Window를 돌려가며 데이터 추출

        parameter
        ----------
        data(pyspark.DataFrame)
        training_col(list): 학습에 사용할 column 명
        target_col(str): label 정보가 되는 column 명
        input_window_size(int): 모델의 input 데이터에 적용할 window 크기
        label_window_size(int): 모델의 label 데이터에 적용할 window 크기

        return
        ----------
        result(list): Window 크기로 분할 된 데이터 셋 

        """
        
        result = []

        for i in tqdm(range(len(data)-input_window_size-(label_window_size+1)+2)):
            input_feature = data[training_col].iloc[i:i+input_window_size].values.tolist()
            label_feature = data[target_col].iloc[i+input_window_size-1:i+input_window_size-1+label_window_size].values.tolist()
            
            result.append((input_feature, label_feature))
            
        return result
    
    def apply_window(self, data, training_col, target_col, input_window_size, label_window_size):
        """
        Pyspark DataFrame에 Window를 돌려가며 데이터 추출

        parameter
        ----------
        data(pyspark.DataFrame)
        training_col(list): 학습에 사용할 column 명
        target_col(str): label 정보가 되는 column 명
        input_window_size(int): 모델의 input 데이터에 적용할 window 크기
        label_window_size(int): 모델의 label 데이터에 적용할 window 크기

        return
        ----------
        result(list): Window 크기로 분할 된 데이터 셋 

        """
        
        if self.framework.lower() == 'pyspark':
            
            return self._pyspark_slider(data, training_col, target_col, input_window_size, label_window_size)
        
        elif self.framework.lower() == 'pandas':
            if isinstance(data, pyspark.sql.dataframe.DataFrame):
                data = data.toPandas()
                
            return self._pandas_slider(data, training_col, target_col, input_window_size, label_window_size)
        
        
class LogScaler():
    
    def __init__(self, framework):
        self.framework = framework
        
        
    def transform(self, data, columns):
        if self.framework == 'pyspark':
            print("기능 개발 필요")
            
        elif self.framework == 'pandas':
            if isinstance(data, pyspark.sql.dataframe.DataFrame):
                data = data.toPandas()
                
            data = data.replace(0, 1)
            data[columns] = np.log(data[columns])
            
            return data
        
    def inverse_transform(self, data, columns):
        if self.framework == 'pyspark':
            print("기능 개발 필요")
            
        elif self.framework == 'pandas':
            if isinstance(data, pyspark.sql.dataframe.DataFrame):
                data = data.toPandas()
                
            data[columns] = np.exp(data[columns])
            
            return data
        


class TrainTestSpliter():
    
    def __init__(self, framework):
        self.framework = framework
        
    def split_by_time(self, data, time_column_name, train_start_time, train_end_time, test_start_time, test_end_time):
        """
        시간을 기준으로 Train/Test 데이터 셋 분할

        parameter
        ----------
        data(pyspark.sql.dataframe.DataFrame): 분할 하기 전 전체 데이터
        time_column_name(str): 분할 기준이 되는 timestamp 타입의 컬럼명
        train_start_time(str): Train 데이터를 위한 시작 날짜
        train_end_time(str): Train 데이터를 위한 종료 날짜
        test_start_time(str): Test 데이터를 위한 시작 날짜
        test_end_time(str): Test 데이터를 위한 종료 날짜

        return
        ----------
        train_set(pyspark.sql.dataframe.DataFrame): 날짜 기준 분할 된 Train 데이터 셋
        test_set(pyspark.sql.dataframe.DataFrame): 날짜 기준 분할 된 Test 데이터 셋

        """
        
        if self.framework == 'pyspark':
            train_set = data.filter((col(time_column_name)>=train_start_time) & (col(time_column_name)<=train_end_time))
            test_set = data.filter((col(time_column_name)>=test_start_time) & (col(time_column_name)<=test_end_time))
            
        elif self.framework == 'pandas':
            
            def time_converter(time):
                return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            
            train_start_time = time_converter(train_start_time)
            train_end_time = time_converter(train_end_time)
            test_start_time = time_converter(test_start_time)
            test_end_time = time_converter(test_end_time)
            
            train_set = data.loc[(data[time_column_name]>train_start_time)&(data[time_column_name]<train_end_time), :]
            test_set = data.loc[(data[time_column_name]>test_start_time)&(data[time_column_name]<test_end_time), :]
    
        return train_set, test_set


def gen_time_covariate(data, time_column_name, timeunit='hour'):
    """
    시간 정보를 통해 covaritate 생성
    
    parameter
    ----------
    data(pyspark.sql.dataframe.DataFrame): Covariate를 구하려고 하는 data
    time_column_name(str): Covariate를 구하기 위한 대상 timestamp 컬럼
    timeunit(str): Covariate 산출을 위한 시간 단위
    
    return
    ----------
    data(pyspark.sql.dataframe.DataFrame): "{timeunit}_cov"라는 컬럼 값이 추가 된 data
    
    """
    
    def gen_zscore(val):
        return(val-avg) / std
        # return stats.zscore(val)
        
    
    sub_data = data.select(time_column_name)
    sub_data = sub_data.withColumn(f"{timeunit}_cov", eval(timeunit)(time_column_name))
    
    time_list = sub_data.select(f"{timeunit}_cov").rdd.flatMap(lambda x: x).collect()
    
    cov = stats.zscore(time_list)
    avg = sub_data.select(mean(col(f"{timeunit}_cov"))).collect()[0][0]
    std = sub_data.select(stddev(col(f"{timeunit}_cov"))).collect()[0][0]
    
    udf_zscore = udf(gen_zscore, DoubleType())
    sub_data = sub_data.withColumn(f"{timeunit}_cov", udf_zscore(col(f"{timeunit}_cov")))
    
    return data.join(sub_data, on=time_column_name, how='left')


def split_train_test_by_time(data, time_column_name, train_start_time, train_end_time, test_start_time, test_end_time):
    """
    시간을 기준으로 Train/Test 데이터 셋 분할
    
    parameter
    ----------
    data(pyspark.sql.dataframe.DataFrame): 분할 하기 전 전체 데이터
    time_column_name(str): 분할 기준이 되는 timestamp 타입의 컬럼명
    train_start_time(str): Train 데이터를 위한 시작 날짜
    train_end_time(str): Train 데이터를 위한 종료 날짜
    test_start_time(str): Test 데이터를 위한 시작 날짜
    test_end_time(str): Test 데이터를 위한 종료 날짜
    
    return
    ----------
    train_set(pyspark.sql.dataframe.DataFrame): 날짜 기준 분할 된 Train 데이터 셋
    test_set(pyspark.sql.dataframe.DataFrame): 날짜 기준 분할 된 Test 데이터 셋
    
    """
    
    train_set = data.filter((col(time_column_name)>=train_start_time) & (col(time_column_name)<=train_end_time))
    test_set = data.filter((col(time_column_name)>=test_start_time) & (col(time_column_name)<=test_end_time))
    
    return train_set, test_set
        