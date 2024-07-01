import os
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import DEVICE, mse_loss
from src.utils import TQDM_BAR_FORMAT, find_item

# Global variable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
CONFIG_DIR = os.path.join(ROOT, 'toby', 'configs')


def evaluate(dataset, model, batch_size):
    
    # pred_output_result = []
    pred_mean = []
    pred_std = []
    ground_truth = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model.to(DEVICE)
    model.eval()
    
    print("[INFO] Initialize Test Model")
    print(('%20s'*2)%('Step', 'GPU Meomry'))
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader), bar_format=TQDM_BAR_FORMAT) as tq:
            for step, feat in enumerate(tq):

                input_feat = feat['inputs'].to(DEVICE)
                label_feat = feat['labels'].to(DEVICE)

                output_collection, mu_collection, sigma_collection = model.forward(input_feat,
                                                                                   label_feat,
                                                                                   teacher_forcing=0.0)
                
                # pred_output_result.append(output_collection.squeeze().detach().cpu().numpy())
                pred_mean.append(mu_collection.squeeze().detach().cpu().numpy())
                pred_std.append(sigma_collection.squeeze().detach().cpu().numpy())
                ground_truth.append(label_feat[:, 1:].detach().cpu().numpy())
                
    return pred_mean, pred_std, ground_truth