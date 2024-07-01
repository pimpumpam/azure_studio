import os
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataLoader import DataLoad, convert_dtype
from src.preprocessing import S1APDataset, Scaler
from src.model import DEVICE, Model, negative_log_loss, negative_log_likelihood
from src.utils import TQDM_BAR_FORMAT, find_item

# Global variable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
CONFIG_DIR = os.path.join(ROOT, 'toby', 'configs')

def train(dataset, model, criterion, optimizer, batch_size, num_epoch):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.to(DEVICE)
    
    print("[INFO] Initialize Train Model")
    print(('%20s'*3)%('Epoch', 'GPU Meomry', 'Loss'))
    
    for epoch in range(num_epoch):
        with tqdm(dataloader, total=len(dataloader), bar_format=TQDM_BAR_FORMAT) as tq:
            for step, feat in enumerate(tq):
                
                input_feat = feat['inputs'].to(DEVICE)
                label_feat = feat['labels'].to(DEVICE)
                
                optimizer.zero_grad()
                _, mu_collection, sigma_collection = model.forward(input_feat,
                                                                   label_feat,
                                                                   teacher_forcing=0.5)
                
                mu_collection = mu_collection.to(DEVICE)
                sigma_collection = sigma_collection.to(DEVICE)
                
                target = torch.roll(label_feat, -1, 1)[:, :-1]
                loss = criterion(target, mu_collection, sigma_collection)
                loss.backward()
                optimizer.step()
                
                mem = f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.3g}G"
                tq.set_description(('%20s'*3)%(f"{epoch+1}/{num_epoch}", mem, f"{loss.item():.4}"))
