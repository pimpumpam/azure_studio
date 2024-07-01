import os
import yaml
import argparse
import numpy as np
from pathlib import Path

from sklearn.metrics import median_absolute_error

import torch
import torch.nn as nn

from src.utils import find_item

# Global variable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
CONFIG_DIR = os.path.join(ROOT, 'toby', 'configs')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNLayers(nn.Module):
    def __init__(self, model_config):
        super(RNNLayers, self).__init__()
        self.model = nn.ModuleList()
        
        for module, arg in model_config:
            layer = eval(module)(*arg)
            try:
                layer.flatten_parameters()
            except:
                pass
            self.model.append(layer)
    
    def forward(self, x, hidden_state, cell_state):
        for layer in self.model:
            x, (hidden_state, cell_state) = layer(x, (hidden_state, cell_state))
        
        return x, hidden_state, cell_state

    
class LinearLayers(nn.Module):
    def __init__(self, model_config):
        super(LinearLayers, self).__init__()
        self.model = nn.ModuleList()
        
        for module, arg in model_config:
            layer = eval(module)(*arg)
            self.model.append(layer)
            
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x
    
    
class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        
        self.num_layers = find_item(model_config, 'num_layers')
        self.output_size = find_item(model_config, 'output_size')
        self.hidden_size = find_item(model_config, 'rnn_hidden_size')
        
        self.encoder = RNNLayers(find_item(model_config, 'encoder'))
        self.decoder = RNNLayers(find_item(model_config, 'decoder'))
        self.mean_squeezer = LinearLayers(find_item(model_config, 'squeezer'))
        self.mean_predictor = LinearLayers(find_item(model_config, 'predictor'))
        self.std_squeezer = LinearLayers(find_item(model_config, 'squeezer'))
        self.std_predictor = LinearLayers(find_item(model_config, 'predictor'))
        self.softplus = nn.Softplus()
        
        
    def forward(self, x, y, teacher_forcing=0.5):
        batch_size, seq_len = x.shape[0], y.shape[1]-1
        
        output_collection = torch.zeros(batch_size, seq_len, self.output_size)
        mu_collection = torch.zeros(batch_size, seq_len, self.output_size) 
        sigma_collection = torch.zeros(batch_size, seq_len, self.output_size) 
        
        # encoder
        h0 = create_init_state(batch_size, self.num_layers, self.hidden_size, device=DEVICE)
        c0 = create_init_state(batch_size, self.num_layers, self.hidden_size, device=DEVICE)
        _, encoder_hidden, encoder_cell = self.encoder(x, h0, c0)
        
        
        # decoder
        decoder_x = y[:, 0].view(-1, 1, 1) # batch x 1 x 1
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        for idx in range(1, seq_len+1):
            decoder_y, decoder_hidden, decoder_cell = self.decoder(decoder_x, decoder_hidden, decoder_cell)
            
            mu = self.mean_predictor(decoder_y.squeeze())
            sigma = self.std_predictor(decoder_y.squeeze())
            sigma = self.softplus(sigma)
                        
            
            # output_collection[:, idx-1, :] = decoder_y
            mu_collection[:, idx-1, :] = mu # batch x 1
            sigma_collection[:, idx-1, :] = sigma # batch x 1
            
        
            if np.random.random() < teacher_forcing:
                decoder_x = y[:, idx].view(-1, 1, 1)
            else:
                decoder_x = mu.unsqueeze(-1).clone().detach()
        
        
        return output_collection, mu_collection, sigma_collection

    
def negative_log_loss(true, mu, sigma):
    
    try:
        batch_size, seq_len, num_feat = true.size()
    except:
        true = true.unsqueeze(-1)
    
    distribution = torch.distributions.normal.Normal(mu, sigma)
    # likelihood = torch.sum(distribution.log_prob(true))
    likelihood = torch.mean(distribution.log_prob(true))
    
    return -likelihood


def negative_log_likelihood(true, mu, sigma):
    try:
        batch_size, seq_len, num_feat = true.size()
    except:
        true = true.unsqueeze(-1)
    
    return 0.5*torch.mean(sigma + torch.exp(-sigma)*(true-mu)**2)


def mse_loss(pred, truth):
    return np.mean((pred-truth)**2)

def mae_loss(pred, truth):
    return np.mean(np.abs(pred-truth))

def mape_loss(pred, truth):
    return np.mean(np.abs((pred-truth) / truth)) * 100

def median_abs_loss(pred, truth):
    return median_absolute_error(truth, pred)


def create_init_state(batch_size, num_layers, hidden_dims, device=None):
    
    return torch.zeros(num_layers, batch_size, hidden_dims).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model' , type=str, default='deepAR_lstm.yaml', help='모델 아키텍처 config 파일명')
    
    args = parser.parse_args()
    
    with open(os.path.join(CONFIG_DIR, 'model', args.model)) as y:
        MODEL_CONFIG = yaml.load(y, Loader=yaml.FullLoader)
        
    model = Model(MODEL_CONFIG)
    
    print(model)
        
    