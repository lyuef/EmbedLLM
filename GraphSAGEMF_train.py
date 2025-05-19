from models.Embedllm_dynamic import TextMF_dyn,TextMF_dyn_ML
from utils import train as tr , parser_maker as pm ,load_and_process_data as lpd , load_model as lm 
import torch 
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout
from datetime import datetime
from torch_geometric.data import Data
import torch.nn.functional as F

def build_cograph(responses):
    """build cograph from responses"""
    n_models = responses.shape[0]
    
    # likely matrix
    agreement = torch.mm(responses.float(), responses.float().T)
    total = responses.shape[1]
    likely = agreement / (total - agreement + 1e-8)
    likely_norm = F.softmax(likely, dim=1)
    
    src = torch.repeat_interleave(torch.arange(n_models), n_models)
    dst = torch.arange(n_models).repeat(n_models)
    edge_index = torch.stack([src, dst], dim=0)
    
    return Data(
        x=responses,
        edge_index=edge_index,
        edge_weight=likely_norm.flatten()
    )
def main() :
    parser = pm.parser_make()
    args = parser.parse_args()

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

