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

def main() :
    parser = pm.parser_make()
    args = parser.parse_args()

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

if __name__ == "__main_" : 
    main()
