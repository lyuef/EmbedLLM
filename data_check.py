from utils import train as tr , parser_maker as pm ,load_and_process_data as lpd , load_model as lm 
import torch 
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout

if __name__ == "__main__" :
    parser = pm.parser_make()
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    
    pd.set_option('display.max_rows', None)
    
    train_model_id_counts = train_data.groupby("model_id").size()
    test_model_id_counts = test_data.groupby('model_id').size()
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print(train_model_id_counts)
        print(test_model_id_counts)
    with open(args.output_save_path, "w", encoding="utf-8") as f:
        f.write(captured_output.getvalue())
    


    
