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
    
    total_prompt_count = train_data["prompt_id"].nunique()

    print(f"总共有 {total_prompt_count} 个唯一的 prompt_id。")

    # print(train_data.columns)
    print(test_data.columns)

    # print(train_data.head())
    print(test_data.head())

    # captured_output = io.StringIO()
    # with redirect_stdout(captured_output):
        
    # with open(args.output_save_path, "w", encoding="utf-8") as f:
    #     f.write(captured_output.getvalue())
    


    
