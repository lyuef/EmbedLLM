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
    
    print(train_data.columns)
    
    print("原始数据列名:", train_data.columns.tolist())

    # 检查每个 model_id 是否与所有 prompt_id 组合存在
    unique_models = train_data['model_id'].unique()
    unique_prompts = train_data['prompt_id'].unique()

    print(len(unique_models),len(unique_prompts))
    
    # 生成所有可能的组合（笛卡尔积）
    cross_combinations = pd.MultiIndex.from_product(
        [unique_models, unique_prompts],
        names=['model_id', 'prompt_id']
    ).to_frame(index=False)
    
    # 提取实际存在的组合
    existing_combinations = train_data[['model_id', 'prompt_id']].drop_duplicates()
    
    # 通过左连接找出缺失的组合
    missing_combinations = cross_combinations.merge(
        existing_combinations,
        on=['model_id', 'prompt_id'],
        how='left',
        indicator=True
    )
    missing_combinations = missing_combinations[missing_combinations['_merge'] == 'left_only']
    
    # 输出结果
    if missing_combinations.empty:
        print("\n所有 model_id 和 prompt_id 的组合均存在。")
    else:
        print("\n以下组合缺失:")
        print(missing_combinations[['model_id', 'prompt_id']].to_string(index=False))

    
