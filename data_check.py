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
    
    test_models = test_data['model_id'].unique()
    test_prompts = test_data['prompt_id'].unique()
    print(f'train models : {min(unique_models)},{max(unique_models)}')
    print(f'train prompts : {min(unique_prompts)},{max(unique_prompts)}')
    print(f'test models : {min(test_models)},{max(test_models)}')
    print(f'test promtps : {min(test_prompts)},{max(test_prompts)}')
    
    # 检查数据集大小和重合情况
    print("\n" + "="*50)
    print("数据集大小和重合情况分析")
    print("="*50)
    
    # 数据集大小
    print(f"Train data 大小: {len(train_data)} 条记录")
    print(f"Test data 大小: {len(test_data)} 条记录")
    
    # 模型重合情况
    train_models_set = set(unique_models)
    test_models_set = set(test_models)
    model_overlap = train_models_set.intersection(test_models_set)
    
    print(f"\nTrain data 中的模型数量: {len(train_models_set)}")
    print(f"Test data 中的模型数量: {len(test_models_set)}")
    print(f"重合的模型数量: {len(model_overlap)}")
    if len(model_overlap) > 0:
        print(f"重合的模型ID: {sorted(list(model_overlap))}")
    
    # 问题重合情况
    train_prompts_set = set(unique_prompts)
    test_prompts_set = set(test_prompts)
    prompt_overlap = train_prompts_set.intersection(test_prompts_set)
    
    print(f"\nTrain data 中的问题数量: {len(train_prompts_set)}")
    print(f"Test data 中的问题数量: {len(test_prompts_set)}")
    print(f"重合的问题数量: {len(prompt_overlap)}")
    if len(prompt_overlap) > 0:
        print(f"重合的问题ID范围: {min(prompt_overlap)} - {max(prompt_overlap)}")
    
    # (model_id, prompt_id) 组合重合情况
    train_combinations = set(zip(train_data['model_id'], train_data['prompt_id']))
    test_combinations = set(zip(test_data['model_id'], test_data['prompt_id']))
    combination_overlap = train_combinations.intersection(test_combinations)
    
    print(f"\nTrain data 中的 (model_id, prompt_id) 组合数量: {len(train_combinations)}")
    print(f"Test data 中的 (model_id, prompt_id) 组合数量: {len(test_combinations)}")
    print(f"重合的组合数量: {len(combination_overlap)}")
    
    # 重合比例
    if len(train_combinations) > 0:
        overlap_ratio_train = len(combination_overlap) / len(train_combinations) * 100
        print(f"重合组合占 train data 的比例: {overlap_ratio_train:.2f}%")
    
    if len(test_combinations) > 0:
        overlap_ratio_test = len(combination_overlap) / len(test_combinations) * 100
        print(f"重合组合占 test data 的比例: {overlap_ratio_test:.2f}%")
    
    print("="*50)

    
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

    # 生成所有可能的组合（笛卡尔积）
    cross_combinations = pd.MultiIndex.from_product(
        [test_models, test_prompts],
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
        # print(missing_combinations[['model_id', 'prompt_id']].to_string(index=False))#
        print(len(missing_combinations))
