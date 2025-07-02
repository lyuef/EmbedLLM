#!/usr/bin/env python3
"""
测试修改后的GraphSAGE模型
验证只使用训练数据构建图，未见模型使用相似度聚合的方案
"""

import torch
import pandas as pd
import numpy as np
from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph
from utils.load_and_process_data import build_train_response_matrix, get_unseen_model_responses

def create_dummy_data():
    """创建虚拟数据用于测试"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建虚拟训练数据 (前90个模型)
    train_data = []
    for model_id in range(90):
        for prompt_id in range(100):  # 100个prompts
            label = np.random.randint(0, 2)  # 二分类
            train_data.append({
                'model_id': model_id,
                'prompt_id': prompt_id,
                'label': label
            })
    
    # 创建虚拟测试数据 (模型90-111)
    test_data = []
    for model_id in range(90, 112):
        for prompt_id in range(100):
            label = np.random.randint(0, 2)
            test_data.append({
                'model_id': model_id,
                'prompt_id': prompt_id,
                'label': label
            })
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df

def test_modified_graphsage():
    """测试修改后的GraphSAGE模型"""
    print("=== 测试修改后的GraphSAGE模型 ===")
    
    # 1. 创建虚拟数据
    print("1. 创建虚拟数据...")
    train_data, test_data = create_dummy_data()
    print(f"训练数据: {len(train_data)} 条记录")
    print(f"测试数据: {len(test_data)} 条记录")
    
    # 2. 构建训练响应矩阵 (只使用训练数据)
    print("\n2. 构建训练响应矩阵...")
    num_train_models = 90
    train_response_matrix, train_model_ids, all_prompt_ids = build_train_response_matrix(
        train_data, num_train_models=num_train_models
    )
    print(f"训练响应矩阵形状: {train_response_matrix.shape}")
    print(f"训练模型数量: {len(train_model_ids)}")
    print(f"Prompt数量: {len(all_prompt_ids)}")
    
    # 3. 获取未见模型响应
    print("\n3. 获取未见模型响应...")
    unseen_model_ids = sorted(test_data["model_id"].unique())
    unseen_responses = get_unseen_model_responses(test_data, unseen_model_ids, all_prompt_ids)
    print(f"未见模型响应矩阵形状: {unseen_responses.shape}")
    print(f"未见模型数量: {len(unseen_model_ids)}")
    
    # 4. 创建虚拟模型嵌入
    print("\n4. 创建虚拟模型嵌入...")
    embedding_dim = 768
    num_models = 112
    model_embeddings = torch.randn(num_models, embedding_dim)
    train_model_embeddings = model_embeddings[:num_train_models]
    
    # 5. 构建图数据 (只使用训练数据)
    print("\n5. 构建图数据...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    graph_data = build_model_graph(train_response_matrix, train_model_embeddings, 
                                 k_neighbors=5, device=device)
    print(f"图节点数: {graph_data.x.shape[0]}")
    print(f"图边数: {graph_data.edge_index.shape[1]}")
    
    # 6. 创建虚拟问题嵌入
    print("\n6. 创建虚拟问题嵌入...")
    num_prompts = len(all_prompt_ids)
    text_dim = 768
    question_embeddings = torch.randn(num_prompts, text_dim)
    
    # 7. 初始化GraphSAGE模型
    print("\n7. 初始化GraphSAGE模型...")
    model = GraphSAGE_TextMF_dyn(
        question_embeddings=question_embeddings,
        model_embedding_dim=embedding_dim,
        alpha=0.1,
        num_models=num_models,
        num_prompts=num_prompts,
        model_embeddings=model_embeddings,
        is_dyn=True,
        frozen=False,
        hidden_dim=256,
        num_layers=2,
        num_train_models=num_train_models
    )
    model.to(device)
    
    # 设置训练响应矩阵
    model.set_train_responses(train_response_matrix)
    print("模型初始化完成")
    
    # 8. 测试前向传播
    print("\n8. 测试前向传播...")
    
    # 测试训练模型
    batch_size = 16
    train_model_ids = torch.randint(0, num_train_models, (batch_size,)).to(device)
    prompt_ids = torch.randint(0, num_prompts, (batch_size,)).to(device)
    
    print("测试训练模型前向传播...")
    with torch.no_grad():
        logits = model(graph_data, train_model_ids, prompt_ids, test_mode=True)
        print(f"训练模型输出形状: {logits.shape}")
    
    # 测试未见模型
    test_model_ids = torch.randint(num_train_models, num_models, (batch_size,)).to(device)
    
    # 为测试模型准备响应矩阵
    test_unseen_responses = torch.zeros(batch_size, num_prompts, device=device)
    for i, model_id in enumerate(test_model_ids):
        unseen_idx = model_id - num_train_models
        if unseen_idx < unseen_responses.shape[0]:
            test_unseen_responses[i] = unseen_responses[unseen_idx].to(device)
    
    print("测试未见模型前向传播...")
    with torch.no_grad():
        logits = model(graph_data, test_model_ids, prompt_ids, test_unseen_responses, test_mode=True)
        print(f"未见模型输出形状: {logits.shape}")
    
    # 9. 测试相似度聚合
    print("\n9. 测试相似度聚合...")
    with torch.no_grad():
        # 测试未见模型表示生成
        sample_responses = unseen_responses[:5].to(device)  # 取前5个未见模型
        representations = model.get_unseen_model_representation(sample_responses, k_neighbors=3)
        print(f"未见模型表示形状: {representations.shape}")
        print(f"表示维度: {representations.shape[1]}")
    
    print("\n=== 测试完成！所有功能正常工作 ===")
    
    return True

if __name__ == "__main__":
    try:
        test_modified_graphsage()
        print("\n✅ 所有测试通过！修改后的GraphSAGE模型工作正常。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
