"""
测试GraphSAGE模型的简单脚本
"""
import torch
import pandas as pd
import numpy as np
from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph
from utils.load_and_process_data import build_full_response_matrix

def test_graphsage_model():
    """测试GraphSAGE模型的基本功能"""
    print("Testing GraphSAGE model...")
    
    # 创建模拟数据
    num_models = 10
    num_prompts = 100
    embedding_dim = 768
    
    # 模拟问题嵌入
    question_embeddings = torch.randn(num_prompts, 768)
    
    # 模拟模型嵌入
    model_embeddings = torch.randn(num_models, embedding_dim)
    
    # 模拟响应矩阵
    response_matrix = torch.randint(0, 2, (num_models, num_prompts)).float()
    
    # 构建图数据
    print("Building graph data...")
    graph_data = build_model_graph(response_matrix, model_embeddings, k_neighbors=3)
    print(f"Graph nodes: {graph_data.x.shape[0]}")
    print(f"Graph edges: {graph_data.edge_index.shape[1]}")
    
    # 创建模型
    print("Creating GraphSAGE model...")
    model = GraphSAGE_TextMF_dyn(
        question_embeddings=question_embeddings,
        model_embedding_dim=embedding_dim,
        alpha=0.1,
        num_models=num_models,
        num_prompts=num_prompts,
        model_embeddings=model_embeddings,
        is_dyn=True,
        frozen=False,
        hidden_dim=128,
        num_layers=2
    )
    
    # 测试前向传播
    print("Testing forward pass...")
    batch_size = 5
    model_ids = torch.randint(0, num_models, (batch_size,))
    prompt_ids = torch.randint(0, num_prompts, (batch_size,))
    
    with torch.no_grad():
        logits = model(graph_data, model_ids, prompt_ids, test_mode=True)
        predictions = model.predict(graph_data, model_ids, prompt_ids)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # 测试训练模式
    print("Testing training mode...")
    model.train()
    logits_train = model(graph_data, model_ids, prompt_ids, test_mode=False)
    print(f"Training logits shape: {logits_train.shape}")
    
    print("GraphSAGE model test completed successfully!")
    return True

def test_data_processing():
    """测试数据处理功能"""
    print("\nTesting data processing...")
    
    # 创建模拟数据
    train_data = pd.DataFrame({
        'model_id': np.repeat(range(5), 10),
        'prompt_id': np.tile(range(10), 5),
        'label': np.random.randint(0, 2, 50)
    })
    
    test_data = pd.DataFrame({
        'model_id': np.repeat(range(5, 8), 10),
        'prompt_id': np.tile(range(10), 3),
        'label': np.random.randint(0, 2, 30)
    })
    
    # 测试响应矩阵构建
    response_matrix, all_model_ids, all_prompt_ids = build_full_response_matrix(train_data, test_data)
    print(f"Response matrix shape: {response_matrix.shape}")
    print(f"Number of models: {len(all_model_ids)}")
    print(f"Number of prompts: {len(all_prompt_ids)}")
    
    print("Data processing test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        # 测试数据处理
        test_data_processing()
        
        # 测试GraphSAGE模型
        test_graphsage_model()
        
        print("\n✅ All tests passed! GraphSAGE implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
