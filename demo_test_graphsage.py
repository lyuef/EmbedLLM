"""
GraphSAGE + Matrix Factorization 演示测试脚本
快速测试功能是否正常（只运行5个epoch）
"""
import argparse
import torch
import pandas as pd
import numpy as np
import random
from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph
from utils.load_and_process_data import build_full_response_matrix, get_unseen_model_responses
from utils.evaluate import evaluate_graphsage
from torch.optim import Adam
from torch import nn
from datetime import datetime
import os
import sys
from tqdm import tqdm

def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_random_data_loaders(train_data, test_data, batch_size=64, random_seed=None):
    """创建随机打乱的数据加载器"""
    from torch.utils.data import Dataset, DataLoader
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # 合并所有数据以获取完整的模型列表
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    all_model_ids = sorted(all_data["model_id"].unique())
    
    # 随机打乱模型ID
    shuffled_model_ids = np.random.permutation(all_model_ids)
    
    # 分割：前90个作为训练模型，后22个作为测试模型
    train_model_ids = shuffled_model_ids[:90]
    test_model_ids = shuffled_model_ids[90:112]
    
    print(f"训练模型ID (前5个): {train_model_ids[:5]}")
    print(f"测试模型ID (前5个): {test_model_ids[:5]}")
    
    # 过滤数据
    train_data_filtered = train_data[train_data["model_id"].isin(train_model_ids)]
    test_data_filtered = test_data[test_data["model_id"].isin(test_model_ids)]
    
    # 创建ID映射
    id_mapping = {original_id: new_id for new_id, original_id in enumerate(shuffled_model_ids)}
    
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    
    class CustomDataset(Dataset):
        def __init__(self, data, id_mapping):
            mapped_model_ids = [id_mapping[mid] for mid in data["model_id"].values]
            self.models = torch.tensor(mapped_model_ids, dtype=torch.int64)
            self.prompts = torch.tensor(data["prompt_id"].to_numpy(), dtype=torch.int64)
            self.labels = torch.tensor(data["label"].to_numpy(), dtype=torch.int64)

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return self.models[index], self.prompts[index], self.labels[index]

    train_dataset = CustomDataset(train_data_filtered, id_mapping)
    test_dataset = CustomDataset(test_data_filtered, id_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, shuffled_model_ids, id_mapping

def demo_test():
    """运行演示测试"""
    print("🚀 GraphSAGE + Matrix Factorization 演示测试")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 检查文件
    required_files = [
        "data/train.csv",
        "data/test.csv",
        "data/question_embeddings.pth",
        "data/model_embeddings_static.pth"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    try:
        print("📊 加载数据...")
        train_data = pd.read_csv("data/train.csv")
        test_data = pd.read_csv("data/test.csv")
        question_embeddings = torch.load("data/question_embeddings.pth", weights_only=True)
        model_embeddings = torch.load("data/model_embeddings_static.pth", weights_only=True)
        
        num_prompts = question_embeddings.shape[0]
        num_models = 112
        
        print(f"   - 模型数量: {num_models}")
        print(f"   - 问题数量: {num_prompts}")
        
        # 设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - 使用设备: {device}")
        
        # 创建数据加载器
        print("\n🔀 创建随机数据加载器...")
        train_loader, test_loader, shuffled_model_ids, id_mapping = create_random_data_loaders(
            train_data, test_data, batch_size=64, random_seed=42
        )
        
        # 构建响应矩阵
        print("\n🔗 构建响应矩阵...")
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        all_prompt_ids = sorted(all_data["prompt_id"].unique())
        response_matrix = torch.zeros(num_models, len(all_prompt_ids), dtype=torch.float32)
        
        for _, row in all_data.iterrows():
            if row["model_id"] in shuffled_model_ids:
                model_idx = id_mapping[row["model_id"]]
                prompt_idx = all_prompt_ids.index(row["prompt_id"])
                response_matrix[model_idx, prompt_idx] = row["label"]
        
        print(f"   - 响应矩阵形状: {response_matrix.shape}")
        
        # 构建图数据
        print("\n📈 构建模型图...")
        train_response_matrix = response_matrix[:90]
        train_model_embeddings = model_embeddings[shuffled_model_ids[:90]]
        
        graph_data = build_model_graph(
            train_response_matrix, 
            train_model_embeddings, 
            k_neighbors=10, 
            device=device
        )
        print(f"   - 图节点数: {graph_data.x.shape[0]}")
        print(f"   - 图边数: {graph_data.edge_index.shape[1]}")
        
        # 准备未见模型响应
        unseen_responses = response_matrix[90:112]
        
        # 创建模型
        print("\n🧠 初始化GraphSAGE模型...")
        reordered_embeddings = model_embeddings[shuffled_model_ids]
        
        model = GraphSAGE_TextMF_dyn(
            question_embeddings=question_embeddings,
            model_embedding_dim=1024,
            alpha=0.1,
            num_models=num_models,
            num_prompts=num_prompts,
            model_embeddings=reordered_embeddings,
            is_dyn=True,
            frozen=False,
            hidden_dim=256,
            num_layers=2,
            num_train_models=90
        )
        model.to(device)
        model.set_train_responses(train_response_matrix)
        
        print(f"   - 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 简单训练测试（只训练5个epoch）
        print("\n🏋️ 开始演示训练（5个epoch）...")
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(5):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for models, prompts, labels in train_loader:
                models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
                
                train_mask = models < 90
                if train_mask.sum() == 0:
                    continue
                
                optimizer.zero_grad()
                logits = model(graph_data, models[train_mask], prompts[train_mask], test_mode=False)
                loss = loss_fn(logits, labels[train_mask])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            train_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # 测试
            model.eval()
            test_loss, test_accuracy = evaluate_graphsage(model, graph_data, test_loader, device, unseen_responses.to(device))
            
            print(f"Epoch {epoch+1}/5 | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Test Acc: {test_accuracy:.6f}")
        
        print("\n✅ 演示测试完成！")
        print("🎯 脚本功能正常，可以运行完整测试")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_test()
    if success:
        print("\n🎉 演示测试成功！可以运行完整的测试脚本")
        print("运行完整测试: python test_graphsage_random.py")
    else:
        print("\n💥 演示测试失败！请检查错误信息")
