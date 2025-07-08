"""
使用模拟数据测试GraphSAGE测试脚本的所有函数
"""
import os
import torch
import pandas as pd
import numpy as np
import random
from datetime import datetime
import tempfile
import shutil

def create_mock_data():
    """创建模拟数据用于测试"""
    print("🔧 创建模拟数据...")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 模拟参数
    num_models = 10  # 简化为10个模型
    num_prompts = 50  # 50个问题
    num_train_samples = 300  # 训练样本数
    num_test_samples = 100   # 测试样本数
    
    # 创建临时数据目录
    temp_data_dir = "temp_test_data"
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 1. 创建模拟训练数据
    print("   - 创建模拟训练数据...")
    train_data = []
    for i in range(num_train_samples):
        model_id = np.random.randint(0, num_models)
        prompt_id = np.random.randint(0, num_prompts)
        label = np.random.randint(0, 2)  # 0或1
        category_id = np.random.randint(0, 5)
        
        train_data.append({
            'prompt_id': prompt_id,
            'model_id': model_id,
            'category_id': category_id,
            'label': label,
            'prompt': f'This is test prompt {prompt_id}',
            'model_name': f'model_{model_id}',
            'category': f'category_{category_id}'
        })
    
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(f"{temp_data_dir}/train.csv", index=False)
    print(f"      训练数据形状: {train_df.shape}")
    
    # 2. 创建模拟测试数据
    print("   - 创建模拟测试数据...")
    test_data = []
    for i in range(num_test_samples):
        model_id = np.random.randint(0, num_models)
        prompt_id = np.random.randint(0, num_prompts)
        label = np.random.randint(0, 2)
        category_id = np.random.randint(0, 5)
        
        test_data.append({
            'prompt_id': prompt_id,
            'model_id': model_id,
            'category_id': category_id,
            'label': label,
            'prompt': f'This is test prompt {prompt_id}',
            'model_name': f'model_{model_id}',
            'category': f'category_{category_id}'
        })
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(f"{temp_data_dir}/test.csv", index=False)
    print(f"      测试数据形状: {test_df.shape}")
    
    # 3. 创建模拟问题嵌入
    print("   - 创建模拟问题嵌入...")
    question_embeddings = torch.randn(num_prompts, 768)  # 768维嵌入
    torch.save(question_embeddings, f"{temp_data_dir}/question_embeddings.pth")
    print(f"      问题嵌入形状: {question_embeddings.shape}")
    
    # 4. 创建模拟模型嵌入
    print("   - 创建模拟模型嵌入...")
    model_embeddings = torch.randn(num_models, 1024)  # 1024维嵌入
    torch.save(model_embeddings, f"{temp_data_dir}/model_embeddings_static.pth")
    print(f"      模型嵌入形状: {model_embeddings.shape}")
    
    print(f"✅ 模拟数据创建完成，保存在: {temp_data_dir}/")
    return temp_data_dir

def test_functions_with_mock_data(data_dir):
    """使用模拟数据测试所有函数"""
    print("\n🧪 测试GraphSAGE函数...")
    
    try:
        # 导入测试脚本中的函数
        import sys
        sys.path.append('.')
        
        from test_graphsage_random import (
            build_train_response_matrix_with_progress,
            build_test_response_matrix_with_progress,
            create_random_data_loaders,
            set_random_seed
        )
        from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph
        
        # 设置随机种子
        set_random_seed(42)
        
        # 1. 测试数据加载
        print("\n1️⃣ 测试数据加载...")
        train_data = pd.read_csv(f"{data_dir}/train.csv")
        test_data = pd.read_csv(f"{data_dir}/test.csv")
        question_embeddings = torch.load(f"{data_dir}/question_embeddings.pth", weights_only=True)
        model_embeddings = torch.load(f"{data_dir}/model_embeddings_static.pth", weights_only=True)
        
        print(f"   ✅ 训练数据: {train_data.shape}")
        print(f"   ✅ 测试数据: {test_data.shape}")
        print(f"   ✅ 问题嵌入: {question_embeddings.shape}")
        print(f"   ✅ 模型嵌入: {model_embeddings.shape}")
        
        # 2. 测试随机数据加载器
        print("\n2️⃣ 测试随机数据加载器...")
        train_loader, test_loader, shuffled_model_ids, id_mapping = create_random_data_loaders(
            train_data, test_data, batch_size=16, random_seed=42
        )
        
        print(f"   ✅ 训练加载器批次数: {len(train_loader)}")
        print(f"   ✅ 测试加载器批次数: {len(test_loader)}")
        print(f"   ✅ 打乱后的模型ID: {shuffled_model_ids}")
        
        # 3. 测试响应矩阵构建
        print("\n3️⃣ 测试响应矩阵构建...")
        response_matrix_path = f"{data_dir}/test_response_matrix.pth"
        
        train_response_matrix, all_prompt_ids = build_train_response_matrix_with_progress(
            train_data, shuffled_model_ids, save_path=response_matrix_path
        )
        
        print(f"   ✅ 训练响应矩阵: {train_response_matrix.shape}")
        print(f"   ✅ 问题ID数量: {len(all_prompt_ids)}")
        
        # 测试保存和加载
        print("   - 测试响应矩阵保存和加载...")
        train_response_matrix_2, all_prompt_ids_2 = build_train_response_matrix_with_progress(
            train_data, shuffled_model_ids, save_path=response_matrix_path
        )
        
        if torch.equal(train_response_matrix, train_response_matrix_2):
            print("   ✅ 响应矩阵保存/加载功能正常")
        else:
            print("   ❌ 响应矩阵保存/加载功能异常")
        
        # 4. 测试测试响应矩阵构建
        print("\n4️⃣ 测试测试响应矩阵构建...")
        test_response_matrix = build_test_response_matrix_with_progress(
            test_data, shuffled_model_ids, all_prompt_ids
        )
        
        print(f"   ✅ 测试响应矩阵: {test_response_matrix.shape}")
        
        # 5. 测试图构建
        print("\n5️⃣ 测试模型图构建...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用前7个模型作为训练模型（模拟90个中的前7个）
        train_response_matrix_7 = train_response_matrix[:7]
        train_model_embeddings = model_embeddings[shuffled_model_ids[:7]]
        
        graph_data = build_model_graph(
            train_response_matrix_7,
            train_model_embeddings,
            k_neighbors=3,  # 减少邻居数
            device=device
        )
        
        print(f"   ✅ 图节点数: {graph_data.x.shape[0]}")
        print(f"   ✅ 图边数: {graph_data.edge_index.shape[1]}")
        
        # 6. 测试模型初始化
        print("\n6️⃣ 测试GraphSAGE模型初始化...")
        num_models = len(shuffled_model_ids)
        num_prompts = question_embeddings.shape[0]
        
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
            hidden_dim=128,  # 减少隐藏层维度
            num_layers=2,
            num_train_models=7  # 前7个作为训练模型
        )
        model.to(device)
        model.set_train_responses(train_response_matrix_7)
        
        print(f"   ✅ 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✅ 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 7. 测试模型前向传播
        print("\n7️⃣ 测试模型前向传播...")
        
        # 准备未见模型响应（后3个模型，即模型7,8,9）
        unseen_model_ids = shuffled_model_ids[7:10]  # 后3个模型
        
        # 重新构建正确的测试响应矩阵
        test_data_for_unseen = test_data[test_data["model_id"].isin(unseen_model_ids)]
        unseen_responses = torch.zeros(3, len(all_prompt_ids), dtype=torch.float32)
        
        prompt_id_to_idx = {prompt_id: idx for idx, prompt_id in enumerate(all_prompt_ids)}
        
        for _, row in test_data_for_unseen.iterrows():
            model_id = row["model_id"]
            prompt_id = row["prompt_id"]
            label = row["label"]
            
            if model_id in unseen_model_ids and prompt_id in prompt_id_to_idx:
                unseen_model_idx = list(unseen_model_ids).index(model_id)
                prompt_idx = prompt_id_to_idx[prompt_id]
                unseen_responses[unseen_model_idx, prompt_idx] = label
        
        print(f"   - 未见模型响应矩阵形状: {unseen_responses.shape}")
        
        # 测试1: 只有训练模型的批次
        print("   - 测试训练模型前向传播...")
        batch_size = 4
        train_model_ids = torch.randint(0, 7, (batch_size,)).to(device)  # 只选择前7个训练模型
        test_prompt_ids = torch.randint(0, num_prompts, (batch_size,)).to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(graph_data, train_model_ids, test_prompt_ids, 
                          unseen_responses.to(device), test_mode=True)
            predictions = model.predict(graph_data, train_model_ids, test_prompt_ids, 
                                      unseen_responses.to(device))
        
        print(f"      ✅ 训练模型logits形状: {logits.shape}")
        print(f"      ✅ 训练模型预测形状: {predictions.shape}")
        
        # 测试2: 只有未见模型的批次
        print("   - 测试未见模型前向传播...")
        unseen_model_ids_batch = torch.tensor([7, 8, 9], dtype=torch.long).to(device)  # 后3个模型
        unseen_prompt_ids = torch.randint(0, num_prompts, (3,)).to(device)
        
        with torch.no_grad():
            logits = model(graph_data, unseen_model_ids_batch, unseen_prompt_ids, 
                          unseen_responses.to(device), test_mode=True)
            predictions = model.predict(graph_data, unseen_model_ids_batch, unseen_prompt_ids, 
                                      unseen_responses.to(device))
        
        print(f"      ✅ 未见模型logits形状: {logits.shape}")
        print(f"      ✅ 未见模型预测形状: {predictions.shape}")
        print(f"      ✅ 未见模型预测值: {predictions.cpu().numpy()}")
        
        # 测试3: 混合批次（训练模型+未见模型）
        print("   - 测试混合模型前向传播...")
        mixed_model_ids = torch.tensor([0, 1, 7, 8], dtype=torch.long).to(device)  # 2个训练+2个未见
        mixed_prompt_ids = torch.randint(0, num_prompts, (4,)).to(device)
        
        with torch.no_grad():
            logits = model(graph_data, mixed_model_ids, mixed_prompt_ids, 
                          unseen_responses.to(device), test_mode=True)
            predictions = model.predict(graph_data, mixed_model_ids, mixed_prompt_ids, 
                                      unseen_responses.to(device))
        
        print(f"      ✅ 混合模型logits形状: {logits.shape}")
        print(f"      ✅ 混合模型预测形状: {predictions.shape}")
        print(f"      ✅ 混合模型预测值: {predictions.cpu().numpy()}")
        
        # 8. 测试数据加载器迭代
        print("\n8️⃣ 测试数据加载器迭代...")
        train_batch_count = 0
        test_batch_count = 0
        
        for models, prompts, labels in train_loader:
            train_batch_count += 1
            if train_batch_count == 1:  # 只检查第一个批次
                print(f"   ✅ 训练批次形状 - 模型: {models.shape}, 问题: {prompts.shape}, 标签: {labels.shape}")
                break
        
        for models, prompts, labels in test_loader:
            test_batch_count += 1
            if test_batch_count == 1:  # 只检查第一个批次
                print(f"   ✅ 测试批次形状 - 模型: {models.shape}, 问题: {prompts.shape}, 标签: {labels.shape}")
                break
        
        print(f"   ✅ 训练加载器总批次数: {len(train_loader)}")
        print(f"   ✅ 测试加载器总批次数: {len(test_loader)}")
        
        print("\n🎉 所有函数测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_temp_data(data_dir):
    """清理临时数据"""
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"🧹 清理临时数据: {data_dir}")

def check_real_data_files():
    """检查真实数据文件是否存在"""
    print("\n🔍 检查真实数据文件...")
    print("=" * 60)
    
    required_files = [
        "data/train.csv",
        "data/test.csv", 
        "data/question_embeddings.pth",
        "data/model_embeddings_static.pth"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - 存在")
            
            # 尝试加载文件以检查格式
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                    print(f"   📊 数据形状: {data.shape}")
                    print(f"   📋 列名: {list(data.columns)}")
                    if 'model_id' in data.columns:
                        unique_models = data['model_id'].nunique()
                        print(f"   🤖 唯一模型数: {unique_models}")
                    if 'prompt_id' in data.columns:
                        unique_prompts = data['prompt_id'].nunique()
                        print(f"   ❓ 唯一问题数: {unique_prompts}")
                        
                elif file_path.endswith('.pth'):
                    tensor = torch.load(file_path, map_location='cpu', weights_only=True)
                    print(f"   📐 张量形状: {tensor.shape}")
                    print(f"   🔢 数据类型: {tensor.dtype}")
                    
            except Exception as e:
                print(f"   ⚠️ 加载文件时出错: {e}")
                
        else:
            print(f"❌ {file_path} - 不存在")
            all_exist = False
        
        print()
    
    return all_exist

def main():
    """主函数"""
    print("🚀 GraphSAGE测试脚本功能验证")
    print("=" * 80)
    
    # 1. 使用模拟数据测试所有函数
    temp_data_dir = create_mock_data()
    
    try:
        success = test_functions_with_mock_data(temp_data_dir)
        
        if success:
            print("\n✅ 模拟数据测试通过！所有函数工作正常。")
        else:
            print("\n❌ 模拟数据测试失败！请检查代码。")
            return False
            
    finally:
        cleanup_temp_data(temp_data_dir)
    
    # 2. 检查真实数据文件
    real_data_exists = check_real_data_files()
    
    print("=" * 80)
    if success and real_data_exists:
        print("🎉 所有测试通过！可以运行完整的测试脚本：")
        print("   python test_graphsage_random.py")
    elif success:
        print("✅ 函数测试通过，但缺少真实数据文件")
        print("   请确保数据文件存在后运行: python test_graphsage_random.py")
    else:
        print("❌ 函数测试失败，请检查代码后重试")
    
    return success and real_data_exists

if __name__ == "__main__":
    main()
