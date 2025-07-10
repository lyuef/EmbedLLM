"""
GraphSAGE + Matrix Factorization 随机测试脚本
每次训练随机打乱112个模型，前90个作为训练，后22个作为测试
每个epoch后进行测试并保存结果
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

class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def build_train_response_matrix_with_progress(train_data, shuffled_model_ids, save_path=None):
    """
    只使用训练数据构建响应矩阵，避免数据泄露，并显示进度条
    """
    print("🔗 构建训练响应矩阵（仅使用训练数据）...")
    
    # 检查是否已存在保存的响应矩阵
    if save_path and os.path.exists(save_path):
        print(f"   - 发现已保存的响应矩阵: {save_path}")
        try:
            saved_data = torch.load(save_path, weights_only=True)
            response_matrix = saved_data['response_matrix']
            saved_model_ids = saved_data['model_ids']
            all_prompt_ids = saved_data['prompt_ids']
            
            # 检查模型ID是否匹配
            if np.array_equal(saved_model_ids, shuffled_model_ids):
                print(f"   - 模型ID匹配，直接加载响应矩阵: {response_matrix.shape}")
                return response_matrix, all_prompt_ids
            else:
                print(f"   - 模型ID不匹配，重新构建响应矩阵")
        except Exception as e:
            print(f"   - 加载失败: {e}，重新构建响应矩阵")
    
    # 获取所有prompt ID
    all_prompt_ids = sorted(train_data["prompt_id"].unique())
    num_models = len(shuffled_model_ids)
    num_prompts = len(all_prompt_ids)
    
    # 创建ID映射
    model_id_to_idx = {model_id: idx for idx, model_id in enumerate(shuffled_model_ids)}
    prompt_id_to_idx = {prompt_id: idx for idx, prompt_id in enumerate(all_prompt_ids)}
    
    # 初始化响应矩阵
    response_matrix = torch.zeros(num_models, num_prompts, dtype=torch.float32)
    
    print(f"   - 响应矩阵形状: {response_matrix.shape}")
    print(f"   - 处理训练数据...")
    
    # 使用进度条处理数据
    for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="构建响应矩阵"):
        model_id = row["model_id"]
        prompt_id = row["prompt_id"]
        label = row["label"]
        
        if model_id in model_id_to_idx and prompt_id in prompt_id_to_idx:
            model_idx = model_id_to_idx[model_id]
            prompt_idx = prompt_id_to_idx[prompt_id]
            response_matrix[model_idx, prompt_idx] = label
    
    # 保存响应矩阵
    if save_path:
        print(f"   - 保存响应矩阵到: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'response_matrix': response_matrix,
            'model_ids': shuffled_model_ids,
            'prompt_ids': all_prompt_ids
        }, save_path)
    
    return response_matrix, all_prompt_ids

def create_random_data_loaders(train_data, test_data, batch_size=64, random_seed=None):
    """
    创建随机打乱的数据加载器
    每次调用都会重新随机打乱模型顺序
    """
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
    
    print(f"训练模型ID (前90个): {train_model_ids[:10]}... (显示前10个)")
    print(f"测试模型ID (后22个): {test_model_ids}")
    
    # 过滤数据
    train_data_filtered = train_data[train_data["model_id"].isin(train_model_ids)]
    test_data_filtered = test_data[test_data["model_id"].isin(test_model_ids)]
    
    # 创建ID映射 (将原始ID映射到0-111的连续ID)
    id_mapping = {original_id: new_id for new_id, original_id in enumerate(shuffled_model_ids)}
    
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    
    class CustomDataset(Dataset):
        def __init__(self, data, id_mapping):
            # 将原始模型ID映射到新的连续ID
            mapped_model_ids = [id_mapping[mid] for mid in data["model_id"].values]
            self.models = torch.tensor(mapped_model_ids, dtype=torch.int64)
            self.prompts = torch.tensor(data["prompt_id"].to_numpy(), dtype=torch.int64)
            self.labels = torch.tensor(data["label"].to_numpy(), dtype=torch.int64)
            
            self.num_models = len(data["model_id"].unique())
            self.num_prompts = num_prompts
            self.num_classes = len(data["label"].unique())

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return self.models[index], self.prompts[index], self.labels[index]

    train_dataset = CustomDataset(train_data_filtered, id_mapping)
    test_dataset = CustomDataset(test_data_filtered, id_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, shuffled_model_ids, id_mapping

def build_unseen_response_matrix_from_train(train_data, shuffled_model_ids, all_prompt_ids):
    """
    从训练数据中为未见模型（后22个）构建响应矩阵，避免数据泄露
    """
    print("🔗 从训练数据构建未见模型响应矩阵（避免数据泄露）...")
    
    # 获取未见模型ID（打乱后的后22个）
    unseen_model_ids = shuffled_model_ids[90:112]
    
    # 创建prompt ID映射
    prompt_id_to_idx = {prompt_id: idx for idx, prompt_id in enumerate(all_prompt_ids)}
    
    # 初始化未见模型响应矩阵（只包含后22个模型）
    unseen_response_matrix = torch.zeros(22, len(all_prompt_ids), dtype=torch.float32)
    
    # 从训练数据中过滤未见模型的数据
    train_data_filtered = train_data[train_data["model_id"].isin(unseen_model_ids)]
    
    print(f"   - 未见模型响应矩阵形状: {unseen_response_matrix.shape}")
    print(f"   - 未见模型ID: {unseen_model_ids}")
    print(f"   - 从训练数据中提取未见模型响应...")
    
    # 使用进度条处理数据
    for _, row in tqdm(train_data_filtered.iterrows(), total=len(train_data_filtered), desc="构建未见模型响应矩阵"):
        model_id = row["model_id"]
        prompt_id = row["prompt_id"]
        label = row["label"]
        
        if model_id in unseen_model_ids and prompt_id in prompt_id_to_idx:
            # 将模型ID映射到未见模型矩阵的索引（0-21）
            unseen_model_idx = list(unseen_model_ids).index(model_id)
            prompt_idx = prompt_id_to_idx[prompt_id]
            unseen_response_matrix[unseen_model_idx, prompt_idx] = label
    
    print(f"   - 成功构建未见模型响应矩阵，数据来源：训练集")
    return unseen_response_matrix

def train_graphsage_with_epoch_testing(model, graph_data, train_loader, test_loader, 
                                     num_epochs, lr, device, unseen_responses=None, 
                                     weight_decay=1e-5, log_file=None, contrastive_weight=0.1):
    """
    训练GraphSAGE模型，每个epoch后进行测试，支持图对比学习
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # 记录训练历史
    train_history = []
    test_history = []
    contrastive_loss_history = []
    max_accuracy = 0
    best_epoch = 0
    
    print("开始训练...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_contrastive_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for models, prompts, labels in progress_bar:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            
            # 只训练前90个模型
            train_mask = models < 90
            if train_mask.sum() == 0:
                continue
            
            optimizer.zero_grad()
            
            # 前向传播，获取logits和图对比学习损失
            output = model(graph_data, models[train_mask], prompts[train_mask], test_mode=False)
            
            if isinstance(output, tuple):
                # 训练模式：返回(logits, contrastive_loss)
                logits, contrastive_loss = output
            else:
                # 兼容性：如果只返回logits
                logits = output
                contrastive_loss = torch.tensor(0.0, device=device)
            
            # 主任务损失
            main_loss = loss_fn(logits, labels[train_mask])
            
            # 总损失 = 主任务损失 + 图对比学习损失
            total_batch_loss = main_loss + contrastive_weight * contrastive_loss
            
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_main_loss += main_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'total': f'{total_batch_loss.item():.4f}',
                'main': f'{main_loss.item():.4f}',
                'contrast': f'{contrastive_loss.item():.4f}'
            })
        
        train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_main_loss = total_main_loss / num_batches if num_batches > 0 else 0
        avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0
        
        # 测试阶段
        model.eval()
        test_loss, test_accuracy = evaluate_graphsage(model, graph_data, test_loader, device, unseen_responses)
        
        # 更新最佳结果
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            best_epoch = epoch + 1
        
        # 记录历史
        train_history.append(train_loss)
        test_history.append(test_accuracy)
        contrastive_loss_history.append(avg_contrastive_loss)
        
        # 输出结果
        result_line = f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.6f} (Main: {avg_main_loss:.6f}, Contrast: {avg_contrastive_loss:.6f}) | Test Loss: {test_loss:.6f} | Test Acc: {test_accuracy:.6f} | Best: {max_accuracy:.6f} (Epoch {best_epoch})"
        print(result_line)
        
        # 如果有日志文件，也写入文件
        if log_file:
            log_file.write(result_line + "\n")
            log_file.flush()
    
    print("=" * 80)
    print(f"训练完成！最佳准确率: {max_accuracy:.6f} (第 {best_epoch} 轮)")
    
    return max_accuracy, best_epoch, train_history, test_history, contrastive_loss_history

def create_test_args():
    """创建测试参数"""
    args = argparse.Namespace()
    
    # 数据路径
    args.train_data_path = "data/train.csv"
    args.test_data_path = "data/test.csv"
    args.question_embedding_path = "data/question_embeddings.pth"
    args.model_embeddings_path = "data/model_embeddings_static.pth"
    
    # 模型参数
    args.embedding_dim = 1024  # 匹配model_embeddings_static.pth的维度
    args.alpha = 0.05
    args.batch_size = 64
    args.num_epochs = 50
    args.learning_rate = 0.0005
    args.weight_decay = 1e-5
    
    # GNN参数
    args.hidden_dim = 256
    args.num_layers = 2
    args.k_neighbors = 5
    args.gnn_type = "GAT"  # 或 "GAT"
    args.num_heads = 8
    
    # 动态嵌入参数
    args.is_dyn = True
    args.frozen = False
    
    # 随机种子
    args.random_seed = 42
    
    # 输出设置
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H-%M-%S")
    output_dir = f"output/{current_date}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    log_filename = f"GraphSAGE_Random_Test_train90_test22_epochs{args.num_epochs}_lr{args.learning_rate}_seed{args.random_seed}_{current_time}.txt"
    args.log_save_path = os.path.join(output_dir, log_filename)
    
    return args

def run_graphsage_random_test():
    """运行GraphSAGE随机测试"""
    print("🚀 GraphSAGE + Matrix Factorization 随机测试")
    print("=" * 80)
    
    # 创建参数
    args = create_test_args()
    
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # 检查文件是否存在
    required_files = [
        args.train_data_path,
        args.test_data_path,
        args.question_embedding_path,
        args.model_embeddings_path
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    try:
        print("📊 加载数据集...")
        train_data = pd.read_csv(args.train_data_path)
        test_data = pd.read_csv(args.test_data_path)
        question_embeddings = torch.load(args.question_embedding_path)
        model_embeddings = torch.load(args.model_embeddings_path)
        
        num_prompts = question_embeddings.shape[0]
        num_models = 112  # 固定为112个模型
        
        print(f"   - 模型数量: {num_models}")
        print(f"   - 问题数量: {num_prompts}")
        print(f"   - 训练数据大小: {len(train_data)}")
        print(f"   - 测试数据大小: {len(test_data)}")
        
        # 设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - 使用设备: {device}")
        
        # 创建随机数据加载器
        print("\n🔀 创建随机数据加载器...")
        train_loader, test_loader, shuffled_model_ids, id_mapping = create_random_data_loaders(
            train_data, test_data, batch_size=args.batch_size, random_seed=args.random_seed
        )
        
        # 构建训练响应矩阵（仅使用训练数据，避免数据泄露）
        current_date = datetime.now().strftime("%Y-%m-%d")
        response_matrix_save_path = f"data/response_matrix_{current_date}_seed{args.random_seed}.pth"
        
        train_response_matrix, all_prompt_ids = build_train_response_matrix_with_progress(
            train_data, shuffled_model_ids, save_path=response_matrix_save_path
        )
        
        # 构建图数据（只使用前90个训练模型）
        print("\n📈 构建模型图...")
        train_response_matrix_90 = train_response_matrix[:90]  # 只用前90个模型
        train_model_embeddings = model_embeddings[shuffled_model_ids[:90]]  # 对应的模型嵌入
        
        graph_data = build_model_graph(
            train_response_matrix_90, 
            train_model_embeddings, 
            k_neighbors=args.k_neighbors, 
            device=device
        )
        print(f"   - 图节点数: {graph_data.x.shape[0]}")
        print(f"   - 图边数: {graph_data.edge_index.shape[1]}")
        
        # 从训练数据构建未见模型响应矩阵（避免数据泄露）
        unseen_responses = build_unseen_response_matrix_from_train(
            train_data, shuffled_model_ids, all_prompt_ids
        )
        
        # 创建模型
        print("\n🧠 初始化GraphSAGE模型...")
        # 重新排列模型嵌入以匹配新的顺序
        reordered_embeddings = model_embeddings[shuffled_model_ids]
        
        model = GraphSAGE_TextMF_dyn(
            question_embeddings=question_embeddings,
            model_embedding_dim=args.embedding_dim,
            alpha=args.alpha,
            num_models=num_models,
            num_prompts=num_prompts,
            model_embeddings=reordered_embeddings,
            is_dyn=args.is_dyn,
            frozen=args.frozen,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_train_models=90,
            gnn_type=args.gnn_type,
            num_heads=args.num_heads
        )
        model.to(device)
        
        # 设置训练响应矩阵（用于未见模型的相似度计算）
        model.set_train_responses(train_response_matrix_90)
        
        print(f"   - 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 训练配置
        print(f"\n⚙️ 训练配置:")
        print(f"   - 训练模型: 前90个 (ID: {shuffled_model_ids[:5]}...)")
        print(f"   - 测试模型: 后22个 (ID: {shuffled_model_ids[90:95]}...)")
        print(f"   - 动态嵌入: {args.is_dyn}")
        print(f"   - 冻结参数: {args.frozen}")
        print(f"   - 训练轮数: {args.num_epochs}")
        print(f"   - 学习率: {args.learning_rate}")
        print(f"   - 批次大小: {args.batch_size}")
        print(f"   - 随机种子: {args.random_seed}")
        
        # 开始训练
        print(f"\n🏋️ 开始训练...")
        print(f"📝 训练日志将保存到: {args.log_save_path}")
        print("=" * 80)
        
        # 设置输出重定向
        tee_output = TeeOutput(args.log_save_path)
        original_stdout = sys.stdout
        
        try:
            # 写入配置信息到日志
            sys.stdout = tee_output
            print("=" * 80)
            print("GraphSAGE + Matrix Factorization 随机测试日志")
            print("=" * 80)
            print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"项目: GraphSAGE + Matrix Factorization")
            print(f"测试类型: 随机打乱模型顺序测试")
            print(f"训练模型: 前90个")
            print(f"测试模型: 后22个")
            print(f"随机种子: {args.random_seed}")
            print(f"训练轮数: {args.num_epochs}")
            print(f"学习率: {args.learning_rate}")
            print(f"批次大小: {args.batch_size}")
            print(f"动态嵌入: {args.is_dyn}")
            print(f"冻结参数: {args.frozen}")
            print(f"GNN类型: {args.gnn_type}")
            print(f"注意力头数: {args.num_heads}")
            print(f"设备: {device}")
            print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
            print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print(f"打乱后的模型顺序 (前10个): {shuffled_model_ids[:10]}")
            print(f"训练模型ID: {shuffled_model_ids[:90]}")
            print(f"测试模型ID: {shuffled_model_ids[90:112]}")
            print("=" * 80)
            
            # 开始训练
            result = train_graphsage_with_epoch_testing(
                model, graph_data, train_loader, test_loader,
                num_epochs=args.num_epochs, lr=args.learning_rate,
                device=device, unseen_responses=unseen_responses.to(device),
                weight_decay=args.weight_decay, log_file=tee_output.file,
                contrastive_weight=0.1  # 图对比学习权重
            )
            
            # 处理返回值（兼容新旧版本）
            if len(result) == 5:
                max_acc, best_epoch, train_history, test_history, contrastive_loss_history = result
            else:
                max_acc, best_epoch, train_history, test_history = result
                contrastive_loss_history = []
            
            print("=" * 80)
            print("训练总结:")
            print(f"最佳测试准确率: {max_acc:.6f}")
            print(f"最佳轮次: {best_epoch}")
            print(f"最终训练损失: {train_history[-1]:.6f}")
            print(f"最终测试准确率: {test_history[-1]:.6f}")
            print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
        finally:
            # 恢复原始输出
            sys.stdout = original_stdout
            tee_output.close()
        
        print("=" * 80)
        print(f"✅ 测试完成!")
        print(f"🎯 最佳准确率: {max_acc:.6f} (第 {best_epoch} 轮)")
        print(f"📝 详细日志已保存到: {args.log_save_path}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_graphsage_random_test()
    if success:
        print("\n🎉 GraphSAGE随机测试完成!")
    else:
        print("\n💥 GraphSAGE随机测试失败!")
