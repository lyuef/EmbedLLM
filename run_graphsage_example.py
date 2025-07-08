"""
GraphSAGE训练示例脚本
展示如何使用GraphSAGE模型进行训练
"""
import argparse
import torch
import pandas as pd
import numpy as np
from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph
from utils.load_and_process_data import build_full_response_matrix, load_and_process_data
from utils.train import train_graphsage
from datetime import datetime
import os
import sys
from contextlib import redirect_stdout

class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()  # 确保实时写入
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def create_sample_args():
    """创建示例参数"""
    args = argparse.Namespace()
    
    # 数据路径 (需要根据实际情况修改)
    args.train_data_path = "data/train.csv"  
    args.test_data_path = "data/test.csv"    
    args.question_embedding_path = "data/question_embeddings.pth"  
    
    # 模型参数
    args.embedding_dim = 768
    args.alpha = 0.1
    args.batch_size = 64
    args.num_epochs = 50
    args.learning_rate = 0.001
    
    # 数据分割参数
    args.model_use_train_l = 0
    args.model_use_train_r = 90   # 训练前90个模型
    args.model_use_test_l = 90
    args.model_use_test_r = 112   # 测试后22个模型
    
    # 动态嵌入参数
    args.is_dyn = True
    args.frozen = False
    
    # 生成带日期的输出路径
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"output/{current_date}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成包含关键参数的文件名
    model_filename = f"GraphSAGE_train_{args.model_use_train_l}_{args.model_use_train_r}_test_{args.model_use_test_l}_{args.model_use_test_r}_dyn_{args.is_dyn}_frozen_{args.frozen}.pth"
    # args.model_save_path = os.path.join(output_dir, model_filename)
    args.model_save_path = None 

    # 生成训练输出日志文件路径
    log_filename = f"GraphSAGE_train_{args.model_use_train_l}_{args.model_use_train_r}_test_{args.model_use_test_l}_{args.model_use_test_r}_dyn_{args.is_dyn}_frozen_{args.frozen}.txt"
    args.log_save_path = os.path.join(output_dir, log_filename)
    
    return args

def run_graphsage_training():
    """运行GraphSAGE训练"""
    print("🚀 Starting GraphSAGE Training Example")
    print("=" * 50)
    
    # 创建示例参数
    args = create_sample_args()
    
    # 检查文件是否存在
    required_files = [
        args.train_data_path,
        args.test_data_path,
        args.question_embedding_path,
        "data/model_embeddings_static.pth"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Please ensure all data files are available before running.")
        print("   You can modify the paths in create_sample_args() function.")
        return False
    
    try:
        print("📊 Loading dataset...")
        train_data = pd.read_csv(args.train_data_path)
        test_data = pd.read_csv(args.test_data_path)
        question_embeddings = torch.load(args.question_embedding_path)
        
        num_prompts = question_embeddings.shape[0]
        num_models = len(test_data["model_id"].unique())
        
        print(f"   - Number of models: {num_models}")
        print(f"   - Number of prompts: {num_prompts}")
        print(f"   - Training data size: {len(train_data)}")
        print(f"   - Test data size: {len(test_data)}")
        
        # 构建响应矩阵
        print("\n🔗 Building response matrix for graph construction...")
        response_matrix, all_model_ids, all_prompt_ids = build_full_response_matrix(train_data, test_data)

        # 随机打乱
        response_matrix = response_matrix[torch.randperm(response_matrix.size(0))]
        print(f"   - Response matrix shape: {response_matrix.shape}")
        
        # 加载模型嵌入
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - Using device: {device}")
        
        model_embeddings = torch.load("data/model_embeddings_static.pth", map_location=device)
        print(f"   - Model embeddings shape: {model_embeddings.shape}")
        
        # 构建图数据
        print("\n📈 Building model graph...")
        graph_data = build_model_graph(response_matrix, model_embeddings, k_neighbors=10, device=device)
        print(f"   - Graph nodes: {graph_data.x.shape[0]}")
        print(f"   - Graph edges: {graph_data.edge_index.shape[1]}")
        
        # 加载和处理数据
        print("\n🔄 Processing data loaders...")
        train_loader, test_loader = load_and_process_data(
            train_data=train_data, test_data=test_data, batch_size=args.batch_size,
            model_use_train_l=args.model_use_train_l, model_use_train_r=args.model_use_train_r,
            model_use_test_l=args.model_use_test_l, model_use_test_r=args.model_use_test_r
        )
        
        # 创建模型
        print("\n🧠 Initializing GraphSAGE model...")
        model = GraphSAGE_TextMF_dyn(
            question_embeddings=question_embeddings,
            model_embedding_dim=args.embedding_dim,
            alpha=args.alpha,
            num_models=num_models,
            num_prompts=num_prompts,
            model_embeddings=model_embeddings,
            is_dyn=args.is_dyn,
            frozen=args.frozen,
            hidden_dim=256,
            num_layers=2
        )
        model.to(device)
        
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 训练配置
        print(f"\n⚙️ Training configuration:")
        print(f"   - Training models: {args.model_use_train_l} to {args.model_use_train_r-1}")
        print(f"   - Test models: {args.model_use_test_l} to {args.model_use_test_r-1}")
        print(f"   - Dynamic embedding: {args.is_dyn}")
        print(f"   - Frozen: {args.frozen}")
        print(f"   - Epochs: {args.num_epochs}")
        print(f"   - Learning rate: {args.learning_rate}")
        print(f"   - Batch size: {args.batch_size}")
        
        # 开始训练
        print(f"\n🏋️ Starting training...")
        print(f"📝 Training output will be saved to: {args.log_save_path}")
        print("=" * 50)
        
        # 设置输出重定向到文件和控制台
        tee_output = TeeOutput(args.log_save_path)
        original_stdout = sys.stdout
        
        try:
            # 写入训练配置到日志文件
            sys.stdout = tee_output
            print("=" * 50)
            print("GraphSAGE Training Log")
            print("=" * 50)
            print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Training models: {args.model_use_train_l} to {args.model_use_train_r-1}")
            print(f"Test models: {args.model_use_test_l} to {args.model_use_test_r-1}")
            print(f"Dynamic embedding: {args.is_dyn}")
            print(f"Frozen: {args.frozen}")
            print(f"Epochs: {args.num_epochs}")
            print(f"Learning rate: {args.learning_rate}")
            print(f"Batch size: {args.batch_size}")
            print(f"Device: {device}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print("=" * 50)
            
            # 开始训练
            max_acc = train_graphsage(
                model, graph_data, train_loader, test_loader,
                num_epochs=args.num_epochs, lr=args.learning_rate,
                device=device, save_path=args.model_save_path
            )
            
            print("=" * 50)
            print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Maximum accuracy achieved: {max_acc:.4f}")
            print(f"Model saved to: {args.model_save_path}")
            print("=" * 50)
            
        finally:
            # 恢复原始输出
            sys.stdout = original_stdout
            tee_output.close()
        
        print("=" * 50)
        print(f"✅ Training completed!")
        print(f"🎯 Maximum accuracy achieved: {max_acc:.4f}")
        print(f"💾 Model saved to: {args.model_save_path}")
        print(f"📝 Training log saved to: {args.log_save_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_graphsage_training()
    if success:
        print("\n🎉 GraphSAGE training example completed successfully!")
    else:
        print("\n💥 GraphSAGE training example failed!")
