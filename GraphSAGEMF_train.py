from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph
from models.Embedllm_dynamic import TextMF_dyn, TextMF_dyn_ML
from utils import train as tr, parser_maker as pm, load_and_process_data as lpd, load_model as lm
import torch
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout
from datetime import datetime
import os

def main():
    parser = pm.parser_make()
    args = parser.parse_args()

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    question_embeddings = torch.load(args.question_embedding_path)
    num_prompts = question_embeddings.shape[0]
    num_models = len(test_data["model_id"].unique())
    
    # 设置训练模型数量 (前90个)
    num_train_models = 90

    # 只使用训练数据构建响应矩阵 (避免数据泄露)
    print("Building train response matrix for graph construction...")
    train_response_matrix, train_model_ids, all_prompt_ids = lpd.build_train_response_matrix(
        train_data, num_train_models=num_train_models
    )
    
    # 获取未见模型的响应矩阵
    print("Building unseen model responses...")
    unseen_model_ids = sorted(test_data["model_id"].unique())[num_train_models:]
    unseen_responses = lpd.get_unseen_model_responses(test_data, unseen_model_ids, all_prompt_ids)
    
    # 加载动态嵌入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_embeddings = torch.load("data/model_embeddings_static.pth", map_location=device)
    
    # 只使用训练模型的嵌入构建图
    train_model_embeddings = model_embeddings[:num_train_models]
    
    # 构建图数据 (只使用训练数据)
    print("Building model graph using only training data...")
    graph_data = build_model_graph(train_response_matrix, train_model_embeddings, 
                                 k_neighbors=10, device=device)
    
    # 加载训练和测试数据
    train_loader, test_loader = lpd.load_and_process_data(
        train_data=train_data, test_data=test_data, batch_size=args.batch_size,
        model_use_train_l=args.model_use_train_l, model_use_train_r=args.model_use_train_r,
        model_use_test_l=args.model_use_test_l, model_use_test_r=args.model_use_test_r
    )
    
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print("Initializing GraphSAGE model...")
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
            num_layers=2,
            num_train_models=num_train_models
        )
        model.to(device)
        
        # 设置训练响应矩阵
        model.set_train_responses(train_response_matrix)
        
        print(f'train_l : {args.model_use_train_l} , train_r : {args.model_use_train_r} , test_l : {args.model_use_test_l} , test_r : {args.model_use_test_r}')
        print(f'GraphSAGE parameters: hidden_dim=256, num_layers=2, k_neighbors=10')
        print(f'Dynamic embedding: is_dyn={args.is_dyn}, frozen={args.frozen}')
        print(f'Training models: {num_train_models}, Total models: {num_models}')
        print(f'Graph construction: Only using training data (no data leakage)')

        print("Training GraphSAGE model...")
        max_acc = tr.train_graphsage(
            model, graph_data, train_loader, test_loader,
            num_epochs=args.num_epochs, lr=args.learning_rate,
            device=device, unseen_responses=unseen_responses,
            save_path=args.model_save_path
        )
    
    # 保存输出结果
    folder_path = "output/" + datetime.now().strftime("%Y-%m-%d")
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + "/" + f"GraphSAGE_dyn_train_l_{args.model_use_train_l}_train_r_{args.model_use_train_r}_test_l_{args.model_use_test_l}_test_r_{args.model_use_test_r}_dyn_{args.is_dyn}_frozen_{args.frozen}.txt"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(captured_output.getvalue())
    
    print(f"Results saved to {file_path}")
    print(f"Maximum accuracy achieved: {max_acc:.4f}")

if __name__ == "__main__":
    main()
