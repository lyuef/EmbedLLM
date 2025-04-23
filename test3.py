from models.Embedllm_dynamic import TextMF_dyn,TextMF_dyn_ML
from utils import train as tr , parser_maker as pm ,load_and_process_data as lpd , load_model as lm 
import torch 
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout
from datetime import datetime
import os
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
import utils.evaluate as eva 
# test 3 ranges of dataset at same time
if __name__ == "__main__" :
    parser = pm.parser_make()
    args = parser.parse_args()

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    question_embeddings = torch.load(args.question_embedding_path)
    num_prompts = question_embeddings.shape[0]
    num_models = len(test_data["model_id"].unique())
    model_names = list(np.unique(list(test_data["model_name"])))
    train_loader , test_loaders = lpd.load_and_process_data_mutitest(train_data,test_data,(0,112),(0,90),(90,112),batch_size=args.batch_size,model_use_train_l=args.model_use_train_l,model_use_train_r=args.model_use_train_r,shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_embeddings = torch.load("data/model_embeddings_static.pth",map_location=device)
    
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print("Initializing model...")
        model = TextMF_dyn(question_embeddings=question_embeddings, 
                   model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                   num_models=num_models, num_prompts=num_prompts,model_embeddings=model_embeddings,is_dyn=args.is_dyn,frozen = args.frozen)
       
        model.to(device)
        
        tr.train_mutitest(model, train_loader,  args.num_epochs, args.learning_rate,
            device, *test_loaders,save_path=args.model_save_path)
        

    folder_path = "output/" + datetime.now().strftime("%Y-%m-%d")
    os.makedirs(folder_path,exist_ok=True)
    file_path = folder_path + "/" + f"MF_dyn_train_l_{args.model_use_train_l}_train_r_{args.model_use_train_r}_dyn_{args.is_dyn}_frozen_{args.frozen}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(captured_output.getvalue())