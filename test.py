import argparse
import random
import torch
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout
from utils import load_and_process_data as lpd,evaluate as eva ,load_model as lm ,parser_maker as pm
from models import Embedllm_dynamic

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def test(net,test_loader,device) :

    test_loss, test_accuracy = eva.evaluate(net, test_loader, device)
    print(f'Test Loss: {test_loss} , Test Accuracy: {test_accuracy}')

if __name__ == "__main__":
    parser = pm.parser_make()
    parser.add_argument("--model_use_l",type = int,default=0 )
    parser.add_argument("--model_use_r",type = int ,default= 112)
    args = parser.parse_args(["--model_use_l","90","--model_use_r","112","--output_save_path","output/result.txt","--model_load_path","data/fir90_dyn.pth","--embedding_dim","1024"])

    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
    
        print("Loading dataset...")
        train_data = pd.read_csv(args.train_data_path)
        test_data = pd.read_csv(args.test_data_path)
        question_embeddings = torch.load(args.question_embedding_path)
        num_prompts = question_embeddings.shape[0]
        num_models = len(test_data["model_id"].unique())
        model_names = list(np.unique(list(test_data["model_name"])))

        train_loader, test_loader = lpd.load_and_process_data(train_data, test_data, batch_size=args.batch_size , model_use_l = args.model_use_l,model_use_r = args.model_use_r)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'model_use_l : {args.model_use_l} , model_use_r : {args.model_use_r}')

        print("Initializing model...")
        model = Embedllm_dynamic.TextMF_dyn(question_embeddings=question_embeddings, 
                        model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                        num_models=num_models, num_prompts=num_prompts)
        lm.load_model(model,args.model_load_path,device)
        model.to(device)
        print("Testing models...")
        test(model,test_loader,device=device)

    with open(args.output_save_path, "a", encoding="utf-8") as f:
        f.write(captured_output.getvalue())
        