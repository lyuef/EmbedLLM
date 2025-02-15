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
    parser.add_argument("--loader",type = int,default=0)
    args = parser.parse_args(["--loader","3","--output_save_path","output/result.txt","--model_load_path","data/fir90_dyn.pth","--embedding_dim","1024"])

    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
    
        print("Loading dataset...")
        train_data = pd.read_csv(args.train_data_path)
        test_data = pd.read_csv(args.test_data_path)
        question_embeddings = torch.load(args.question_embedding_path)
        num_prompts = question_embeddings.shape[0]
        num_models = len(test_data["model_id"].unique())
        model_names = list(np.unique(list(test_data["model_name"])))

        train_loader0, test_loader0 = lpd.load_and_process_data(train_data, test_data, batch_size=args.batch_size , model_use_l = 0,model_use_r = 90)
        train_loader1, test_loader1 = lpd.load_and_process_data(train_data, test_data, batch_size=args.batch_size , model_use_l = 90,model_use_r = 112)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = train_loader0 
        if args.loader == 1:
            loader = train_loader1 
        elif args.loader == 2 :
            loader = test_loader0
        elif args.loader == 3 :
            loader = test_loader1
        print(f'loader : {args.loader}')

        print("Initializing model...")
        model = Embedllm_dynamic.TextMF_dyn(question_embeddings=question_embeddings, 
                        model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                        num_models=num_models, num_prompts=num_prompts)
        lm.load_model(model,args.model_load_path,device)

        print("Testing models...")
        test(model,loader,device=device)

    with open(args.output_save_path, "a", encoding="utf-8") as f:
        f.write(captured_output.getvalue())
        