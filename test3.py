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
def train(net, train_loader, test_loader0,test_loader1,test_loader2, num_epochs, lr, device, weight_decay=1e-5, save_path=None):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)
    max_accuracy = 0

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for models, prompts, labels in train_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

        test_loss, test_accuracy = eva.evaluate(net, test_loader, device)
        max_accuracy = max(max_accuracy,test_accuracy)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)

    print(f'Max accuracy : {max_accuracy}')


    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return max_accuracy
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
    train_loader , test_loader0 = lpd.load_and_process_data(train_data=train_data,test_data=test_data,batch_size=args.batch_size,model_use_train_l=0,model_use_train_r=112,model_use_test_l=0,model_use_test_r=112,shuffle=False)
    _,test_loader1 = lpd.load_and_process_data(train_data=train_data,test_data=test_data,batch_size=args.batch_size,model_use_train_l=0,model_use_train_r=112,model_use_test_l=0,model_use_test_r=90,shuffle=False)
    _,test_loader2 = lpd.load_and_process_data(train_data=train_data,test_data=test_data,batch_size=args.batch_size,model_use_train_l=0,model_use_train_r=112,model_use_test_l=90,model_use_test_r=112,shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_embeddings = torch.load("data/model_embeddings_static.pth",map_location=device)
    
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        print("Initializing model...")
        net = TextMF_dyn(question_embeddings=question_embeddings, 
                   model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                   num_models=num_models, num_prompts=num_prompts,model_embeddings=model_embeddings,is_dyn=args.is_dyn,frozen = args.frozen)
       
        net.to(device)
        
        
        # tr.train(model, train_loader, test_loader, num_epochs=args.num_epochs, lr=args.learning_rate,
        #    device=device, save_path=args.model_save_path)
        optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        progress_bar = tqdm(total=args.num_epochs)

        for epoch in range(args.num_epochs):
            net.train()
            total_loss = 0
            for models, prompts, labels in train_loader:
                models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = net(models, prompts)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss}")

            test_loss0, test_accuracy0 = eva.evaluate(net, test_loader0, device)
            print(f"Test Loss: {test_loss0}, Test Accuracy: {test_accuracy0}")
            
            test_loss1, test_accuracy1 = eva.evaluate(net, test_loader1, device)
            print(f"Test Loss: {test_loss1}, Test Accuracy: {test_accuracy1}")

            test_loss2, test_accuracy2 = eva.evaluate(net, test_loader2, device)
            print(f"Test Loss: {test_loss2}, Test Accuracy: {test_accuracy2}")

            progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss0, test_acc=test_accuracy0)
            progress_bar.update(1)

    folder_path = "output/" + datetime.now().strftime("%Y-%m-%d")
    os.makedirs(folder_path,exist_ok=True)
    file_path = folder_path + "/" + "why_i_get_2_lower_acc.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(captured_output.getvalue())