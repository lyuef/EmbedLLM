
import torch 
from torch.utils.data import Dataset,DataLoader
import pandas as pd 

def load_and_process_data(train_data, test_data, batch_size=64 , model_use_l = 0,model_use_r = 112):
    
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1

    model_ids = test_data["model_id"].unique()
    selected_model_ids = model_ids[model_use_l:model_use_r]

    train_data = train_data[train_data["model_id"].isin(selected_model_ids)]
    test_data = test_data[test_data["model_id"].isin(selected_model_ids)]

    class CustomDataset(Dataset):
        def __init__(self, data):
            model_ids = pd.factorize(data["model_id"])[0]  # turn model_ids into int
            model_ids = torch.tensor(model_ids, dtype=torch.int64)
            
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids

            prompts = pd.factorize(data["prompt_id"])[0]
            self.prompts = torch.tensor(prompts, dtype=torch.int64)
            labels = pd.factorize(data["label"])[0]
            self.labels = torch.tensor(labels, dtype=torch.int64)

            self.num_models = len(data["model_id"].unique())
            self.num_prompts = num_prompts
            self.num_classes = len(data["label"].unique())

        def get_num_models(self):
            return self.num_models

        def get_num_prompts(self):
            return self.num_prompts

        def get_num_classes(self):
            return self.num_classes

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return self.models[index], self.prompts[index], self.labels[index]

        def get_dataloaders(self, batch_size):
            return DataLoader(self, batch_size, shuffle=False)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = test_dataset.get_dataloaders(batch_size)

    return train_loader, test_loader
