
import torch 
from torch.utils.data import Dataset,DataLoader
import pandas as pd 
from torch_geometric.data import Data,Dataset

def load_and_process_data(train_data, test_data, batch_size=64 , model_use_train_l = 0,model_use_train_r = 112 , model_use_test_l = 0 , model_use_test_r = 112,shuffle = True):
    
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    

    model_ids = test_data["model_id"].unique()
    if shuffle :
        model_ids = model_ids[torch.randperm(len(model_ids))]

    train_selected_model_ids = model_ids[model_use_train_l:model_use_train_r]
    test_selected_model_ids = model_ids[model_use_test_l:model_use_test_r]

    train_data = train_data[train_data["model_id"].isin(train_selected_model_ids)]
    test_data = test_data[test_data["model_id"].isin(test_selected_model_ids)]

    class CustomDataset(Dataset):
        def __init__(self, data):
            
            model_ids = torch.tensor(data["model_id"].to_numpy(), dtype=torch.int64)
            
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids

            self.prompts = torch.tensor(data["prompt_id"].to_numpy(), dtype=torch.int64)
            self.labels = torch.tensor(data["label"].to_numpy(), dtype=torch.int64)

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

def load_and_process_data_mutitest(train_data, test_data, *test_range,batch_size=64 , model_use_train_l = 0,model_use_train_r = 112 ,shuffle = True):
    """load multi test data"""
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    

    model_ids = test_data["model_id"].unique()
    if shuffle :
        model_ids = model_ids[torch.randperm(len(model_ids))]

    train_selected_model_ids = model_ids[model_use_train_l:model_use_train_r]
    num_tests = len(test_range)
    test_selected_model_ids = [model_ids[test_range[i][0]:test_range[i][1]] for i in range(num_tests)]

    train_data = train_data[train_data["model_id"].isin(train_selected_model_ids)]
    test_data = [test_data[test_data["model_id"].isin(test_selected_model_ids[i])] for i in range(num_tests)]

    class CustomDataset(Dataset):
        def __init__(self, data):
            
            model_ids = torch.tensor(data["model_id"].to_numpy(), dtype=torch.int64)
            
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids

            self.prompts = torch.tensor(data["prompt_id"].to_numpy(), dtype=torch.int64)
            self.labels = torch.tensor(data["label"].to_numpy(), dtype=torch.int64)

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
    test_dataset = [CustomDataset(test_data[i]) for i in range(num_tests)]

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = [test_dataset[i].get_dataloaders(batch_size) for i in range(num_tests)]

    return train_loader, tuple(test_loader)