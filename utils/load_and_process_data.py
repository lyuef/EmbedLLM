
import torch 
import torch.nn.functional as F
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

def build_cograph(responses):
    """build cograph from responses"""
    n_models = responses.shape[0]
        
    # likely matrix
    agreement = torch.mm(responses.float(), responses.float().T)
    total = responses.shape[1]
    likely = agreement / total 
    likely_norm = F.softmax(likely, dim=1)
        
    src = torch.repeat_interleave(torch.arange(n_models), n_models)
    dst = torch.arange(n_models).repeat(n_models)
    edge_index = torch.stack([src, dst], dim=0)
        
    return Data(
        x=responses,
        edge_index=edge_index,
        edge_weight=likely_norm.flatten()
    )

def build_response(data) : 
    """构建模型响应矩阵"""
    model_ids = data["model_id"].unique()
    
    num_prompts = int(max(data["prompt_id"])) + 1

    required_cols = ['model_id','prompt_id','label']
    data = data[required_cols]
    return torch.tensor(data.pivot(index='model_id', columns='prompt_id', values='label').fillna(0).values, dtype=torch.float32)

def build_full_response_matrix(train_data, test_data):
    """
    构建包含所有模型的完整响应矩阵
    Args:
        train_data: 训练数据
        test_data: 测试数据
    Returns:
        torch.Tensor: [num_models, num_prompts] 响应矩阵
    """
    # 合并训练和测试数据
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # 获取所有唯一的模型ID和prompt ID
    all_model_ids = sorted(all_data["model_id"].unique())
    all_prompt_ids = sorted(all_data["prompt_id"].unique())
    
    # 创建完整的响应矩阵
    response_matrix = torch.zeros(len(all_model_ids), len(all_prompt_ids), dtype=torch.float32)
    
    # 填充响应矩阵
    for _, row in all_data.iterrows():
        model_idx = all_model_ids.index(row["model_id"])
        prompt_idx = all_prompt_ids.index(row["prompt_id"])
        response_matrix[model_idx, prompt_idx] = row["label"]
    
    return response_matrix, all_model_ids, all_prompt_ids

def build_train_response_matrix(train_data, num_train_models=90):
    """
    只使用训练数据构建响应矩阵 (避免数据泄露)
    Args:
        train_data: 训练数据
        num_train_models: 训练模型数量
    Returns:
        torch.Tensor: [num_train_models, num_prompts] 训练模型响应矩阵
        list: 训练模型ID列表
        list: prompt ID列表
    """
    # 获取前num_train_models个模型的数据
    train_model_ids = sorted(train_data["model_id"].unique())[:num_train_models]
    train_data_filtered = train_data[train_data["model_id"].isin(train_model_ids)]
    
    # 获取所有prompt ID
    all_prompt_ids = sorted(train_data_filtered["prompt_id"].unique())
    
    # 创建训练响应矩阵
    response_matrix = torch.zeros(len(train_model_ids), len(all_prompt_ids), dtype=torch.float32)
    
    # 填充响应矩阵
    for _, row in train_data_filtered.iterrows():
        model_idx = train_model_ids.index(row["model_id"])
        prompt_idx = all_prompt_ids.index(row["prompt_id"])
        response_matrix[model_idx, prompt_idx] = row["label"]
    
    return response_matrix, train_model_ids, all_prompt_ids

def get_unseen_model_responses(test_data, unseen_model_ids, all_prompt_ids):
    """
    获取未见模型的响应矩阵
    Args:
        test_data: 测试数据
        unseen_model_ids: 未见模型ID列表
        all_prompt_ids: 所有prompt ID列表
    Returns:
        torch.Tensor: [num_unseen_models, num_prompts] 未见模型响应矩阵
    """
    # 过滤未见模型的数据
    unseen_data = test_data[test_data["model_id"].isin(unseen_model_ids)]
    
    # 创建未见模型响应矩阵
    response_matrix = torch.zeros(len(unseen_model_ids), len(all_prompt_ids), dtype=torch.float32)
    
    # 填充响应矩阵
    for _, row in unseen_data.iterrows():
        if row["model_id"] in unseen_model_ids and row["prompt_id"] in all_prompt_ids:
            model_idx = unseen_model_ids.index(row["model_id"])
            prompt_idx = all_prompt_ids.index(row["prompt_id"])
            response_matrix[model_idx, prompt_idx] = row["label"]
    
    return response_matrix



def load_and_process_data_cograph(train_data, test_data, batch_size=64 , model_use_train_l = 0,model_use_train_r = 112 , model_use_test_l = 0 , model_use_test_r = 112,shuffle = True):
    """load and process data for cograph"""
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    

    model_ids = test_data["model_id"].unique()
    if shuffle :
        model_ids = model_ids[torch.randperm(len(model_ids))]

    train_selected_model_ids = model_ids[model_use_train_l:model_use_train_r]
    test_selected_model_ids = model_ids[model_use_test_l:model_use_test_r]

    train_data = train_data[train_data["model_id"].isin(train_selected_model_ids)]
    test_data = test_data[test_data["model_id"].isin(test_selected_model_ids)]

    train_responses = build_response(train_data)
    graphdata = build_cograph(train_responses)

    return graphdata
