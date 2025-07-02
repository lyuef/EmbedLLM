import torch
from torch.optim import Adam
from torch import nn
import utils.evaluate as eva 
from tqdm import tqdm
def train(net, train_loader, test_loader, num_epochs, lr, device, weight_decay=1e-5, save_path=None):
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

def train_graphsage(net, graph_data, train_loader, test_loader, num_epochs, lr, device, 
                   unseen_responses=None, weight_decay=1e-5, save_path=None):
    """
    训练GraphSAGE模型
    Args:
        net: GraphSAGE_TextMF_dyn模型
        graph_data: 图数据 (只包含训练模型)
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        unseen_responses: 未见模型的响应矩阵 [num_unseen_models, num_prompts]
        weight_decay: 权重衰减
        save_path: 模型保存路径
    """
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)
    max_accuracy = 0

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        
        for models, prompts, labels in train_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            
            # 只训练前90个模型
            train_mask = models < net.num_train_models
            if train_mask.sum() == 0:
                continue
            
            optimizer.zero_grad()
            # 训练时不需要未见模型响应，因为只训练训练模型
            logits = net(graph_data, models[train_mask], prompts[train_mask], test_mode=False)
            loss = loss_fn(logits, labels[train_mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

        # 测试阶段 - 需要未见模型响应
        test_loss, test_accuracy = eva.evaluate_graphsage(net, graph_data, test_loader, device, unseen_responses)
        max_accuracy = max(max_accuracy, test_accuracy)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)

    print(f'Max accuracy : {max_accuracy}')

    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return max_accuracy

def train_graphsage_mutitest(net, graph_data, train_loader, num_epochs, lr, device, *test_loaders, weight_decay=1e-5, save_path=None):
    """
    训练GraphSAGE模型 - 多测试集版本
    """
    from models.Embedllm_GraphSAGE import update_graph_data
    
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)
    max_accuracy = [0] * len(test_loaders)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        
        # 每个epoch开始时更新图数据
        current_graph_data = update_graph_data(net, graph_data)
        current_graph_data = current_graph_data.to(device)
        
        for models, prompts, labels in train_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            
            # 只训练前90个模型
            train_mask = models < 90
            if train_mask.sum() == 0:
                continue
            
            optimizer.zero_grad()
            logits = net(current_graph_data, models[train_mask], prompts[train_mask])
            loss = loss_fn(logits, labels[train_mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")
        
        # 测试多个测试集
        test_graph_data = update_graph_data(net, graph_data)
        test_graph_data = test_graph_data.to(device)
        
        test_cnt = 0
        for test_loader in test_loaders:
            test_loss, test_accuracy = eva.evaluate_graphsage(net, test_graph_data, test_loader, device)
            max_accuracy[test_cnt] = max(max_accuracy[test_cnt], test_accuracy)
            print(f"Test : {test_cnt}, Loss: {test_loss}, Test Accuracy: {test_accuracy}")
            test_cnt += 1

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)
    
    for i in range(len(max_accuracy)):
        print(f'Test : {i+1},Max accuracy : {max_accuracy[i]}')

    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return max_accuracy
def train_mutitest(net, train_loader, num_epochs, lr, device, *test_loaders,weight_decay=1e-5, save_path=None):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)
    max_accuracy = [0]*len(test_loaders)

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
        test_cnt = 0
        for test_loader in test_loaders :
            test_loss, test_accuracy = eva.evaluate(net, test_loader, device)
            max_accuracy[test_cnt] = max(max_accuracy[test_cnt],test_accuracy)
            print(f"Test : {test_cnt}, Loss: {test_loss}, Test Accuracy: {test_accuracy}")
            test_cnt += 1 

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)
    for i in range(len(max_accuracy)):
        print(f'Test : {i+1},Max accuracy : {max_accuracy[i]}')


    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
