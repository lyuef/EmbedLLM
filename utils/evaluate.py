import torch 
from torch import nn
def evaluate(net, test_loader, device):
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for models, prompts, labels in test_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            pred_labels = net.predict(models, prompts)
            correct += (pred_labels == labels).sum().item()
            total_loss += loss.item()
            num_samples += labels.shape[0]

    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples
    net.train()
    return mean_loss, accuracy

def evaluate_graphsage(net, graph_data, test_loader, device, unseen_responses=None):
    """
    评估GraphSAGE模型
    Args:
        net: GraphSAGE_TextMF_dyn模型
        graph_data: 图数据
        test_loader: 测试数据加载器
        device: 设备
        unseen_responses: 未见模型的响应矩阵 [22, num_prompts] (只包含后22个未见模型)
    Returns:
        tuple: (平均损失, 准确率)
    """
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for models, prompts, labels in test_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            
            # 为当前batch准备未见模型响应
            batch_unseen_responses = None
            if unseen_responses is not None:
                # 获取当前batch中未见模型的响应
                unseen_mask = models >= net.num_train_models  # 模型ID >= 90的是未见模型
                if unseen_mask.sum() > 0:
                    # 构建当前batch的未见模型响应矩阵
                    batch_size = models.shape[0]
                    num_prompts = unseen_responses.shape[1]
                    batch_unseen_responses = torch.zeros(batch_size, num_prompts, device=device)
                    
                    for i, model_id in enumerate(models):
                        if model_id >= net.num_train_models:
                            # 未见模型：映射到unseen_responses中的正确索引
                            # model_id范围是90-111，需要映射到0-21
                            unseen_idx = model_id.item() - net.num_train_models
                            if 0 <= unseen_idx < unseen_responses.shape[0]:
                                batch_unseen_responses[i] = unseen_responses[unseen_idx].to(device)
            
            logits = net(graph_data, models, prompts, batch_unseen_responses, test_mode=True)
            loss = loss_fn(logits, labels)
            pred_labels = net.predict(graph_data, models, prompts, batch_unseen_responses)
            correct += (pred_labels == labels).sum().item()
            total_loss += loss.item()
            num_samples += labels.shape[0]

    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples
    net.train()
    return mean_loss, accuracy
