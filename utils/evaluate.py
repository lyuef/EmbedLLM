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