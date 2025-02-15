import torch
from torch.optim import Adam
from torch import nn
import utils.evaluate as eva 
from tqdm import tqdm
def train(net, train_loader, test_loader, num_epochs, lr, device, weight_decay=1e-5, save_path=None):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)

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
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)
    
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")