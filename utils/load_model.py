import torch 
def load_model(net, path, device):
    net.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")