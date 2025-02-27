import torch
from torch import nn
class TextMF_dyn(nn.Module):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2,model_embeddings = None,is_dyn = False,frozen = False):
        super(TextMF_dyn, self).__init__()
        # Model embedding network
        if is_dyn :
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not(frozen)) 
            self.P.weight.data.copy_(model_embeddings)
        else : 
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not(frozen))
        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, model_embedding_dim)

        # Noise/Regularization level
        self.alpha = alpha
        self.classifier = nn.Linear(model_embedding_dim, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in question embedding to reduce overfitting
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        return self.classifier(p * q)
    
    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) # During inference no noise is applied
        return torch.argmax(logits, dim=1)
class TextMF_dyn_ML(nn.Module):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2,model_embeddings = None,is_dyn = False,frozen = False):
        super(TextMF_dyn_ML, self).__init__()
        # Model embedding network
        if is_dyn :
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not(frozen)) 
            self.P.weight.data.copy_(model_embeddings)
        else : 
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not(frozen))
        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, model_embedding_dim)
        # add 4 full connected layers
        self.alpha = alpha
        self.fc1 = nn.Linear(model_embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)  
        self.fc4 = nn.Linear(128, 64)   

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in question embedding to reduce overfitting
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)

        q = torch.relu(self.fc1(p*q))
        q = self.ln1(q)
        q = torch.relu(self.fc2(q))
        q = self.ln2(q)
        q = torch.relu(self.fc3(q))
        return self.classifier(torch.relu(self.fc4(q)))
    
    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) # During inference no noise is applied
        return torch.argmax(logits, dim=1)