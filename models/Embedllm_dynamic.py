import torch
from torch import nn
class TextMF_dyn(nn.Module):
    model_embedding_dim_origin = 232
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2,model_embeddings = None,is_dyn = False,frozen = False):
        super(TextMF_dyn, self).__init__()
        # Model embedding network
        self.model_proj = nn.Linear(model_embedding_dim,TextMF_dyn.model_embedding_dim_origin)
        if is_dyn :
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not(frozen)) 
            self.P.weight.data.copy_(model_embeddings)    
        else : 
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not(frozen))
        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, TextMF_dyn.model_embedding_dim_origin)

        # Noise/Regularization level
        self.alpha = alpha
        self.classifier = nn.Linear(TextMF_dyn.model_embedding_dim_origin, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        p = self.model_proj(p)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in question embedding to reduce overfitting
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        return self.classifier(p * q)
    
    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) 
        return torch.argmax(logits, dim=1)
class TextMF_dyn_ML(TextMF_dyn):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2, model_embeddings=None, is_dyn=False, frozen=False):
        super().__init__(question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim, num_classes, model_embeddings, is_dyn, frozen)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.model_embedding_dim_origin, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        p = self.model_proj(p)
        q = self.Q(prompt)
        if not test_mode:
            q += torch.rand_like(q)*self.alpha 
        q = self.text_proj(q)
        return self.fc_layers(p*q)  
    
    @torch.no_grad() 
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) 
        return torch.argmax(logits, dim=1)