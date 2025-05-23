import torch
from torch import nn
import torch.nn.functional as F
from Embedlllm_dynamic import TextMF_dyn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
class GraphSAGE_MF(nn.module) :
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2,model_embeddings = None,is_dyn = False):
        super().__init__()

        self.conv1 = SA