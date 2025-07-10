import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data

class GraphContrastiveWithNegatives(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=5):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
    
    def forward(self, node_embeddings, edge_index, edge_weights=None):
        """
        负样本挖掘的图对比学习
        Args:
            node_embeddings: [num_nodes, embed_dim] 节点表示
            edge_index: [2, num_edges] 边索引
            edge_weights: [num_edges] 边权重（可选）
        """
        num_nodes = node_embeddings.shape[0]
        num_edges = edge_index.shape[1]
        
        if num_edges == 0 or num_nodes < 2:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        # 构建邻接矩阵
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=node_embeddings.device)
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj_matrix[src, dst] = 1
            adj_matrix[dst, src] = 1  # 无向图
        
        embeddings_norm = F.normalize(node_embeddings, p=2, dim=1)
        total_loss = 0
        valid_edges = 0
        
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            # 正样本：相连的节点
            pos_sim = torch.dot(embeddings_norm[src], embeddings_norm[dst])
            
            # 负样本：随机选择不相连的节点
            non_neighbors = (adj_matrix[src] == 0).nonzero().squeeze()
            if len(non_neighbors.shape) == 0:  # 只有一个元素
                non_neighbors = non_neighbors.unsqueeze(0)
            
            if len(non_neighbors) > 0:
                # 选择负样本数量
                actual_negatives = min(self.num_negatives, len(non_neighbors))
                neg_indices = non_neighbors[torch.randperm(len(non_neighbors))[:actual_negatives]]
                
                # 计算负样本相似度
                neg_sims = torch.mm(embeddings_norm[src:src+1], embeddings_norm[neg_indices].t()).squeeze()
                if neg_sims.dim() == 0:  # 只有一个负样本
                    neg_sims = neg_sims.unsqueeze(0)
                
                # InfoNCE损失
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / self.temperature
                labels = torch.zeros(1, dtype=torch.long, device=node_embeddings.device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                
                total_loss += loss
                valid_edges += 1
        
        return total_loss / max(valid_edges, 1)

class GraphSAGE_TextMF_dyn(nn.Module):
    model_embedding_dim_origin = 232
    
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, 
                 text_dim=768, num_classes=2, model_embeddings=None, is_dyn=False, frozen=False,
                 hidden_dim=256, num_layers=2, num_train_models=90, gnn_type="GraphSAGE", num_heads=8):
        super(GraphSAGE_TextMF_dyn, self).__init__()
        
        # 动态模型嵌入 (类似TextMF_dyn)
        self.model_proj = nn.Linear(model_embedding_dim, GraphSAGE_TextMF_dyn.model_embedding_dim_origin)
        if is_dyn:
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not frozen)
            self.P.weight.data.copy_(model_embeddings)
        else:
            self.P = nn.Embedding(num_models, model_embedding_dim).requires_grad_(not frozen)
        
        # 图神经网络层 (支持GraphSAGE和GAT)
        self.gnn_type = gnn_type
        self.num_heads = num_heads
        self.gnn_layers = nn.ModuleList()
        
        if num_layers == 1:
            if gnn_type == "GAT":
                self.gnn_layers.append(GATConv(GraphSAGE_TextMF_dyn.model_embedding_dim_origin, 
                                             GraphSAGE_TextMF_dyn.model_embedding_dim_origin, heads=1))
            else:  # GraphSAGE
                self.gnn_layers.append(SAGEConv(GraphSAGE_TextMF_dyn.model_embedding_dim_origin, 
                                              GraphSAGE_TextMF_dyn.model_embedding_dim_origin, aggr='mean'))
        else:
            # 第一层：232 → hidden_dim
            if gnn_type == "GAT":
                self.gnn_layers.append(GATConv(GraphSAGE_TextMF_dyn.model_embedding_dim_origin, 
                                             hidden_dim, heads=num_heads))
            else:  # GraphSAGE
                self.gnn_layers.append(SAGEConv(GraphSAGE_TextMF_dyn.model_embedding_dim_origin, 
                                              hidden_dim, aggr='mean'))
            
            # 中间层
            for _ in range(num_layers - 2):
                if gnn_type == "GAT":
                    # GAT中间层：hidden_dim * num_heads → hidden_dim
                    self.gnn_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
                else:  # GraphSAGE
                    self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            
            # 最后一层：hidden_dim * num_heads → 232 (对于GAT)
            if gnn_type == "GAT":
                self.gnn_layers.append(GATConv(hidden_dim * num_heads, 
                                             GraphSAGE_TextMF_dyn.model_embedding_dim_origin, heads=1))
            else:  # GraphSAGE
                self.gnn_layers.append(SAGEConv(hidden_dim, 
                                              GraphSAGE_TextMF_dyn.model_embedding_dim_origin, aggr='mean'))
        
        # 问题嵌入和分类器 (与TextMF_dyn相同)
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, GraphSAGE_TextMF_dyn.model_embedding_dim_origin)
        self.classifier = nn.Linear(GraphSAGE_TextMF_dyn.model_embedding_dim_origin, num_classes)
        
        # 参数
        self.alpha = alpha
        self.is_dyn = is_dyn
        self.frozen = frozen
        self.num_models = num_models
        self.num_train_models = num_train_models  # 训练模型数量 (前90个)
        
        # 存储训练模型的响应矩阵，用于计算未见模型的相似度
        self.train_responses = None
        
        # 图对比学习模块
        self.graph_contrastive = GraphContrastiveWithNegatives(
            temperature=0.1, 
            num_negatives=5
        )
        
    def set_train_responses(self, train_responses):
        """set train response matrix"""
        self.train_responses = train_responses
        
    def get_unseen_model_representation(self, unseen_responses, k_neighbors=10):
        """
        为未见模型生成表示(trivial)
        Args:
            unseen_responses: [batch_size, num_prompts] 未见模型的响应向量
            k_neighbors: 使用的邻居数量
        Returns:
            torch.Tensor: [batch_size, model_embedding_dim_origin] 未见模型的表示
        """
        if self.train_responses is None:
            raise ValueError("use set_train_responses first.")
        
        device = unseen_responses.device
        batch_size = unseen_responses.shape[0]
        
        # 计算与训练模型的相似度
        unseen_norm = F.normalize(unseen_responses.float(), p=2, dim=1)
        train_norm = F.normalize(self.train_responses.float(), p=2, dim=1).to(device)
        
        # [batch_size, num_train_models]
        similarities = torch.mm(unseen_norm, train_norm.t())
        
        # 获取top-k相似的训练模型
        top_k_values, top_k_indices = torch.topk(similarities, k_neighbors, dim=1)
        
        # 获取训练模型的原始嵌入并投影 (不使用GraphSAGE增强，因为这里需要与训练时一致)
        with torch.no_grad():
            train_embeddings = self.P.weight[:self.num_train_models]  # 只取前90个训练模型
            train_embeddings = self.model_proj(train_embeddings)
            
        # 使用相似度加权聚合top-k邻居的表示
        # top_k_indices: [batch_size, k_neighbors]
        # top_k_values: [batch_size, k_neighbors]
        
        # 归一化相似度权重
        weights = F.softmax(top_k_values, dim=1)  # [batch_size, k_neighbors]
        
        # 获取邻居表示
        neighbor_embeddings = train_embeddings[top_k_indices]  # [batch_size, k_neighbors, embedding_dim]
        
        # 加权聚合
        unseen_representations = torch.sum(neighbor_embeddings * weights.unsqueeze(-1), dim=1)
        
        return unseen_representations
    
    def get_unseen_model_representation_with_graph(self, unseen_responses, enhanced_train_embeddings, graph_data, k_neighbors=5):
        """
        为未见模型构建扩展图并应用GraphSAGE生成表示
        Args:
            unseen_responses: [batch_size, num_prompts] 未见模型的响应向量
            enhanced_train_embeddings: [num_train_models, embedding_dim] GraphSAGE增强的训练模型嵌入
            graph_data: 原始图数据 (只包含训练模型)
            k_neighbors: 使用的邻居数量
        Returns:
            torch.Tensor: [batch_size, model_embedding_dim_origin] 未见模型的GraphSAGE增强表示
        """
        if self.train_responses is None:
            raise ValueError("use set_train_responses first.")
        
        device = unseen_responses.device
        batch_size = unseen_responses.shape[0]
        num_train = enhanced_train_embeddings.shape[0]
        
        # 1. 计算未见模型与训练模型的相似度
        unseen_norm = F.normalize(unseen_responses.float(), p=2, dim=1)
        train_norm = F.normalize(self.train_responses.float(), p=2, dim=1).to(device)
        similarities = torch.mm(unseen_norm, train_norm.t())
        
        # 2. 为每个未见模型找到top-k训练模型邻居
        top_k_values, top_k_indices = torch.topk(similarities, k_neighbors, dim=1)
        
        # 3. 为未见模型生成初始嵌入（基于相似度加权聚合）
        weights = F.softmax(top_k_values, dim=1)  # [batch_size, k_neighbors]
        neighbor_embeddings = enhanced_train_embeddings[top_k_indices]  # [batch_size, k_neighbors, embedding_dim]
        initial_unseen_embeddings = torch.sum(neighbor_embeddings * weights.unsqueeze(-1), dim=1)  # [batch_size, embedding_dim]
        
        # 4. 构建扩展的节点特征矩阵
        extended_x = torch.cat([enhanced_train_embeddings, initial_unseen_embeddings], dim=0)  # [num_train + batch_size, embedding_dim]
        
        # 5. 构建扩展的边索引（添加未见模型到训练模型的连接）
        new_edges = []
        edge_weights = []
        
        for i, (top_k_idx, top_k_val) in enumerate(zip(top_k_indices, top_k_values)):
            unseen_node_id = num_train + i
            for j, (train_node_id, similarity_score) in enumerate(zip(top_k_idx, top_k_val)):
                # 双向连接：未见模型 <-> 训练模型
                new_edges.extend([
                    [unseen_node_id, train_node_id.item()],
                    [train_node_id.item(), unseen_node_id]
                ])
                # 使用归一化的相似度作为边权重
                weight = weights[i, j].item()
                edge_weights.extend([weight, weight])
        
        # 6. 合并原始图的边和新边
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(device)
            extended_edge_index = torch.cat([graph_data.edge_index, new_edge_index], dim=1)
            
            # 合并边权重（原始图的边权重设为1.0）
            original_edge_weights = torch.ones(graph_data.edge_index.shape[1], device=device)
            new_edge_weights = torch.tensor(edge_weights, dtype=torch.float, device=device)
            extended_edge_weights = torch.cat([original_edge_weights, new_edge_weights], dim=0)
        else:
            extended_edge_index = graph_data.edge_index
            extended_edge_weights = torch.ones(graph_data.edge_index.shape[1], device=device)
        
        # 7. 对扩展图应用GNN（只更新未见模型的嵌入）
        x = extended_x
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, extended_edge_index)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
        
        # 8. 返回未见模型的GraphSAGE增强嵌入
        unseen_enhanced_embeddings = x[num_train:]  # [batch_size, embedding_dim]
        
        return unseen_enhanced_embeddings
        
    def forward(self, graph_data, model_ids, prompt_ids, unseen_responses=None, test_mode=False):
        """
        前向传播
        Args:
            graph_data: 图数据 (只包含训练模型)
            model_ids: 模型ID
            prompt_ids: 问题ID
            unseen_responses: 未见模型的响应矩阵 [batch_size, num_prompts]
            test_mode: 是否为测试模式
        """
        # 1. 获取当前的动态嵌入
        current_embeddings = self.P.weight  # [num_models, embedding_dim]
        current_embeddings = self.model_proj(current_embeddings)  # 投影到统一维度
        
        # 2. 对训练模型使用GraphSAGE更新嵌入
        train_embeddings = current_embeddings[:self.num_train_models]  # 只取前90个训练模型
        edge_index = graph_data.edge_index
        
        x = train_embeddings
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 更新训练模型的表示
        enhanced_train_embeddings = x  # [num_train_models, model_embedding_dim_origin]

        # 3. 处理模型嵌入
        batch_size = model_ids.shape[0]
        model_embeddings = torch.zeros(batch_size, GraphSAGE_TextMF_dyn.model_embedding_dim_origin, 
                                     device=model_ids.device)
        
        # 分别处理训练模型和未见模型
        train_mask = model_ids < self.num_train_models
        unseen_mask = model_ids >= self.num_train_models
        
        # 训练模型：直接使用GraphSAGE增强的嵌入
        if train_mask.sum() > 0:
            train_model_ids = model_ids[train_mask]
            model_embeddings[train_mask] = enhanced_train_embeddings[train_model_ids]
        
        # 未见模型：使用GraphSAGE增强的相似度聚合表示
        if unseen_mask.sum() > 0:
            if unseen_responses is None:
                # 如果没有提供未见模型响应，使用原始嵌入
                unseen_model_ids = model_ids[unseen_mask]
                model_embeddings[unseen_mask] = current_embeddings[unseen_model_ids]
            else:
                # 使用GraphSAGE增强的响应相似度生成表示
                unseen_batch_responses = unseen_responses[unseen_mask]
                unseen_representations = self.get_unseen_model_representation_with_graph(
                    unseen_batch_responses, enhanced_train_embeddings, graph_data
                )
                model_embeddings[unseen_mask] = unseen_representations
        
        # 4. 问题嵌入处理
        q = self.Q(prompt_ids)
        if not test_mode:
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        
        # 5. 分类
        logits = self.classifier(model_embeddings * q)
        
        # 6. 图对比学习损失（仅在训练时计算）
        if not test_mode:
            graph_contrastive_loss = self.graph_contrastive(
                enhanced_train_embeddings,
                graph_data.edge_index,
                graph_data.edge_attr
            )
            return logits, graph_contrastive_loss
        else:
            return logits
    
    @torch.no_grad()
    def predict(self, graph_data, model_ids, prompt_ids, unseen_responses=None):
        logits = self.forward(graph_data, model_ids, prompt_ids, unseen_responses, test_mode=True)
        return torch.argmax(logits, dim=1)

def build_model_graph(model_responses, model_embeddings, k_neighbors=10, device='cpu'):
    """
    构建模型间的相似性图
    Args:
        model_responses: [num_models, num_prompts] 模型响应矩阵
        model_embeddings: [num_models, embedding_dim] 预训练嵌入
        k_neighbors: 每个节点的邻居数量
        device: 设备
    Returns:
        Data: PyTorch Geometric图数据对象
    """
    # 确保所有张量都在同一设备上
    model_responses = model_responses.to(device)
    model_embeddings = model_embeddings.to(device)
    
    num_models = model_responses.shape[0]
    
    # 计算响应相似度 (余弦相似度)
    model_responses_norm = F.normalize(model_responses.float(), p=2, dim=1)
    response_sim = torch.mm(model_responses_norm, model_responses_norm.t())
    
    # 计算嵌入相似度
    # model_embeddings_norm = F.normalize(model_embeddings.float(), p=2, dim=1)
    # embed_sim = torch.mm(model_embeddings_norm, model_embeddings_norm.t())
    
    # 组合相似度 
    similarity =  response_sim # + 0.3 * embed_sim
    
    # 构建k-NN图
    edge_list = []
    edge_weights = []
    
    for i in range(num_models):
        # 获取top-k相似的邻居 (排除自己)
        sim_scores = similarity[i]
        sim_scores[i] = -1  # 排除自己
        
        top_k_indices = torch.topk(sim_scores, k_neighbors).indices
        
        for j in top_k_indices:
            edge_list.append([i, j.item()])
            edge_weights.append(similarity[i, j].item())
    
    # 转换为PyTorch Geometric格式
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # 创建图数据对象
    graph_data = Data(
        x=model_embeddings.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_weights.to(device)
    )
    
    return graph_data

def update_graph_data(model, original_graph_data):
    """
    使用当前的动态嵌入更新图数据
    """
    # 获取当前的动态嵌入
    current_embeddings = model.P.weight.detach()
    
    # 更新图数据的节点特征
    updated_graph_data = Data(
        x=current_embeddings,
        edge_index=original_graph_data.edge_index,
        edge_attr=original_graph_data.edge_attr
    )
    
    return updated_graph_data
