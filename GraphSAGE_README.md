# GraphSAGE + TextMF 动态嵌入模型

本文档介绍如何使用GraphSAGE增强的TextMF模型来提高对未见模型的泛化能力。

##  目标

通过GraphSAGE学习模型间的关系图，提高在未见模型上的预测准确率：
- 训练集：前90个模型
- 测试集：后22个未见模型
- 目标：从73%提升到75-80%的准确率

##  架构设计

### 核心组件

1. **GraphSAGE_TextMF_dyn**: 支持动态嵌入的GraphSAGE模型
2. **图构建**: 基于余弦相似度的模型关系图
3. **动态嵌入**: 支持预训练嵌入的动态更新
4. **联合训练**: 图网络和分类器端到端训练

### 模型流程

```
预训练模型嵌入 → GraphSAGE聚合 → 增强嵌入 → TextMF分类器 → 预测结果
```

## 📁 文件结构

```
models/
├── Embedllm_GraphSAGE.py      # GraphSAGE模型实现
├── Embedllm_dynamic.py        # 原始动态嵌入模型

utils/
├── load_and_process_data.py   # 数据处理（新增图数据构建）
├── train.py                   # 训练函数（新增GraphSAGE训练）
├── evaluate.py                # 评估函数（新增GraphSAGE评估）

GraphSAGEMF_train.py           # GraphSAGE训练脚本
test_graphsage.py              # 测试脚本
run_graphsage_example.py       # 使用示例
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install torch torch-geometric pandas numpy tqdm
```

### 2. 数据准备

确保以下文件存在：
- `data/train_data.csv`: 训练数据
- `data/test_data.csv`: 测试数据  
- `data/question_embeddings.pth`: 问题嵌入
- `data/model_embeddings_static.pth`: 预训练模型嵌入

### 3. 运行测试

```bash
python test_graphsage.py
```

### 4. 训练模型

#### 使用命令行参数：

```bash
python GraphSAGEMF_train.py \
    --train_data_path data/train_data.csv \
    --test_data_path data/test_data.csv \
    --question_embedding_path data/question_embeddings.pth \
    --embedding_dim 768 \
    --alpha 0.1 \
    --batch_size 64 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_use_train_l 0 \
    --model_use_train_r 90 \
    --model_use_test_l 90 \
    --model_use_test_r 112 \
    --is_dyn True \
    --frozen False \
    --model_save_path data/graphsage_model.pth
```

#### 使用示例脚本：

```bash
python run_graphsage_example.py
```

## ⚙️ 关键参数

### GraphSAGE参数
- `hidden_dim`: 隐藏层维度 (默认: 256)
- `num_layers`: GraphSAGE层数 (默认: 2)
- `k_neighbors`: 每个节点的邻居数 (默认: 10)

### 动态嵌入参数
- `is_dyn`: 是否使用动态嵌入 (True/False)
- `frozen`: 是否冻结嵌入参数 (True/False)

### 图构建参数
- 响应相似度权重: 0.7
- 嵌入相似度权重: 0.3
- 聚合函数: mean

## 📊 实验配置

### 基准对比

1. **TextMF (基础)**: ~68% 准确率
2. **TextMF_dyn (动态)**: ~73% 准确率  
3. **GraphSAGE_TextMF_dyn (目标)**: 75-80% 准确率

### 训练策略

- **训练集**: 模型ID 0-89 (90个模型)
- **测试集**: 模型ID 90-111 (22个未见模型)
- **图构建**: 使用全部112个模型的响应数据
- **联合训练**: 同时优化图网络和分类器

## 🔧 自定义使用

### 创建自定义模型

```python
from models.Embedllm_GraphSAGE import GraphSAGE_TextMF_dyn, build_model_graph

# 构建图数据
graph_data = build_model_graph(response_matrix, model_embeddings, k_neighbors=10)

# 创建模型
model = GraphSAGE_TextMF_dyn(
    question_embeddings=question_embeddings,
    model_embedding_dim=768,
    alpha=0.1,
    num_models=112,
    num_prompts=35000,
    model_embeddings=model_embeddings,
    is_dyn=True,
    frozen=False,
    hidden_dim=256,
    num_layers=2
)
```

### 自定义训练

```python
from utils.train import train_graphsage

max_acc = train_graphsage(
    model, graph_data, train_loader, test_loader,
    num_epochs=50, lr=0.001, device=device
)
```

## 🎯 预期改进

### 技术优势
- **全局信息利用**: 通过图结构学习模型间关系
- **冷启动处理**: 为未见模型生成更好的表示
- **端到端优化**: 图网络和分类器联合训练

### 性能提升
- **准确率提升**: 预期从73%提升到75-80%
- **泛化能力**: 对未见模型的预测更准确
- **鲁棒性**: 对数据噪声更不敏感

## 🐛 故障排除

### 常见问题

1. **内存不足**: 减少batch_size或k_neighbors
2. **CUDA错误**: 检查PyTorch Geometric安装
3. **维度不匹配**: 确保嵌入维度一致

### 调试技巧

```python
# 检查图数据
print(f"Graph nodes: {graph_data.x.shape}")
print(f"Graph edges: {graph_data.edge_index.shape}")

# 检查模型输出
logits = model(graph_data, model_ids, prompt_ids)
print(f"Logits shape: {logits.shape}")
```

## 📈 性能监控

训练过程中会输出：
- 每个epoch的训练损失
- 测试集损失和准确率
- 最终最高准确率

结果保存在 `output/YYYY-MM-DD/` 目录下。

## 🤝 贡献

欢迎提交问题和改进建议！主要改进方向：
- 不同聚合函数的实验
- 图构建策略优化
- 超参数调优
- 性能优化

---

**注意**: 请确保所有数据文件路径正确，并根据实际数据调整参数设置。
