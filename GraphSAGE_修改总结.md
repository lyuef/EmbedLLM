# GraphSAGE模型修改总结

## 修改目标
解决原始实现中的数据泄露问题：构建图时不应该使用测试集数据，应该只使用训练集数据。同时实现对未见模型的合理表示学习。

## 核心修改方案

### 1. 图构建策略修改
**原方案**: 使用所有模型（训练+测试）的数据构建图
**新方案**: 只使用前90个训练模型的数据构建图

- 避免数据泄露
- 更符合真实场景的冷启动问题
- 图构建完全基于训练数据

### 2. 未见模型表示生成
对于测试阶段的未见模型（模型91-112）：
1. 获取该模型对所有prompt的01回答向量
2. 计算与前90个训练模型的余弦相似度
3. 选择top-k个最相似的训练模型
4. 用相似度加权聚合这些模型的表示作为未见模型的表示

## 详细修改内容

### 1. 模型架构修改 (`models/Embedllm_GraphSAGE.py`)

#### 新增参数
- `num_train_models`: 训练模型数量（默认90）
- `train_responses`: 存储训练模型响应矩阵

#### 新增方法
- `set_train_responses()`: 设置训练模型响应矩阵
- `get_unseen_model_representation()`: 为未见模型生成表示

#### 前向传播逻辑修改
```python
def forward(self, graph_data, model_ids, prompt_ids, unseen_responses=None, test_mode=False):
    # 1. 对训练模型使用GraphSAGE更新嵌入
    # 2. 分别处理训练模型和未见模型
    #    - 训练模型：使用GraphSAGE增强的嵌入
    #    - 未见模型：使用相似度聚合的表示
```

### 2. 数据处理函数修改 (`utils/load_and_process_data.py`)

#### 新增函数
- `build_train_response_matrix()`: 只使用训练数据构建响应矩阵
- `get_unseen_model_responses()`: 获取未见模型的响应矩阵

### 3. 评估函数修改 (`utils/evaluate.py`)

#### 修改评估逻辑
- `evaluate_graphsage()`: 支持未见模型响应矩阵参数
- 动态为batch中的未见模型准备响应数据

### 4. 训练函数修改 (`utils/train.py`)

#### 修改训练逻辑
- `train_graphsage()`: 支持未见模型响应参数
- 训练时只使用训练模型，测试时支持未见模型

### 5. 主训练脚本修改 (`GraphSAGEMF_train.py`)

#### 修改数据流程
```python
# 1. 只使用训练数据构建响应矩阵
train_response_matrix, train_model_ids, all_prompt_ids = build_train_response_matrix(train_data, 90)

# 2. 获取未见模型响应
unseen_responses = get_unseen_model_responses(test_data, unseen_model_ids, all_prompt_ids)

# 3. 只使用训练模型嵌入构建图
train_model_embeddings = model_embeddings[:90]
graph_data = build_model_graph(train_response_matrix, train_model_embeddings, ...)

# 4. 设置训练响应矩阵
model.set_train_responses(train_response_matrix)
```

## 技术细节

### 相似度计算
- 使用余弦相似度计算未见模型与训练模型的相似度
- 基于模型对所有prompt的01响应向量

### 表示聚合
- 使用softmax归一化相似度作为权重
- 加权聚合top-k个最相似训练模型的表示

### 图构建
- 组合响应相似度（70%）和嵌入相似度（30%）
- 构建k-NN图（k=10）

## 优势

1. **避免数据泄露**: 图构建完全基于训练数据
2. **更现实的设置**: 模拟真实场景中的冷启动问题
3. **灵活的表示学习**: 为未见模型提供合理的初始表示
4. **保持性能**: 训练模型仍然使用GraphSAGE增强

## 测试验证

创建了 `test_modified_graphsage.py` 测试脚本，验证：
- 训练响应矩阵构建
- 未见模型响应获取
- 图数据构建
- 模型前向传播
- 相似度聚合功能

## 使用方法

运行修改后的训练脚本：
```bash
python GraphSAGEMF_train.py --train_data_path data/train.csv --test_data_path data/test.csv --question_embedding_path data/question_embeddings.pth
```

模型会自动：
1. 使用前90个模型构建图
2. 为未见模型生成基于相似度的表示
3. 在训练和测试中正确处理不同类型的模型
