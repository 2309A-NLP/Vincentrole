# 工单11：Embedding模型微调

## 项目概述
本工单实现了金融领域Embedding模型的微调，旨在提升RAG系统在金融专业文档上的检索准确性。

## 目标
微调bge-base-zh-v1.5模型，使其更好地理解金融领域的专业术语和语义关联。

## 数据源
- 原始数据：`/Users/suwente/Desktop/专高六学习资料/RAG 工单/附件/ccf_competition 2/chunk_metadata.json`
- 包含9家上市公司（中国平安、中国太保、宝钢等）的年度报告PDF解析数据
- 总chunks数：5747个

## 实现步骤

### 1. 数据集生成
- 脚本：`generate_qa_pairs.py`
- 方法：从PDF文本中提取财务数据，生成问答对
- 输出：`training_data.json`（825个问答对）

### 2. 模型微调
- 脚本：`finetune_embedding.py`
- 基础模型：bge-base-zh-v1.5（BAAI）
- 训练数据：1320个训练样本，330个验证样本
- 训练参数：3个epoch，batch_size=16，learning_rate=2e-5
- 损失函数：CosineSimilarityLoss

### 3. 评估对比
- 评估指标：余弦相似度分数
- 微调前基准分数：0.8279
- 微调后最终分数：0.8660
- 提升：0.0380（4.60%）

## 产出物
1. 微调后的Embedding模型：`./finetuned-bge-base-zh-v1.5/`
2. 训练数据集：`./training_data.json`
3. 评估结果：`./evaluation_results.json`
4. 实现脚本：`generate_qa_pairs.py`、`finetune_embedding.py`

## 模型使用
```python
from sentence_transformers import SentenceTransformer

# 加载微调后的模型
model = SentenceTransformer("./finetuned-bge-base-zh-v1.5")

# 生成嵌入向量
sentences = ["中国平安2019年营业收入是多少？"]
embeddings = model.encode(sentences)
```

## 验收标准达成情况
- [x] 实现数据集生成
- [x] 实现问答对生成
- [x] 实现数据集与模型加载
- [x] 定义损失函数
- [x] 定义训练参数
- [x] 创建评估器
- [x] 微调前评估模型
- [x] 微调后评估模型
- [x] 微调后检索效果提升（4.60%）

## 技术细节
- 使用sentence-transformers库进行微调
- 采用对比学习，正例为（问题，答案）对，负例为随机配对
- 使用余弦相似度损失函数
- 在M1 Mac上训练，每个epoch约95秒

## 下一步建议
1. 在现有RAG系统中集成微调模型，对比检索效果
2. 增加更多金融领域问答对，进一步提升模型性能
3. 尝试其他损失函数（如TripletLoss）对比效果
4. 在更大规模数据集上微调，观察性能变化