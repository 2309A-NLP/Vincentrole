# -*- coding: utf-8 -*-
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy
from datasets import Dataset

# 准备测试数据集
data_samples = {
    'question': ['患者失眠怎么办？'],
    'answer': ['建议喝莲子心茶...'], # 这里的回答来自你的系统输出
    'contexts': [['莲子心茶清心降火...', '失眠属于不寐...']], # 检索到的上下文
    'ground_truth': ['建议服用朱砂安神丸或咨询医生'] # 标准答案（如有）
}
dataset = Dataset.from_dict(data_samples)

# 评测
result = evaluate(dataset, metrics=[context_precision, answer_relevancy])
print(result)