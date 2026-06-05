#!/usr/bin/env python3
"""
微调bge-base-zh-v1.5模型用于金融领域检索
使用sentence-transformers库
"""

import json
import random
import os
from typing import List, Dict, Tuple
import numpy as np

# 检查依赖
try:
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    import torch
    print("依赖库导入成功")
except ImportError as e:
    print(f"请先安装依赖: pip install sentence-transformers torch")
    print(f"错误: {e}")
    exit(1)

# 配置
TRAINING_DATA_PATH = "./training_data.json"
BASE_MODEL_PATH = "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
OUTPUT_MODEL_PATH = "./finetuned-bge-base-zh-v1.5"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

def load_training_data() -> List[Dict]:
    """加载训练数据"""
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def create_training_examples(data: List[Dict]) -> List[InputExample]:
    """创建训练样本"""
    examples = []
    
    # 正例：问题-答案对
    for item in data:
        question = item["question"]
        answer = item["answer"]
        context = item["context"]
        
        # 使用上下文作为答案的扩展
        answer_with_context = f"{answer}。{context[:200]}"
        
        # 正例对（相似度1.0）
        examples.append(InputExample(texts=[question, answer_with_context], label=1.0))
    
    # 负例：随机配对
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    
    for i in range(len(data)):
        # 随机选择一个不同的答案作为负例
        j = random.randint(0, len(data)-1)
        while j == i:
            j = random.randint(0, len(data)-1)
        
        examples.append(InputExample(texts=[questions[i], answers[j]], label=0.0))
    
    return examples

def split_train_val(examples: List[InputExample], val_ratio: float = 0.2) -> Tuple[List[InputExample], List[InputExample]]:
    """划分训练集和验证集"""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - val_ratio))
    return examples[:split_idx], examples[split_idx:]

def main():
    """主函数"""
    print("=== Embedding模型微调 ===")
    
    # 1. 加载训练数据
    print("1. 加载训练数据...")
    training_data = load_training_data()
    print(f"   加载问答对: {len(training_data)}个")
    
    # 2. 创建训练样本
    print("2. 创建训练样本...")
    examples = create_training_examples(training_data)
    print(f"   创建训练样本: {len(examples)}个")
    
    # 3. 划分训练集和验证集
    print("3. 划分数据集...")
    train_examples, val_examples = split_train_val(examples)
    print(f"   训练集: {len(train_examples)}个, 验证集: {len(val_examples)}个")
    
    # 4. 加载基础模型
    print("4. 加载基础模型bge-base-zh-v1.5...")
    try:
        # 尝试从本地路径加载
        if os.path.exists(BASE_MODEL_PATH):
            model = SentenceTransformer(BASE_MODEL_PATH)
            print(f"   从本地加载: {BASE_MODEL_PATH}")
        else:
            # 从HuggingFace Hub加载
            model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
            print("   从HuggingFace Hub加载")
    except Exception as e:
        print(f"   模型加载失败: {e}")
        print("   尝试使用离线模式...")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
    
    # 5. 创建数据加载器
    print("5. 创建数据加载器...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 6. 定义损失函数
    print("6. 定义损失函数...")
    train_loss = losses.CosineSimilarityLoss(model)
    
    # 7. 创建评估器
    print("7. 创建评估器...")
    val_sentences1 = [ex.texts[0] for ex in val_examples]
    val_sentences2 = [ex.texts[1] for ex in val_examples]
    val_scores = [ex.label for ex in val_examples]
    
    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences1, val_sentences2, val_scores,
        name="financial-qa-eval"
    )
    
    # 8. 微调前评估
    print("8. 微调前评估...")
    baseline_score = evaluator(model)
    print(f"   基准分数: {baseline_score:.4f}")
    
    # 9. 微调训练
    print("9. 开始微调训练...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=1000,
        warmup_steps=100,
        output_path=OUTPUT_MODEL_PATH,
        save_best_model=True
    )
    
    # 10. 微调后评估
    print("10. 微调后评估...")
    model = SentenceTransformer(OUTPUT_MODEL_PATH)
    final_score = evaluator(model)
    print(f"    最终分数: {final_score:.4f}")
    
    # 11. 保存评估结果
    print("11. 保存评估结果...")
    eval_results = {
        "baseline_score": baseline_score,
        "final_score": final_score,
        "improvement": final_score - baseline_score,
        "improvement_percent": ((final_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0,
        "training_samples": len(train_examples),
        "validation_samples": len(val_examples),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }
    
    with open("./evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    print("\n=== 微调完成 ===")
    print(f"微调前分数: {baseline_score:.4f}")
    print(f"微调后分数: {final_score:.4f}")
    print(f"提升: {eval_results['improvement']:.4f} ({eval_results['improvement_percent']:.2f}%)")
    print(f"模型保存路径: {OUTPUT_MODEL_PATH}")
    print(f"评估结果保存路径: ./evaluation_results.json")

if __name__ == "__main__":
    main()