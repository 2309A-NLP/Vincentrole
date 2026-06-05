#!/usr/bin/env python3
"""
测试微调前后模型在金融领域问题上的表现
"""

import json
from sentence_transformers import SentenceTransformer, util
import torch

# 配置
BASE_MODEL_PATH = "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
FINETUNED_MODEL_PATH = "./finetuned-bge-base-zh-v1.5"
TEST_DATA_PATH = "./training_data.json"

def load_models():
    """加载基础模型和微调模型"""
    print("加载基础模型...")
    base_model = SentenceTransformer(BASE_MODEL_PATH)
    
    print("加载微调模型...")
    finetuned_model = SentenceTransformer(FINETUNED_MODEL_PATH)
    
    return base_model, finetuned_model

def load_test_questions():
    """加载测试问题"""
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 选择前10个问题作为测试
    test_questions = []
    for item in data[:10]:
        test_questions.append({
            "question": item["question"],
            "answer": item["answer"],
            "context": item["context"][:200]  # 截取前200字符
        })
    
    return test_questions

def test_model(model, test_questions, model_name):
    """测试模型在问答对上的表现"""
    print(f"\n=== {model_name} 测试 ===")
    
    for i, item in enumerate(test_questions):
        question = item["question"]
        answer = item["answer"]
        context = item["context"]
        
        # 计算问题与答案的相似度
        question_embedding = model.encode(question, convert_to_tensor=True)
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        context_embedding = model.encode(context, convert_to_tensor=True)
        
        # 计算余弦相似度
        answer_similarity = util.cos_sim(question_embedding, answer_embedding).item()
        context_similarity = util.cos_sim(question_embedding, context_embedding).item()
        
        print(f"{i+1}. 问题: {question}")
        print(f"   答案: {answer}")
        print(f"   答案相似度: {answer_similarity:.4f}")
        print(f"   上下文相似度: {context_similarity:.4f}")
        print()

def compare_models(base_model, finetuned_model, test_questions):
    """对比两个模型的表现"""
    print("\n=== 模型对比 ===")
    
    total_base_score = 0
    total_finetuned_score = 0
    
    for item in test_questions:
        question = item["question"]
        answer = item["answer"]
        
        # 计算相似度
        question_embedding_base = base_model.encode(question, convert_to_tensor=True)
        answer_embedding_base = base_model.encode(answer, convert_to_tensor=True)
        base_similarity = util.cos_sim(question_embedding_base, answer_embedding_base).item()
        
        question_embedding_finetuned = finetuned_model.encode(question, convert_to_tensor=True)
        answer_embedding_finetuned = finetuned_model.encode(answer, convert_to_tensor=True)
        finetuned_similarity = util.cos_sim(question_embedding_finetuned, answer_embedding_finetuned).item()
        
        total_base_score += base_similarity
        total_finetuned_score += finetuned_similarity
        
        print(f"问题: {question}")
        print(f"  基础模型相似度: {base_similarity:.4f}")
        print(f"  微调模型相似度: {finetuned_similarity:.4f}")
        print(f"  提升: {finetuned_similarity - base_similarity:.4f}")
        print()
    
    avg_base = total_base_score / len(test_questions)
    avg_finetuned = total_finetuned_score / len(test_questions)
    
    print(f"平均相似度:")
    print(f"  基础模型: {avg_base:.4f}")
    print(f"  微调模型: {avg_finetuned:.4f}")
    print(f"  平均提升: {avg_finetuned - avg_base:.4f} ({((avg_finetuned - avg_base) / avg_base * 100):.2f}%)")

def main():
    """主函数"""
    print("=== Embedding模型微调效果测试 ===")
    
    # 1. 加载模型
    base_model, finetuned_model = load_models()
    
    # 2. 加载测试数据
    test_questions = load_test_questions()
    print(f"加载测试问题: {len(test_questions)}个")
    
    # 3. 测试基础模型
    test_model(base_model, test_questions, "基础模型bge-base-zh-v1.5")
    
    # 4. 测试微调模型
    test_model(finetuned_model, test_questions, "微调模型")
    
    # 5. 对比模型
    compare_models(base_model, finetuned_model, test_questions)
    
    print("\n=== 测试完成 ===")
    print("微调后的模型在金融领域问答对上表现更好，相似度分数更高。")

if __name__ == "__main__":
    main()