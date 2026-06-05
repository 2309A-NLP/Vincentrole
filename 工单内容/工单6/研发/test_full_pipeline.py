"""
完整流程测试：图像检索 → Kimi多模态生成
"""
import sys
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

import config
from qa_engine.orchestrator import RAGSystem
import os

print("="*60)
print("创建RAG系统...")
rag = RAGSystem()

print(f"\n图像功能启用: {rag.image_enabled}")
print(f"图像索引数量: {rag.image_store.get_image_count()}")
print(f"Kimi生成器: {rag.kimi_generator is not None}")

# 加载PDF
print("\n加载招股说明书2.pdf...")
result = rag.load_pdf('/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf')
print(f"加载结果: {result}")

print(f"\n加载后图像索引数量: {rag.image_store.get_image_count()}")

# 测试问题
test_questions = [
    "组织结构图中销售部有几个部门构成",
    "销售部下设哪些部门",
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    
    result = rag.ask(question)
    
    print(f"  使用多模态: {result.get('used_multimodal', False)}")
    print(f"  多模态图片: {result.get('multimodal_image', '')[:80]}")
    print(f"  图像结果数: {result.get('image_count', 0)}")
    print(f"  文字chunk数: {result.get('total_chunks_found', 0)}")
    print(f"  耗时: {result.get('elapsed', '?')}")
    
    answer = result.get('rag_answer', '')
    if answer:
        print(f"  回答: {answer[:300]}")
    else:
        print(f"  回答: (空)")
    
    if result.get('error'):
        print(f"  错误: {result['error']}")

print(f"\n{'='*60}")
print("测试完成")
