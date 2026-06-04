#!/usr/bin/env python3
"""测试图像问答功能"""
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

from qa_engine.orchestrator import RAGSystem

# 创建RAGSystem实例
print("创建RAG系统...")
rag = RAGSystem()

# 加载PDF文档（这次会提取图像）
print("\n加载PDF文档...")
pdf_files = [
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书1.pdf',
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf',
]

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"\n加载: {os.path.basename(pdf_path)}")
        result = rag.load_pdf(pdf_path)
        print(f"  状态: {result.get('status')}, 图像: {result.get('images_found', 0)}")
    else:
        print(f"\n文件不存在: {pdf_path}")

print(f"\n图像索引总数: {rag.image_store.get_image_count()}")

# 测试图像问答
print("\n测试图像问答...")
test_questions = [
    "组织结构图中销售部有几个部门构成",
    "销售部下设哪些部门",
    "组织架构图中销售部门的组成",
]

for question in test_questions:
    print(f"\n问题: {question}")
    result = rag.ask(question)
    
    if "error" in result:
        print(f"  错误: {result['error']}")
    else:
        print(f"  答案: {result.get('answer', '无答案')[:200]}...")
        print(f"  使用多模态: {result.get('used_multimodal', False)}")
        if result.get('used_multimodal'):
            print(f"  多模态图像: {result.get('multimodal_image', 'N/A')}")
        print(f"  图像结果数: {result.get('image_count', 0)}")
        print(f"  耗时: {result.get('elapsed', 'N/A')}")