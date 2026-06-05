#!/usr/bin/env python3
"""加载PDF文档并提取图像"""
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

from qa_engine.orchestrator import RAGSystem

# 创建RAGSystem实例
print("创建RAG系统...")
rag = RAGSystem()

# 加载PDF文档
pdf_files = [
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书1.pdf',
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf',
]

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"\n加载: {os.path.basename(pdf_path)}")
        result = rag.load_pdf(pdf_path)
        print(f"  加载结果: {result}")
    else:
        print(f"\n文件不存在: {pdf_path}")

# 检查图像索引状态
print(f"\n图像索引数量: {rag.image_store.get_image_count()}")

# 测试图像搜索
if rag.image_store.get_image_count() > 0:
    print("\n测试图像搜索...")
    test_queries = [
        "组织结构图",
        "销售部",
        "部门构成",
    ]
    
    for query in test_queries:
        results = rag.image_store.search(query, top_k=3)
        print(f"\n查询 '{query}':")
        if results:
            for i, r in enumerate(results[:2]):
                score = r.get('score', 0)
                img_path = r.get('image_path', 'N/A')
                print(f"  结果{i+1}: 相似度={score:.3f}, 路径={os.path.basename(img_path)}")
        else:
            print("  无结果")
else:
    print("图像索引为空，无法测试搜索")