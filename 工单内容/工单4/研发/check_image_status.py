#!/usr/bin/env python3
"""检查图像存储状态"""
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

from qa_engine.orchestrator import RAGSystem

# 创建orchestrator实例
rag = RAGSystem()

# 检查图像存储状态
print(f"图像功能启用: {rag.image_enabled}")
print(f"图像索引数量: {rag.image_store.get_image_count()}")

# 尝试搜索
if rag.image_store.get_image_count() > 0:
    results = rag.image_store.search("组织结构图 销售部", top_k=5)
    print(f"\n搜索结果数量: {len(results)}")
    for i, r in enumerate(results[:3]):
        print(f"  结果{i+1}: 相似度={r.get('score', 0):.3f}, 路径={r.get('image_path', 'N/A')}")
else:
    print("\n图像存储为空，需要先加载PDF文档")

# 检查是否有已保存的图像索引
index_dir = '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/image_index'
if os.path.exists(index_dir):
    files = os.listdir(index_dir)
    print(f"\n图像索引目录存在，包含文件: {files}")
else:
    print(f"\n图像索引目录不存在: {index_dir}")
