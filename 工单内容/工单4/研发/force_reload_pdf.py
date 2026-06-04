#!/usr/bin/env python3
"""强制重新加载PDF文档并提取图像"""
import sys
import os
import shutil

# 添加项目路径
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

import config
from qa_engine.orchestrator import RAGSystem

# 清除缓存
print("清除缓存...")
cache_files = [
    config.VECTOR_STORE_CONFIG["索引路径"],
    config.VECTOR_STORE_CONFIG["元数据路径"],
]

for cache_file in cache_files:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"  删除缓存: {cache_file}")

# 清除图像索引
image_index_dir = config.IMAGE_CONFIG["输出目录"]
if os.path.exists(image_index_dir):
    shutil.rmtree(image_index_dir)
    print(f"  删除图像索引目录: {image_index_dir}")

# 创建RAGSystem实例
print("\n创建RAG系统...")
rag = RAGSystem()

# 强制重新加载PDF文档
pdf_files = [
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf',
]

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"\n加载: {os.path.basename(pdf_path)}")
        result = rag.load_pdf(pdf_path)
        print(f"  加载结果: {result}")
        
        # 检查图像是否被提取
        if 'images_found' in result:
            print(f"  提取图像数: {result['images_found']}")
            print(f"  图像类型: {result.get('image_types', {})}")
    else:
        print(f"\n文件不存在: {pdf_path}")

# 检查图像索引状态
print(f"\n图像索引总数: {rag.image_store.get_image_count()}")

# 检查图像文件是否保存
if os.path.exists(image_index_dir):
    files = os.listdir(image_index_dir)
    print(f"\n图像目录 {image_index_dir}:")
    print(f"  文件数: {len(files)}")
    if files:
        print(f"  前5个: {files[:5]}")
else:
    print(f"\n图像目录不存在: {image_index_dir}")