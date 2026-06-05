#!/usr/bin/env python3
"""加载PDF文档并提取图像（确保保存图像文件）"""
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

import config
from qa_engine.orchestrator import RAGSystem

# 创建RAGSystem实例
print("创建RAG系统...")
rag = RAGSystem()

# 检查图像配置
print(f"图像功能启用: {rag.image_enabled}")
print(f"图像提取配置: {config.IMAGE_CONFIG}")

# 加载PDF文档
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
image_dirs = [
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/extracted_images',
    '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/extracted_images',
]

for dir_path in image_dirs:
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        print(f"\n图像目录 {dir_path}:")
        print(f"  文件数: {len(files)}")
        if files:
            print(f"  前5个: {files[:5]}")
    else:
        print(f"\n图像目录不存在: {dir_path}")