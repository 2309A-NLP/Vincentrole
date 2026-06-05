#!/usr/bin/env python3
"""简单测试图像问答功能"""
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

# 导入必要的模块
from qa_engine.orchestrator import RAGSystem

# 创建RAGSystem实例
print("创建RAG系统...")
rag = RAGSystem()

# 检查图像索引状态
print(f"图像功能启用: {rag.image_enabled}")
print(f"图像索引数量: {rag.image_store.get_image_count()}")

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
    print("图像索引为空")

# 测试问答功能
print("\n测试问答功能...")
test_questions = [
    "组织结构图中销售部有几个部门构成",
    "销售部下设哪些部门",
]

for question in test_questions:
    print(f"\n问题: {question}")
    try:
        result = rag.ask(question)
        
        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            answer = result.get('answer', '无答案')
            print(f"  答案: {answer[:200]}...")
            print(f"  使用多模态: {result.get('used_multimodal', False)}")
            if result.get('used_multimodal'):
                print(f"  多模态图像: {result.get('multimodal_image', 'N/A')}")
            print(f"  图像结果数: {result.get('image_count', 0)}")
            print(f"  耗时: {result.get('elapsed', 'N/A')}")
    except Exception as e:
        print(f"  异常: {e}")
        import traceback
        traceback.print_exc()