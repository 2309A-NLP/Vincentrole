"""
检查问题1的详细检索结果
"""
import sys
sys.path.insert(0, '/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4')

import config
from qa_engine.orchestrator import RAGSystem

rag = RAGSystem()
result = rag.load_pdf('/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf')
print(f"加载完成，图像数: {result.get('images_found', 0)}")

question = "组织结构图中销售部有几个部门构成"
print(f"\n问题: {question}")

# 检索图像
image_results = rag.image_store.search(query_text=question, top_k=10)
print(f"\n图像检索结果 (top 10):")
for i, ir in enumerate(image_results):
    score = ir.get("score", 0)
    path = ir.get("image_path", "")
    img_type = ir.get("image_type", "")
    page = ir.get("page_num", 0)
    desc = ir.get("description", "")[:60]
    print(f"  [{i+1}] 分数={score:.3f}, 页={page}, 类型={img_type}")
    print(f"      路径: {path.split('/')[-1]}")
    print(f"      描述: {desc}...")

# 检索文字
print(f"\n文字检索结果:")
retrieval = rag.retriever.retrieve(question, top_k=8, threshold=0.0)
for i, r in enumerate(retrieval['results'][:5]):
    score = r.get("score", 0)
    page = r.get("page", 0)
    text = r.get("text", "")[:100]
    print(f"  [{i+1}] 分数={score:.3f}, 页={page}")
    print(f"      文本: {text}...")

# 检查是否有p40的内容
print(f"\n检查p40的chunk是否存在:")
has_p40 = False
for r in retrieval['results']:
    if r.get("page") == 40:
        has_p40 = True
        print(f"  找到p40: {r.get('text', '')[:100]}...")
if not has_p40:
    print("  未找到p40的内容")
