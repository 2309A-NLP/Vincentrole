"""
诊断：检查问答流程中每一步的状态
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
print(f"加载结果: status={result.get('status')}, images={result.get('images_found', 0)}")

print(f"\n加载后图像索引数量: {rag.image_store.get_image_count()}")

# 测试图像搜索
question = "组织结构图中销售部有几个部门构成"
print(f"\n测试图像搜索: '{question}'")
image_results = rag.image_store.search(query_text=question, top_k=5)
print(f"图像搜索结果数: {len(image_results)}")
for i, ir in enumerate(image_results):
    score = ir.get("score", 0)
    path = ir.get("image_path", "")
    img_type = ir.get("image_type", "")
    exists = os.path.exists(path) if path else False
    print(f"  [{i+1}] 相似度={score:.3f}, 类型={img_type}, 路径={path}")
    print(f"      文件存在: {exists}")

# 测试文字检索
print(f"\n测试文字检索: '{question}'")
retrieval_result = rag.retriever.retrieve(question, top_k=5, threshold=0.3)
print(f"文字检索结果数: {retrieval_result['total_results']}")
for i, r in enumerate(retrieval_result['results'][:3]):
    print(f"  [{i+1}] 页={r.get('page')}, 分数={r.get('score', 0):.3f}")
    print(f"      文本: {r.get('text', '')[:100]}...")

# 判断是否走多模态
best_image_path = ""
best_image_score = 0
for ir in image_results:
    s = ir.get("score", 0)
    if s > best_image_score:
        best_image_score = s
        best_image_path = ir.get("image_path", "")

use_multimodal = (
    rag.kimi_generator is not None
    and best_image_path
    and os.path.exists(best_image_path)
    and best_image_score >= 0.3
)

print(f"\n多模态判断:")
print(f"  kimi_generator存在: {rag.kimi_generator is not None}")
print(f"  best_image_path: {best_image_path}")
print(f"  图片文件存在: {os.path.exists(best_image_path) if best_image_path else False}")
print(f"  best_image_score: {best_image_score:.3f} >= 0.3: {best_image_score >= 0.3}")
print(f"  最终使用多模态: {use_multimodal}")

if use_multimodal:
    print("\n[测试] 调用Kimi多模态...")
    import base64
    with open(best_image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    context = retrieval_result.get("context_text", "")
    gen_result = rag.kimi_generator.generate_with_image(
        question=question,
        context=context[:2000],
        image_base64=img_base64,
    )
    print(f"Kimi回答: {gen_result.get('answer', '无答案')[:300]}")
    print(f"错误: {gen_result.get('error', '无')}")
else:
    print("\n[跳过] 不满足多模态条件，将使用文字模式")
