import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_engine.orchestrator import PDFQASystem

system = PDFQASystem()

# 检查embedding模型
print(f"Embedding model: {system.embedding_model}")
print(f"Vector store: {system.vector_store}")

# 直接测试向量搜索
query = "中国太保2021年的营业收入相比2020年增长了多少"
if system.embedding_model and system.vector_store:
    emb = system.embedding_model.encode([query], is_query=True)[0]
    print(f"\nQuery embedding shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
    
    results = system.vector_store.search(
        query_embedding=emb, query_text=query,
        top_k=5, threshold=0.1
    )
    print(f"\n搜索结果数: {len(results)}")
    for i, r in enumerate(results[:5]):
        chunk = r.get("chunk", r)
        print(f"\n--- Top {i+1} (score: {r.get('score', 0):.4f}) ---")
        text = chunk.get("text", "")[:200]
        print(f"文本: {text}")
        print(f"来源: {chunk.get('source_file', 'N/A')}")
else:
    print("Embedding model or vector store not initialized!")

# 也测试一下全文搜索
if system.fulltext_store:
    print(f"\n\n=== 全文搜索 ===")
    ft_results = system.fulltext_store.search(query, top_k=5)
    print(f"全文搜索结果数: {len(ft_results)}")
    for i, r in enumerate(ft_results[:3]):
        print(f"\n--- FT Top {i+1} (score: {r.get('fulltext_score', 0):.4f}) ---")
        print(f"文本: {r.get('text', '')[:200]}")
