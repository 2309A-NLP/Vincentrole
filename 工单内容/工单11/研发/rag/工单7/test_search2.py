import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import config
from knowledge_base.embeddings import EmbeddingModel

emb_model = EmbeddingModel(
    model_name=config.EMBEDDING_CONFIG["模型名称"],
    model_registry=config.EMBEDDING_MODELS,
)
print(f"Model: {config.EMBEDDING_CONFIG['模型名称']}, dim: {emb_model.dimension}")

from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
stats = client.get_collection_stats("rag_chunks")
print(f"Milvus chunks: {stats['row_count']}")

query = "中国太保2021年的营业收入相比2020年增长了多少"
query_emb = emb_model.encode([query], is_query=True)[0]
print(f"Query emb len: {len(query_emb)}")

results = client.search(
    collection_name="rag_chunks",
    data=[query_emb],
    limit=5,
    output_fields=["text", "source", "page"]
)
print(f"\n结果数: {len(results[0])}")
for i, hit in enumerate(results[0]):
    print(f"\n--- Top {i+1} (score: {hit['distance']:.4f}) ---")
    print(f"文本: {hit['entity'].get('text', '')[:200]}")
    print(f"来源: {hit['entity'].get('source', 'N/A')}")
