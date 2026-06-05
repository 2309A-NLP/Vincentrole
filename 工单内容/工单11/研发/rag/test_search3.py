import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import config
from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.milvus_store import MilvusVectorStore

emb = EmbeddingModel(
    model_name=config.EMBEDDING_CONFIG["模型名称"],
    model_registry=config.EMBEDDING_MODELS,
)

store = MilvusVectorStore(
    dimension=config.EMBEDDING_CONFIG["维度"],
    collection_name=config.MILVUS_CONFIG["collection_name"],
    metadata_path=config.MILVUS_CONFIG["metadata_path"],
    milvus_host=config.MILVUS_CONFIG["host"],
    milvus_port=config.MILVUS_CONFIG["port"],
)

question = "中国太保2021年的营业收入相比2020年增长了多少"
query_emb = emb.encode([question], is_query=True)[0]
results = store.search(
    query_embedding=query_emb,
    top_k=5,
    threshold=0.1,
)
print(f"搜索到 {len(results)} 条结果")
for i, r in enumerate(results):
    print(f"\n=== Result {i+1} ===")
    chunk = r.get("chunk", r)
    print(f"source_file: {chunk.get('source_file', 'N/A')}")
    print(f"page: {chunk.get('page', 'N/A')}")
    print(f"score: {r.get('score', 0):.4f}")
    print(f"text: {chunk.get('text', '')[:300]}")
