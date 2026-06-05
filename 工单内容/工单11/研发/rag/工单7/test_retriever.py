import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import config
from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.milvus_store import MilvusVectorStore
from qa_engine.retriever import Retriever

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

retriever = Retriever(
    vector_store=store,
    embedding_model=emb,
    fulltext_store=None,
)

question = "中国太保2021年的营业收入相比2020年增长了多少"
result = retriever.retrieve(question, top_k=5, threshold=0.0, mode="vector")
print(f"Results: {len(result.get('results', []))}")
print(f"Context length: {len(result.get('context_text', ''))}")
print(f"\nContext:\n{result.get('context_text', '')[:2000]}")
