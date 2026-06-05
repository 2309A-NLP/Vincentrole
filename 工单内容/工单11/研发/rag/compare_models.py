"""
工单12 vs 工单7 对比测试：微调模型 vs 原始模型
"""
import os
import sys
import json
import time

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.milvus_store import MilvusVectorStore
from config import EMBEDDING_MODELS, MILVUS_CONFIG

# 测试问题
TEST_QUERIES = [
    "什么是IC市场？",
    "发行股数是多少？",
    "军用领域收入占比",
    "注册资本是多少？",
    "公司的核心竞争力是什么？",
    "募集资金用途",
    "组织结构图",
    "主要客户有哪些？",
]

TOP_K = 5


def load_model_and_store(model_key):
    """加载模型并连接Milvus"""
    embed_model = EmbeddingModel(
        model_name=model_key,
        device="cpu",
        normalize=True,
        model_registry=EMBEDDING_MODELS,
    )
    vector_store = MilvusVectorStore(
        dimension=768,
        collection_name=MILVUS_CONFIG["collection_name"],
        metadata_path=MILVUS_CONFIG["metadata_path"],
        milvus_host=MILVUS_CONFIG["host"],
        milvus_port=MILVUS_CONFIG["port"],
    )
    return embed_model, vector_store


def search_and_format(embed_model, vector_store, query, top_k=TOP_K):
    """检索并返回结果"""
    query_emb = embed_model.encode([query], is_query=True)[0]
    raw = vector_store.search(
        query_embedding=query_emb,
        query_text=query,
        top_k=top_k,
        threshold=0.0,
    )
    results = []
    for r in raw:
        chunk = r.get("chunk", {})
        text = chunk.get("text", r.get("text", ""))[:80].replace("\n", " ")
        score = float(r.get("score", 0))
        page = chunk.get("page", r.get("page", "?"))
        results.append({"text": text, "score": round(score, 4), "page": page})
    return results


def main():
    print("=" * 70)
    print("工单7 vs 工单12：原始模型 vs 微调模型 检索效果对比")
    print("=" * 70)

    # 加载原始模型
    print("\n[1/2] 加载原始模型 bge-base-zh-v1.5 ...")
    orig_embed, vector_store = load_model_and_store("bge-base-zh-v1.5")
    print("  原始模型加载完成")

    # 加载微调模型
    print("[2/2] 加载微调模型 finetuned-bge-base-zh-v1.5 ...")
    ft_embed, _ = load_model_and_store("finetuned-bge-base-zh-v1.5")
    print("  微调模型加载完成\n")

    # 对比测试
    total_orig_score = 0
    total_ft_score = 0
    ft_wins = 0
    orig_wins = 0

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"测试 {i}/{len(TEST_QUERIES)}: {query}")
        print(f"{'─' * 70}")

        t0 = time.time()
        orig_results = search_and_format(orig_embed, vector_store, query)
        orig_time = time.time() - t0

        t0 = time.time()
        ft_results = search_and_format(ft_embed, vector_store, query)
        ft_time = time.time() - t0

        orig_top_score = orig_results[0]["score"] if orig_results else 0
        ft_top_score = ft_results[0]["score"] if ft_results else 0
        total_orig_score += orig_top_score
        total_ft_score += ft_top_score

        if ft_top_score > orig_top_score:
            ft_wins += 1
        elif orig_top_score > ft_top_score:
            orig_wins += 1

        print(f"\n  工单7 (原始模型) Top1分数: {orig_top_score:.4f}  耗时: {orig_time:.2f}s")
        for j, r in enumerate(orig_results[:3], 1):
            print(f"    {j}. [P{r['page']}|{r['score']:.4f}] {r['text']}")

        print(f"\n  工单12 (微调模型) Top1分数: {ft_top_score:.4f}  耗时: {ft_time:.2f}s")
        for j, r in enumerate(ft_results[:3], 1):
            print(f"    {j}. [P{r['page']}|{r['score']:.4f}] {r['text']}")

    # 汇总
    print(f"\n{'=' * 70}")
    print("对比汇总")
    print(f"{'=' * 70}")
    print(f"测试问题数: {len(TEST_QUERIES)}")
    print(f"原始模型 平均Top1分数: {total_orig_score / len(TEST_QUERIES):.4f}")
    print(f"微调模型 平均Top1分数: {total_ft_score / len(TEST_QUERIES):.4f}")
    print(f"微调模型胜出: {ft_wins}/{len(TEST_QUERIES)}")
    print(f"原始模型胜出: {orig_wins}/{len(TEST_QUERIES)}")
    print(f"持平: {len(TEST_QUERIES) - ft_wins - orig_wins}/{len(TEST_QUERIES)}")

    improvement = (total_ft_score - total_orig_score) / len(TEST_QUERIES)
    if improvement > 0:
        print(f"\n微调模型平均Top1分数提升: +{improvement:.4f}")
    elif improvement < 0:
        print(f"\n微调模型平均Top1分数下降: {improvement:.4f}")
    else:
        print("\n持平")


if __name__ == "__main__":
    main()
