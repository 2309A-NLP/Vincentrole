"""
工单13 - 性能优化对比测试
测试不同重排器 + LLM + 参数的组合，找出 < 3秒的最优方案
"""
import sys, os, time, json, logging
from datetime import datetime

W6_DIR = "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单6/研发"
sys.path.insert(0, W6_DIR)
os.chdir(W6_DIR)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

TEST_QUESTIONS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
]


def test_config(label: str, reranker_type: str, top_k: int,
                 model: str, max_tokens: int, context_chunks: int) -> dict:
    """测试一种配置组合"""
    try:
        from knowledge_base.embeddings import EmbeddingModel
        from knowledge_base.vector_store import VectorStore
        from knowledge_base.fulltext_store import FullTextStore
        from knowledge_base.reranker import create_reranker
        from qa_engine.generator import LLMGenerator

        # 加载组件（仅首次）
        emb = EmbeddingModel(model_name="bge-base-zh-v1.5", device="cpu",
                             normalize=True,
                             model_registry=__import__('config').EMBEDDING_MODELS)
        vec = VectorStore(dimension=768, metadata_path="data/chunk_metadata.json",
                          index_path="data/vector_index.faiss")
        ft = FullTextStore(index_dir="data/fulltext_index")

        gen = LLMGenerator(provider="openai", model=model,
                           api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                           api_key="sk-your-api-key-here",
                           max_tokens=max_tokens, temperature=0.1, timeout=15)

        reranker = None
        if reranker_type:
            reranker = create_reranker(
                reranker_type=reranker_type,
                model_path="/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3",
                llm_config={"模型": "qwen-plus",
                            "API地址": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                            "API密钥": "sk-your-api-key-here"},
            )

        times = []
        for q in TEST_QUESTIONS[:3]:  # 每种配置测3个问题
            t0 = time.perf_counter()

            # 编码
            qv = emb.encode([q])
            t1 = time.perf_counter()

            # 向量检索
            vr = vec.search(qv[0], query_text=q, top_k=top_k)
            t2 = time.perf_counter()

            # CrossEncoder自带BM25，所以跳过fulltext
            # 用向量检索结果直接
            candidates = vr

            # 重排
            if reranker:
                candidates = reranker.rerank(q, candidates)
            t3 = time.perf_counter()

            # 上下文
            context = "\n".join([r.get("text", "")[:300] for r in candidates[:context_chunks]])
            t4 = time.perf_counter()

            # LLM
            result = gen.generate(q, context)
            t5 = time.perf_counter()

            # 只统计稳态（忽略首次embedding加载）
            encode_t = t1 - t0
            search_t = t2 - t1
            rerank_t = t3 - t2
            context_t = t4 - t3
            llm_t = t5 - t4
            total = t5 - t0

            times.append({
                "encode": encode_t, "search": search_t, "rerank": rerank_t,
                "context": context_t, "llm": llm_t, "total": total,
                "answer": result.get("answer", "")[:60],
            })

        # 汇总
        avg = {k: sum(r[k] for r in times) / len(times) for k in ["encode", "search", "rerank", "context", "llm", "total"]}
        return {"label": label, "avg": avg, "details": times}

    except Exception as e:
        return {"label": label, "error": str(e)}


def main():
    print("=" * 80)
    print("工单13 — RAG 性能优化方案对比")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 测试 6 种配置
    configs = [
        # (label, reranker_type, top_k, model, max_tokens, context_chunks)
        ("① 基线(CrossEncoder+qwen-plus)", "crossencoder", 8, "qwen-plus", 1024, 5),
        ("② TF-IDF重排+qwen-plus", "tfidf", 8, "qwen-plus", 1024, 5),
        ("③ CrossEncoder+top5+qwen-plus", "crossencoder", 5, "qwen-plus", 1024, 3),
        ("④ TF-IDF+top5+qwen-turbo", "tfidf", 5, "qwen-turbo", 512, 3),
        ("⑤ 无重排+qwen-turbo", "", 5, "qwen-turbo", 512, 3),
        ("⑥ 无重排+top3+qwen-turbo", "", 3, "qwen-turbo", 256, 2),
    ]

    results = []
    for label, rr, tk, mdl, mt, cc in configs:
        print(f"\n▶ 测试: {label}", end=" ", flush=True)
        r = test_config(label, rr, tk, mdl, mt, cc)
        results.append(r)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            a = r["avg"]
            print(f"✅ 总计:{a['total']*1000:.0f}ms "
                  f"(编码:{a['encode']*1000:.0f} 检索:{a['search']*1000:.0f} "
                  f"重排:{a['rerank']*1000:.0f} LLM:{a['llm']*1000:.0f})")

    # 汇总
    print("\n" + "=" * 80)
    print("📊 优化方案对比")
    print("=" * 80)
    print(f"{'方案':<30} {'总计':>7} {'编码':>6} {'检索':>6} {'重排':>6} {'LLM':>7} {'达标':>5}")
    print("-" * 75)
    best = None
    for r in results:
        if "error" in r:
            print(f"{r['label']:<30} ❌ 失败")
            continue
        a = r["avg"]
        total_ms = a['total'] * 1000
        ok = "✅" if total_ms < 3000 else "❌"
        print(f"{r['label']:<30} {total_ms:>6.0f}ms "
              f"{a['encode']*1000:>5.0f} {a['search']*1000:>5.0f} "
              f"{a['rerank']*1000:>5.0f} {a['llm']*1000:>6.0f} {ok:>4}")
        if ok == "✅" and (best is None or total_ms < best['total_ms']):
            best = r
            best['total_ms'] = total_ms

    print("-" * 75)
    print(f"\n🏆 最优方案: {best['label'] if best else '无'}")
    if best:
        a = best["avg"]
        print(f"   总耗时: {a['total']*1000:.0f}ms (< 3秒 ✅)")
        print(f"   重排器: {best['label']}")

    # 保存结果
    output = {
        "测试时间": datetime.now().isoformat(),
        "基线": results[0].get("avg", {}).get("total", 0) * 1000 if "error" not in results[0] else 0,
        "最优": best['total_ms'] if best else 0,
        "提升": f"{(1 - best['total_ms'] / (results[0].get('avg', {}).get('total', 1) * 1000)) * 100:.0f}%" if best and "error" not in results[0] else "N/A",
        "各方案": [
            {
                "方案": r["label"],
                "总耗时(ms)": r["avg"]["total"] * 1000,
                "各阶段(ms)": {k: v * 1000 for k, v in r["avg"].items()},
            }
            for r in results if "error" not in r
        ],
    }
    # Also save as SUCCESS for comparison
    result_path = os.path.join(os.path.dirname(__file__), "优化对比结果.json")
    with open(result_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n📄 结果已保存: {result_path}")


if __name__ == "__main__":
    main()
