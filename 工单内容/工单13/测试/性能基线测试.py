"""
工单13 - RAG 性能基线测试
测量工单6 RAG 管道各阶段耗时，定位瓶颈
"""
import sys
import os
import time
import json
import uuid
import logging
from datetime import datetime

# 加入工单6路径
W6_DIR = "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单6/研发"
sys.path.insert(0, W6_DIR)

os.chdir(W6_DIR)

# ===== 结构化日志 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S.%f",
)
log = logging.getLogger("perf")


# ===== 10个测试问题 =====
TEST_QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]


class RAGProfiler:
    """对工单6的 RAG 管道进行逐阶段计时"""

    def __init__(self):
        self.timings = {}   # stage_name -> [durations]
        self.request_id = None

    def _start(self, stage):
        self._t0 = time.perf_counter()
        self._stage = stage

    def _end(self, sub_stage=""):
        t = time.perf_counter() - self._t0
        stage_key = self._stage + (f"/{sub_stage}" if sub_stage else "")
        if stage_key not in self.timings:
            self.timings[stage_key] = []
        self.timings[stage_key].append(t)
        log.info(f"[{self.request_id}] {stage_key}: {t*1000:.0f}ms")
        return t

    def run_query(self, question: str, qid: int = 0) -> dict:
        """完整跑一次 RAG 管道，返回各阶段耗时"""
        self.request_id = f"Q{qid}|{uuid.uuid4().hex[:6]}"
        stages = {"question_id": qid, "question": question[:50], "request_id": self.request_id}

        try:
            import config
            from knowledge_base.embeddings import EmbeddingModel
            from knowledge_base.vector_store import VectorStore
            from knowledge_base.fulltext_store import FullTextStore
            from knowledge_base.reranker import create_reranker
            from qa_engine.generator import LLMGenerator
            from pdf_parser.chunker import TextChunker

            # ---- Stage 1: Embedding 模型加载（仅首次） ----
            self._start("1_embedding_load")
            if not hasattr(self, "_emb"):
                emb_cfg = config.EMBEDDING_CONFIG
                self._emb = EmbeddingModel(
                    model_name=emb_cfg["模型名称"],
                    device=emb_cfg["设备"],
                    normalize=emb_cfg["归一化"],
                    model_registry=getattr(config, "EMBEDDING_MODELS", {}),
                )
                self._ft_enabled = getattr(config, "FULLTEXT_CONFIG", {}).get("启用", False)
                if self._ft_enabled:
                    self._ft_store = FullTextStore(index_dir=config.FULLTEXT_CONFIG["索引目录"])
                # 向量存储
                self._vec_store = VectorStore(
                    dimension=emb_cfg["维度"],
                    metadata_path="data/chunk_metadata.json",
                    index_path="data/vector_index.faiss",
                )
                # 重排器
                rc = getattr(config, "RERANKER_CONFIG", {})
                self._reranker = create_reranker(
                    reranker_type=rc.get("默认重排器", "tfidf"),
                    model_path=rc.get("reranker_model_path", ""),
                    llm_config=getattr(config, "KIMI_CONFIG", {}),
                ) if rc.get("启用重排", False) else None
                # LLM生成器
                llm = config.LLM_CONFIG
                self._gen = LLMGenerator(
                    provider=llm["提供商"],
                    model=llm["模型"],
                    api_base=llm["API地址"],
                    api_key=llm["API密钥"],
                    max_tokens=llm["最大Token数"],
                    temperature=llm["温度"],
                    timeout=llm["超时"],
                    max_retries=llm["重试次数"],
                )
            stages["1_embedding_load"] = self._end()
            if stages["1_embedding_load"] > 1:
                stages["1_embedding_load"] *= 0  # 首次加载不计入

            # ---- Stage 2: Query 编码 ----
            self._start("2_query_encode")
            query_vec = self._emb.encode([question])
            stages["2_query_encode"] = self._end()

            # ---- Stage 3: 向量检索（FAISS） ----
            self._start("3_vector_search")
            vec_results = self._vec_store.search(query_vec[0], query_text=question, top_k=8)
            stages["3_vector_search"] = self._end()

            # ---- Stage 4: 全文检索 ----
            stages["4_fulltext_search"] = 0
            ft_results = []
            if self._ft_enabled:
                self._start("4_fulltext_search")
                ft_results = self._ft_store.search(question, top_k=8)
                stages["4_fulltext_search"] = self._end()

            # ---- Stage 5: 融合（RRF） ----
            self._start("5_fusion")
            all_results = {r["chunk_id"]: r for r in vec_results}
            for r in ft_results:
                if r["chunk_id"] not in all_results:
                    all_results[r["chunk_id"]] = r
            fused = list(all_results.values())[:10]
            stages["5_fusion"] = self._end()

            # ---- Stage 6: 重排 ----
            stages["6_rerank"] = 0
            if self._reranker and fused:
                self._start("6_rerank")
                reranked = self._reranker.rerank(question, fused)
                stages["6_rerank"] = self._end()

            # ---- Stage 7: 上下文组装 ----
            self._start("7_context_assembly")
            context = "\n".join([r.get("text", "")[:200] for r in (reranked if self._reranker else fused)])
            stages["7_context_assembly"] = self._end()

            # ---- Stage 8: LLM 生成 ----
            self._start("8_llm_generate")
            result = self._gen.generate(question, context)
            stages["8_llm_generate"] = self._end()

            # 总计
            total = sum(v for k, v in stages.items() if k.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_")))
            stages["total"] = total
            stages["success"] = True
            stages["answer"] = result.get("answer", "")[:100]

            log.info(f"[{self.request_id}] ✅ TOTAL: {total*1000:.0f}ms")

        except Exception as e:
            stages["total"] = time.perf_counter() - self._t0 if hasattr(self, '_t0') else -1
            stages["success"] = False
            stages["error"] = str(e)
            log.error(f"[{self.request_id}] ❌ FAILED: {e}")

        return stages


def main():
    print("=" * 70)
    print("工单13 — RAG 性能基线测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标系统: 工单6 (混合检索 + 3重排器)")
    print(f"测试问题: {len(TEST_QUESTIONS)} 个")
    print("=" * 70)

    profiler = RAGProfiler()
    all_results = []

    for q in TEST_QUESTIONS:
        print(f"\n▶ [{q['id']}] {q['question'][:60]}...")
        result = profiler.run_query(q["question"], q["id"])
        if result["success"]:
            print(f"   ✅ {result['total']*1000:.0f}ms | {result.get('answer', '')[:60]}...")
        else:
            print(f"   ❌ {result.get('error', '')}")
        all_results.append(result)

    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print("📊 性能汇总")
    print("=" * 70)

    successful = [r for r in all_results if r.get("success")]
    if successful:
        stages_list = ["1_embedding_load", "2_query_encode", "3_vector_search",
                        "4_fulltext_search", "5_fusion", "6_rerank",
                        "7_context_assembly", "8_llm_generate"]
        print(f"\n{'阶段':<20} {'平均':>8} {'最小':>8} {'最大':>8} {'占比':>8}")
        print("-" * 60)
        for s in stages_list:
            times = [r.get(s, 0) for r in successful]
            avg = sum(times) / len(times) * 1000
            mn = min(times) * 1000
            mx = max(times) * 1000
            total_avg = sum(r["total"] for r in successful) / len(successful) * 1000
            pct = sum(times) / sum(r["total"] for r in successful) * 100
            stage_name = s.split("/")[0] if "/" in s else s
            print(f"{stage_name:<20} {avg:>7.0f}ms {mn:>7.0f}ms {mx:>7.0f}ms {pct:>7.1f}%")

        total_avg = sum(r["total"] for r in successful) / len(successful) * 1000
        print("-" * 60)
        print(f"{'总计':<20} {total_avg:>7.0f}ms")

        # 保存结果
        output = {
            "测试时间": datetime.now().isoformat(),
            "测试系统": "工单6 (混合检索+CrossEncoder重排)",
            "LLM": "qwen-plus (通义千问)",
            "Embedding": "bge-base-zh-v1.5",
            "向量后端": "FAISS",
            "测试问题数": len(successful),
            "各阶段耗时(ms)": {
                s: {
                    "平均": sum(r.get(s, 0) for r in successful) / len(successful) * 1000,
                    "占比": sum(r.get(s, 0) for r in successful) / sum(r["total"] for r in successful) * 100,
                }
                for s in stages_list
            },
            "总平均耗时(ms)": total_avg,
            "逐问题详情": [
                {"qid": r["question_id"], "耗时(ms)": r["total"] * 1000,
                 "各阶段(ms)": {k: v * 1000 for k, v in r.items()
                               if k.startswith(("1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_"))}}
                for r in successful
            ],
        }
        result_path = os.path.join(os.path.dirname(__file__), "基线测试结果.json")
        with open(result_path, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细结果已保存: {result_path}")

    else:
        print("❌ 所有测试均失败")


if __name__ == "__main__":
    main()
