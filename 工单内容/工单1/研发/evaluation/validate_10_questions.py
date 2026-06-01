"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
快速验证 — 10个测试问题检索质量 + 响应时间
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.expanduser("~/Desktop/工单1"))

import config
from qa_engine.orchestrator import RAGSystem

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

pdf_path = os.path.expanduser("~/Desktop/工单1/data/招股说明书1.pdf")

print("=" * 70)
print("  RAG优化版 — 10个测试问题 + 响应时间验证")
print("  (工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化)")
print("=" * 70)

# 初始化
print("\n→ 初始化RAG系统...")
system = RAGSystem(use_cache=True)
load_result = system.load_pdf(pdf_path)
print(f"  加载: {load_result.get('status')}, {load_result.get('pages')}页, {load_result.get('chunks')}个chunk")

# 运行测试
all_results = []
retrieval_times = []
gen_times = []
scores_by_id = {}

print("\n" + "-" * 70)
for i, q in enumerate(TEST_QUESTIONS):
    qid = q["id"]
    question = q["question"]

    # 检索 + 生成计时
    start = time.time()
    result = system.ask(question)
    elapsed = time.time() - start

    # 提取指标
    chunks = result.get("source_chunks", [])
    num_chunks = len(chunks)
    query_analysis = result.get("query_analysis", {})
    error = result.get("error")
    answer = result.get("rag_answer", "")
    answer_ok = bool(answer) and (error is None)
    # 排除纯LLM的"文档中未找到"错误
    has_info = "未找到" not in answer[:50] if answer else False

    top_scores = [c.get("final_score", c.get("score", 0)) for c in chunks[:3]]
    top_score = max(top_scores) if top_scores else 0

    all_results.append({
        "id": qid, "question": question,
        "num_chunks": num_chunks, "top_score": top_score,
        "elapsed_s": round(elapsed, 3),
        "has_answer": answer_ok and has_info,
        "intent": query_analysis.get("intent", "?"),
        "error": error,
    })
    retrieval_times.append(elapsed)

    status = "✓" if (answer_ok and has_info) else "✗"
    print(f"  [{status}] q[{qid}] 检索{num_chunks}chunk, 最高分{top_score:.4f}, {elapsed:.2f}s")
    if error:
        print(f"        错误: {error}")
    if answer:
        print(f"        答案: {answer[:80]}...")

print("-" * 70)

# 汇总
total = len(all_results)
success = sum(1 for r in all_results if r["has_answer"])
avg_time = sum(retrieval_times) / len(retrieval_times)
avg_top3_score = sum(r["top_score"] for r in all_results) / total

print(f"\n{'='*70}")
print(f"  汇总结果")
print(f"{'='*70}")
print(f"  总问题数:            {total}")
print(f"  有回答:              {success}/{total} ({success/total*100:.1f}%)")
print(f"  平均响应时间:        {avg_time:.2f}s")
print(f"  平均最高检索分:      {avg_top3_score:.4f}")

# 检查≤3秒要求
within_3s = sum(1 for t in retrieval_times if t <= 3)
print(f"  响应时间≤3秒:        {within_3s}/{total} ({within_3s/total*100:.1f}%)")

# 按问题ID排序输出
print(f"\n  详细结果（按ID排序）:")
print(f"  {'ID':<6} {'结果':<6} {'chunk数':<8} {'最高分':<10} {'耗时(s)':<10}")
print(f"  {'-'*40}")
for r in sorted(all_results, key=lambda x: x["id"]):
    status = "✓" if r["has_answer"] else "✗"
    print(f"  {r['id']:<6} {status:<6} {r['num_chunks']:<8} {r['top_score']:<10.4f} {r['elapsed_s']:<10.2f}")

# 保存结果
out = os.path.expanduser("~/Desktop/工单1/evaluation/validation_results.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: {out}")
