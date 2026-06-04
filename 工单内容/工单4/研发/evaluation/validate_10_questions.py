"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
快速验证 — 14个测试问题检索质量 + 响应时间（含两份招股说明书）
"""

import os
import sys
import time
import json

PROJECT_DIR = os.path.expanduser("~/Desktop/工单3")
sys.path.insert(0, PROJECT_DIR)

import config
from qa_engine.orchestrator import RAGSystem
from evaluation.test_questions import TEST_QUESTIONS

print("=" * 70)
print("  表格解析优化版 v3.0 — 14个测试问题验证")
print("  (工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化)")
print("=" * 70)

# 加载两份PDF
print("\n→ 初始化RAG系统...")
system = RAGSystem(use_cache=True)

pdf1 = os.path.join(config.DATA_DIR, "招股说明书1.pdf")
pdf2 = os.path.join(config.DATA_DIR, "招股说明书2.pdf")

# 加载两份PDF
for pdf_path, label in [(pdf1, "兴图新科"), (pdf2, "力源信息")]:
    result = system.load_pdf(pdf_path)
    print(f"  加载{label}({os.path.basename(pdf_path)}): {result.get('status')}, "
          f"{result.get('pages')}页, {result.get('chunks')}个chunk, "
          f"{result.get('tables_found', 0)}个表格")

print(f"\n  总chunk数: {system.vector_store.total_chunks}")

# 运行测试
all_results = []
retrieval_times = []

print("\n" + "-" * 70)
for i, q in enumerate(TEST_QUESTIONS):
    qid = q["id"]
    question = q["question"]
    source = q.get("source_pdf", "")
    category = q.get("category", "")

    # 检索 + 生成计时
    start = time.time()
    result = system.ask(question)
    elapsed = time.time() - start

    chunks = result.get("source_chunks", [])
    num_chunks = len(chunks)
    query_analysis = result.get("query_analysis", {})
    error = result.get("error")
    answer = result.get("rag_answer", "")
    answer_ok = bool(answer) and (error is None)
    has_info = "未找到" not in answer[:50] if answer else False

    source_files = result.get("source_files", [])
    top_scores = [c.get("final_score", c.get("score", 0)) for c in chunks[:3]]
    top_score = max(top_scores) if top_scores else 0

    all_results.append({
        "id": qid, "question": question,
        "source": source, "category": category,
        "num_chunks": num_chunks, "top_score": top_score,
        "elapsed_s": round(elapsed, 3),
        "has_answer": answer_ok and has_info,
        "source_files": source_files,
        "table_type": query_analysis.get("table_type", ""),
        "error": error,
    })
    retrieval_times.append(elapsed)

    status = "✓" if (answer_ok and has_info) else "✗"
    marker = ""
    if qid in [1, 2, 3, 4]:
        marker = " [新增]"
    print(f"  [{status}] q[{qid}]{marker} {source} {category}")
    print(f"          检索{num_chunks}chunk, 最高分{top_score:.4f}, {elapsed:.2f}s")
    if source_files:
        print(f"          来源: {', '.join(source_files)}")
    if error:
        print(f"          错误: {error}")
    if answer:
        print(f"          答案: {answer[:100]}...")

print("-" * 70)

# 汇总
total = len(all_results)
success = sum(1 for r in all_results if r["has_answer"])
avg_time = sum(retrieval_times) / len(retrieval_times)
avg_top_score = sum(r["top_score"] for r in all_results) / total

# 分文档统计
xingtu_results = [r for r in all_results if "招股说明书1" in r.get("source", "")]
liyuan_results = [r for r in all_results if "招股说明书2" in r.get("source", "")]
xingtu_success = sum(1 for r in xingtu_results if r["has_answer"])
liyuan_success = sum(1 for r in liyuan_results if r["has_answer"])

print(f"\n{'='*70}")
print(f"  汇总结果")
print(f"{'='*70}")
print(f"  总问题数:            {total}")
print(f"  有回答:              {success}/{total} ({success/total*100:.1f}%)")
print(f"  平均响应时间:        {avg_time:.2f}s")
print(f"  平均最高检索分:      {avg_top_score:.4f}")
print(f"\n  分文档:")
print(f"   兴图新科(10问):     {xingtu_success}/{len(xingtu_results)} ({xingtu_success/len(xingtu_results)*100:.1f}%)")
print(f"   力源信息(4问):      {liyuan_success}/{len(liyuan_results)} ({liyuan_success/len(liyuan_results)*100:.1f}%)")

within_3s = sum(1 for t in retrieval_times if t <= 3)
print(f"\n  响应时间≤3秒:        {within_3s}/{total} ({within_3s/total*100:.1f}%)")

# 详细结果
print(f"\n  详细结果:")
print(f"  {'ID':<6} {'结果':<6} {'类型':<10} {'chunk':<6} {'最高分':<10} {'耗时(s)':<10}")
print(f"  {'-'*48}")
for r in sorted(all_results, key=lambda x: x["id"]):
    status = "✓" if r["has_answer"] else "✗"
    cat = r.get("category", "")[:8]
    print(f"  {r['id']:<6} {status:<6} {cat:<10} {r['num_chunks']:<6} {r['top_score']:<10.4f} {r['elapsed_s']:<10.2f}")

# 保存
out = os.path.join(PROJECT_DIR, "evaluation/validation_results.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: {out}")
