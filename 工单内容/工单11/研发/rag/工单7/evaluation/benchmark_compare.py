"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
优化前后对比分析 — 检索精确度对比 benchmark

用法:
    cd ~/Desktop/工单2 && python evaluation/benchmark_compare.py

说明:
    1. 在「工单1」项目上跑10个测试问题，记录检索结果
    2. 在「工单2」项目上跑同样的10个问题，记录检索结果
    3. 输出对比表格和精确度变化
"""

import os
import sys
import json
import time

# 工单路径
WO1_DIR = os.path.expanduser("~/Desktop/工单1")
WO2_DIR = os.path.expanduser("~/Desktop/工单2")

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


def run_benchmark_for_project(project_dir, label):
    """在指定项目上运行Benchmark"""
    sys.path.insert(0, project_dir)
    
    import config as cfg
    from qa_engine.orchestrator import RAGSystem
    
    pdf_path = os.path.join(project_dir, "data/招股说明书1.pdf")
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # 初始化系统（使用缓存加速比较）
    system = RAGSystem(use_cache=True)
    load_result = system.load_pdf(pdf_path)
    print(f"  加载结果: {load_result.get('status', 'unknown')} ({load_result.get('elapsed', '?')})")
    
    results = []
    for q in TEST_QUESTIONS:
        question = q["question"]
        qid = q["id"]
        
        start = time.time()
        retrieval = system.retriever.retrieve(question, top_k=5, threshold=0.0)
        elapsed = time.time() - start
        
        # 提取关键指标
        top_results = retrieval.get("results", [])
        top_scores = [r.get("final_score", r.get("score", 0)) for r in top_results[:3]]
        top_pages = [r.get("page", 0) for r in top_results[:3]]
        
        results.append({
            "id": qid,
            "question": question,
            "num_chunks": len(top_results),
            "top_score": max(top_scores) if top_scores else 0,
            "avg_top3_score": sum(top_scores[:3]) / len(top_scores[:3]) if top_scores[:3] else 0,
            "pages": top_pages,
            "elapsed_s": round(elapsed, 3),
            "intent": retrieval.get("query_analysis", {}).get("intent", "?"),
        })
        
        print(f"  [问题 {qid}] {question[:30]}...")
        print(f"    找到 {results[-1]['num_chunks']} 个chunk, 最高分: {results[-1]['top_score']:.4f}, 耗时: {elapsed:.3f}s")
    
    sys.path.pop(0)
    return results


def compute_accuracy(results):
    """
    估算检索准确率：
    从10个问题中统计"有足够检索结果"的比例。
    阈值：至少检索到2个chunk，且最高分 > 0.01
    """
    successful = 0
    for r in results:
        if r["num_chunks"] >= 2 and r["top_score"] > 0.01:
            successful += 1
    return successful / len(results) * 100


def main():
    print("=" * 70)
    print("  RAG系统优化前后对比分析")
    print("  工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化")
    print("=" * 70)
    
    # 检查项目目录
    for d, name in [(WO1_DIR, "工单1"), (WO2_DIR, "工单2")]:
        if not os.path.exists(d):
            print(f"  [警告] {name} 目录不存在: {d}")
            return
    
    # 运行工单1（原版）
    results_wo1 = run_benchmark_for_project(WO1_DIR, "优化前 - 工单1（原版）")
    
    # 运行工单2（优化版）
    results_wo2 = run_benchmark_for_project(WO2_DIR, "优化后 - 工单2（优化版）")
    
    # 输出对比表格
    print("\n" + "=" * 80)
    print("  优化前后检索精确度对比")
    print("=" * 80)
    print(f"{'问题ID':<8} {'优化前chunk数':<14} {'优化后chunk数':<14} {'优化前最高分':<14} {'优化后最高分':<14}")
    print("-" * 80)
    
    total_wo1_chunks = 0
    total_wo2_chunks = 0
    total_wo1_score = 0
    total_wo2_score = 0
    
    for r1, r2 in zip(results_wo1, results_wo2):
        print(f"{r1['id']:<8} {r1['num_chunks']:<14} {r2['num_chunks']:<14} {r1['top_score']:<14.4f} {r2['top_score']:<14.4f}")
        total_wo1_chunks += r1['num_chunks']
        total_wo2_chunks += r2['num_chunks']
        total_wo1_score += r1['top_score']
        total_wo2_score += r2['top_score']
    
    avg_wo1_score = total_wo1_score / len(results_wo1)
    avg_wo2_score = total_wo2_score / len(results_wo2)
    
    print("-" * 80)
    print(f"{'平均':<8} {total_wo1_chunks/len(results_wo1):<14.1f} {total_wo2_chunks/len(results_wo2):<14.1f} {avg_wo1_score:<14.4f} {avg_wo2_score:<14.4f}")
    
    # 响应时间对比
    avg_wo1_time = sum(r['elapsed_s'] for r in results_wo1) / len(results_wo1)
    avg_wo2_time = sum(r['elapsed_s'] for r in results_wo2) / len(results_wo2)
    
    print(f"\n响应时间对比（检索阶段）:")
    print(f"  工单1（原版）: 平均 {avg_wo1_time:.3f}s/次")
    print(f"  工单2（优化版）: 平均 {avg_wo2_time:.3f}s/次")
    if avg_wo2_time > avg_wo1_time:
        print(f"  注: 优化版增加了BM25+RRF+重排序，以速度换精度")
    
    # 准确率估算
    acc_wo1 = compute_accuracy(results_wo1)
    acc_wo2 = compute_accuracy(results_wo2)
    
    print(f"\n检索准确率估算（检索到≥2个相关chunk的比例）:")
    print(f"  工单1（原版）: {acc_wo1:.1f}%")
    print(f"  工单2（优化版）: {acc_wo2:.1f}%")
    print(f"  提升: +{acc_wo2 - acc_wo1:.1f}%")
    
    # 保存结果
    output = {
        "wo1": results_wo1,
        "wo2": results_wo2,
        "summary": {
            "avg_chunks_wo1": round(total_wo1_chunks / len(results_wo1), 1),
            "avg_chunks_wo2": round(total_wo2_chunks / len(results_wo2), 1),
            "avg_score_wo1": round(avg_wo1_score, 4),
            "avg_score_wo2": round(avg_wo2_score, 4),
            "avg_time_wo1": round(avg_wo1_time, 3),
            "avg_time_wo2": round(avg_wo2_time, 3),
            "accuracy_wo1": round(acc_wo1, 1),
            "accuracy_wo2": round(acc_wo2, 1),
        }
    }
    
    output_path = os.path.join(WO2_DIR, "evaluation/benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
