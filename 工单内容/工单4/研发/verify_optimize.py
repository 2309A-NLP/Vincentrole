"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
验证脚本 - 对比优化前后检索效果
"""

import sys
import time
import json
sys.path.insert(0, "/Users/suwente/Desktop/工单2")

from qa_engine.orchestrator import RAGSystem

TEST_QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95,  "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33,  "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34,  "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

def main():
    pdf_path = "/Users/suwente/Desktop/工单2/data/招股说明书1.pdf"

    print("═" * 60)
    print("  RAG 优化版检索效果验证")
    print("═" * 60)

    system = RAGSystem(use_cache=False)
    print("\n→ 加载PDF文档...")
    result = system.load_pdf(pdf_path)
    print(f"   加载结果: {result}")

    print("\n" + "─" * 60)
    print("→ 跑测试问题（仅检索，不调用LLM生成）")
    print("─" * 60)

    for q in TEST_QUESTIONS:
        print(f"\n[{q['id']}] {q['question']}")

        # 检索
        ret = system.retriever.retrieve(q["question"], top_k=5)
        print(f"   检索意图: {ret['query_analysis']['intent']}")
        print(f"   实体: {ret['query_analysis']['entities']}")
        print(f"   找到 {ret['total_results']} 个结果:")

        for i, r in enumerate(ret["results"][:3]):
            print(f"      [{i+1}] 页{r['page']} | score={r['score']:.4f} | final={r.get('final_score', 0):.4f} | {r['text'][:60]}...")

        print(f"   上下文长度: {len(ret['context_text'])} 字符")

    print("\n═" * 60)
    print("  验证完成")
    print("═" * 60)

if __name__ == "__main__":
    main()
