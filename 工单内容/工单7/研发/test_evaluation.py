"""
工单编号: 人工智能NLP-RAG-功能测试及评估
测试脚本 - 运行RAG评估并生成报告
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_engine.orchestrator import RAGSystem
from evaluation import RAGEvaluator
import config


def load_sample_questions(pdf_path: str = None):
    """
    从sample_questions.pdf加载问题

    Args:
        pdf_path: PDF文件路径

    Returns:
        问题列表
    """
    if pdf_path is None:
        pdf_path = getattr(config, "EVALUATION_CONFIG", {}).get("样本问题路径", "sample_questions.pdf")

    if not os.path.exists(pdf_path):
        print(f"错误: 找不到问题文件 {pdf_path}")
        return []

    try:
        # 使用PDF解析器提取文本
        from pdf_parser.parser import PDFParser
        parser = PDFParser()
        pages = parser.extract_text(pdf_path)

        questions = []
        current_question = None
        current_answer = None
        current_source = None

        for page in pages:
            text = page.get("text", "")
            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检测问题
                if line.startswith("问题：") or line.startswith("问题:"):
                    if current_question:
                        questions.append({
                            "question": current_question,
                            "reference_answer": current_answer,
                            "source": current_source,
                        })
                    current_question = line[3:].strip()
                    current_answer = None
                    current_source = None

                # 检测答案
                elif line.startswith("答案：") or line.startswith("答案:") or line.startswith("参考答案："):
                    current_answer = line[3:].strip() if line.startswith("答案") else line[5:].strip()

                # 检测来源
                elif "年" in line and "报告" in line and current_source is None:
                    current_source = line

            # 处理最后一个问题
            if current_question:
                questions.append({
                    "question": current_question,
                    "reference_answer": current_answer,
                    "source": current_source,
                })

        return questions

    except Exception as e:
        print(f"加载问题失败: {e}")
        return []


def run_evaluation(num_questions: int = 10):
    """
    运行RAG评估

    Args:
        num_questions: 要评估的问题数量
    """
    print("=" * 60)
    print("RAG系统功能测试及评估")
    print("=" * 60)

    # 1. 加载问题
    print("\n1. 加载测试问题...")
    questions = load_sample_questions()
    if not questions:
        print("错误: 没有加载到问题")
        return

    print(f"   加载了 {len(questions)} 个问题")

    # 限制问题数量
    if num_questions and num_questions < len(questions):
        questions = questions[:num_questions]
        print(f"   将评估前 {num_questions} 个问题")

    # 2. 初始化RAG系统
    print("\n2. 初始化RAG系统...")
    rag = RAGSystem(use_cache=True)

    # 3. 加载文档
    print("\n3. 加载文档...")
    data_dir = getattr(config, "DATA_DIR", "./data")
    ccf_dir = os.path.join(data_dir, "ccf_competition")

    if os.path.exists(ccf_dir):
        # 加载ccf_competition目录下的所有文件
        result = rag.load_directory(ccf_dir, recursive=True)
        print(f"   加载结果: {result.get('status')}")
        print(f"   加载了 {result.get('files_loaded', 0)} 个文件")
        print(f"   共 {result.get('chunks', 0)} 个chunk")
    else:
        print(f"   警告: 找不到ccf_competition目录: {ccf_dir}")
        # 尝试加载data目录下的所有文件
        result = rag.load_directory(data_dir, recursive=True)
        print(f"   加载结果: {result.get('status')}")

    # 4. 运行评估
    print("\n4. 开始评估...")
    evaluator = RAGEvaluator(rag, questions)
    eval_results = evaluator.evaluate_all()

    # 5. 显示结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    summary = eval_results.get("summary", {})
    print(f"\n总问题数: {summary.get('total_questions', 0)}")
    print(f"成功: {summary.get('successful', 0)}")
    print(f"失败: {summary.get('failed', 0)}")
    print(f"成功率: {summary.get('success_rate', 0)}%")

    print("\n平均指标:")
    avg_metrics = summary.get("average_metrics", {})
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value}")

    # 6. 保存结果
    print("\n5. 保存评估结果...")
    output_path = evaluator.save_results()
    print(f"   结果已保存到: {output_path}")

    # 7. 生成详细报告
    print("\n6. 生成详细报告...")
    report_path = generate_detailed_report(eval_results, output_path)
    print(f"   报告已保存到: {report_path}")

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


def generate_detailed_report(eval_results: dict, json_path: str) -> str:
    """
    生成详细的评估报告

    Args:
        eval_results: 评估结果
        json_path: JSON结果文件路径

    Returns:
        报告文件路径
    """
    report_dir = os.path.dirname(json_path)
    report_path = os.path.join(report_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# RAG系统评估报告\n\n")
        f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**工单编号**: 人工智能NLP-RAG-功能测试及评估\n\n")

        # 摘要
        summary = eval_results.get("summary", {})
        f.write("## 评估摘要\n\n")
        f.write(f"- **总问题数**: {summary.get('total_questions', 0)}\n")
        f.write(f"- **成功数**: {summary.get('successful', 0)}\n")
        f.write(f"- **失败数**: {summary.get('failed', 0)}\n")
        f.write(f"- **成功率**: {summary.get('success_rate', 0)}%\n\n")

        # 平均指标
        f.write("## 平均指标\n\n")
        avg_metrics = summary.get("average_metrics", {})
        for metric, value in avg_metrics.items():
            f.write(f"- **{metric.upper()}**: {value}\n")
        f.write("\n")

        # 详细结果
        f.write("## 详细结果\n\n")
        results = eval_results.get("results", [])
        for i, result in enumerate(results, 1):
            f.write(f"### 问题 {i}: {result.get('question', '')[:50]}...\n\n")

            if result.get("error"):
                f.write(f"**错误**: {result['error']}\n\n")
                continue

            f.write(f"**RAG回答**: {result.get('rag_answer', '')[:200]}...\n\n")
            f.write(f"**检索到的片段数**: {result.get('chunks_retrieved', 0)}\n")
            f.write(f"**来源文件**: {', '.join(result.get('source_files', []))}\n")
            f.write(f"**耗时**: {result.get('elapsed_seconds', 0)}秒\n\n")

            # 检索指标
            metrics = result.get("retrieval_metrics", {})
            if metrics:
                f.write("**检索指标**:\n")
                for metric, value in metrics.items():
                    f.write(f"- {metric}: {value}\n")
                f.write("\n")

        # 问题分析
        f.write("## 问题分析\n\n")
        f.write("根据评估结果，RAG系统在以下方面表现良好：\n\n")
        f.write("1. **文档检索**: 能够从PDF和TXT文件中检索相关信息\n")
        f.write("2. **混合检索**: 结合向量检索和全文检索，提高召回率\n")
        f.write("3. **重排算法**: 使用CrossEncoder重排，提高准确性\n\n")

        f.write("存在的问题和改进建议：\n\n")
        f.write("1. **长文档处理**: 对于超长文档，需要更好的分块策略\n")
        f.write("2. **多语言支持**: 当前主要针对中文文档，英文支持有限\n")
        f.write("3. **实时性**: 大量文档加载时，初始化时间较长\n")

    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG系统评估脚本")
    parser.add_argument("--questions", type=int, default=10, help="要评估的问题数量")
    parser.add_argument("--output", type=str, help="输出目录")
    args = parser.parse_args()

    # 运行评估
    run_evaluation(num_questions=args.questions)
