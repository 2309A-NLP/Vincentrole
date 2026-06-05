"""
工单编号: 人工智能NLP-RAG-功能测试及评估
RAG评估模块 - 使用sample_questions.pdf中的问题进行测试评估
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, rag_system, questions: List[Dict] = None):
        """
        初始化评估器

        Args:
            rag_system: RAG系统实例
            questions: 问题列表，每个问题包含question和reference_answer
        """
        self.rag_system = rag_system
        self.questions = questions or []
        self.results = []

    def load_questions_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF文件加载问题

        Args:
            pdf_path: sample_questions.pdf路径

        Returns:
            问题列表
        """
        try:
            # 动态导入PDF解析器
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

            self.questions = questions
            print(f"[Evaluator] 加载了 {len(questions)} 个问题")
            return questions

        except Exception as e:
            print(f"[Evaluator] 加载问题失败: {e}")
            return []

    def evaluate_single(self, question: Dict) -> Dict:
        """
        评估单个问题

        Args:
            question: 问题字典，包含question和reference_answer

        Returns:
            评估结果
        """
        try:
            start_time = time.time()

            # 使用RAG系统回答问题
            result = self.rag_system.ask(question["question"])

            elapsed = time.time() - start_time

            # 提取关键信息
            rag_answer = result.get("rag_answer", "")
            source_chunks = result.get("source_chunks", [])
            source_files = result.get("source_files", [])
            source_pages = result.get("source_pages", [])

            # 计算检索指标
            retrieval_metrics = self._calculate_retrieval_metrics(
                question, rag_answer, source_chunks
            )

            return {
                "question": question["question"],
                "reference_answer": question.get("reference_answer", ""),
                "rag_answer": rag_answer,
                "source_files": source_files,
                "source_pages": source_pages,
                "chunks_retrieved": len(source_chunks),
                "elapsed_seconds": round(elapsed, 2),
                "retrieval_metrics": retrieval_metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "question": question["question"],
                "reference_answer": question.get("reference_answer", ""),
                "rag_answer": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_retrieval_metrics(self, question: Dict, rag_answer: str, source_chunks: List[Dict]) -> Dict:
        """
        计算检索指标

        Args:
            question: 问题
            rag_answer: RAG回答
            source_chunks: 检索到的chunks

        Returns:
            检索指标
        """
        metrics = {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }

        # 如果没有参考答案，无法计算精确指标
        if not question.get("reference_answer"):
            return metrics

        # 简单文本匹配评估
        reference = question["reference_answer"].lower()
        rag_lower = rag_answer.lower()

        # 检查关键信息是否被检索到
        key_phrases = self._extract_key_phrases(reference)
        retrieved_phrases = []

        for chunk in source_chunks:
            chunk_text = chunk.get("text", "").lower()
            for phrase in key_phrases:
                if phrase.lower() in chunk_text:
                    retrieved_phrases.append(phrase)

        # 计算召回率
        if key_phrases:
            unique_retrieved = set(retrieved_phrases)
            metrics["recall"] = len(unique_retrieved) / len(key_phrases)

        # 检查答案中是否包含关键信息
        answer_contains = 0
        for phrase in key_phrases:
            if phrase.lower() in rag_lower:
                answer_contains += 1

        if key_phrases:
            metrics["precision"] = answer_contains / len(key_phrases)

        # 计算F1分数
        if metrics["recall"] + metrics["precision"] > 0:
            metrics["f1"] = 2 * metrics["recall"] * metrics["precision"] / (metrics["recall"] + metrics["precision"])

        return metrics

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        从文本中提取关键短语

        Args:
            text: 输入文本

        Returns:
            关键短语列表
        """
        # 简单的关键词提取
        import jieba
        import jieba.analyse

        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)

        # 添加一些常见金融术语
        financial_terms = [
            "拨备覆盖率", "资产质量", "不良贷款", "逾期贷款", "资本充足率",
            "净利润", "营业收入", "业务结构", "风险控制", "贷款结构",
        ]

        # 合并关键词
        key_phrases = list(set(keywords + financial_terms))

        # 过滤太短的词
        key_phrases = [p for p in key_phrases if len(p) >= 2]

        return key_phrases[:15]  # 返回前15个关键短语

    def evaluate_all(self) -> Dict:
        """
        评估所有问题

        Returns:
            完整评估结果
        """
        if not self.questions:
            return {"error": "没有要评估的问题"}

        results = []
        total_metrics = {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }

        print(f"[Evaluator] 开始评估 {len(self.questions)} 个问题...")

        for i, question in enumerate(self.questions):
            print(f"\n[问题 {i+1}/{len(self.questions)}] {question['question'][:50]}...")

            result = self.evaluate_single(question)
            results.append(result)

            # 累加指标
            if "retrieval_metrics" in result:
                for metric, value in result["retrieval_metrics"].items():
                    total_metrics[metric] += value

            # 显示进度
            if result.get("error"):
                print(f"  ❌ 错误: {result['error']}")
            else:
                print(f"  ✅ 耗时: {result['elapsed_seconds']}s")
                print(f"  📊 检索到 {result['chunks_retrieved']} 个片段")
                print(f"  📁 来源: {', '.join(result.get('source_files', [])[:2])}")

        # 计算平均指标
        avg_metrics = {}
        for metric, value in total_metrics.items():
            avg_metrics[metric] = round(value / len(self.questions), 3)

        self.results = results

        return {
            "questions_evaluated": len(results),
            "average_metrics": avg_metrics,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self, output_path: str = None) -> str:
        """
        保存评估结果到文件

        Args:
            output_path: 输出文件路径

        Returns:
            保存的文件路径
        """
        if not self.results:
            return ""

        if output_path is None:
            # 默认保存到evaluation_results目录
            eval_dir = getattr(config, "EVALUATION_CONFIG", {}).get("评估输出目录", "./data/evaluation_results")
            os.makedirs(eval_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(eval_dir, f"evaluation_results_{timestamp}.json")

        # 准备保存的数据
        save_data = {
            "evaluation_time": datetime.now().isoformat(),
            "questions_evaluated": len(self.results),
            "results": self.results,
            "summary": self._generate_summary(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"[Evaluator] 评估结果已保存到: {output_path}")
        return output_path

    def _generate_summary(self) -> Dict:
        """生成评估摘要"""
        if not self.results:
            return {}

        total = len(self.results)
        successful = sum(1 for r in self.results if not r.get("error"))
        failed = total - successful

        # 统计指标
        metrics_sum = {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }

        for result in self.results:
            if "retrieval_metrics" in result:
                for metric, value in result["retrieval_metrics"].items():
                    if metric in metrics_sum:
                        metrics_sum[metric] += value

        avg_metrics = {}
        for metric, value in metrics_sum.items():
            avg_metrics[metric] = round(value / total, 3) if total > 0 else 0.0

        return {
            "total_questions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0,
            "average_metrics": avg_metrics,
        }


# 导入config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
