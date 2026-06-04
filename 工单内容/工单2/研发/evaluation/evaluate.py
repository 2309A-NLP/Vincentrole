"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
RAG评估体系 — Faithfulness + Answer Relevancy评分

评估维度:
1. Faithfulness（忠实度）: 答案是否基于检索到的文档内容
2. Answer Relevancy（答案相关性）: 答案是否针对问题
3. Context Precision（上下文精度）: 检索结果的相关性

注意：需要 API Key 调用 LLM 进行评估。
如果未配置 API Key，使用基于规则的简易评估。
"""

import os
import json
import re
import sys
import time
from typing import List, Dict, Optional

# 尝试导入配置
sys.path.insert(0, os.path.expanduser("~/Desktop/工单2"))
try:
    import config
    LLM_API_KEY = config.LLM_CONFIG.get("API密钥", "")
    LLM_API_BASE = config.LLM_CONFIG.get("API地址", "https://api.deepseek.com/v1")
    LLM_MODEL = config.LLM_CONFIG.get("模型", "deepseek-v4-flash")
except Exception:
    LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    LLM_API_BASE = "https://api.deepseek.com/v1"
    LLM_MODEL = "deepseek-v4-flash"


TEST_QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
     "expected_answer": "2021年军用领域收入约1.2亿元，2022年约1.5亿元，2023年约1.8亿元"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
     "expected_answer": "参与制定了多项国家军用视频标准"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
     "expected_answer": "军用领域收入占比在85%以上"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
     "expected_answer": "上游涉及电子元器件、芯片、软件等供应商"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
     "expected_answer": "在军用视频处理领域已成为重要供应商"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
     "expected_answer": "下游主要包括国防军工、政府、安防等行业"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
     "expected_answer": "参与的某军用视频系统工程荣获国家科技进步一等奖"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
     "expected_answer": "注册资本为6,000万元人民币"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
     "expected_answer": "法定代表人信息可在招股说明书中查询"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
     "expected_answer": "计划使用募集资金约3.5亿元"},
]


def call_llm(prompt: str, system_prompt: str = "你是一个专业的RAG评估助手。") -> str:
    """调用LLM进行评估"""
    if not LLM_API_KEY:
        return ""
    
    import requests
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(
            f"{LLM_API_BASE.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


class RAGEvaluator:
    """RAG评估器"""

    def evaluate_faithfulness(self, answer: str, context: str) -> dict:
        """
        评估忠实度（Faithfulness）:
        答案中的每个声明是否都能从上下文中找到依据。
        
        返回:
            {"score": 0.0-1.0, "details": "..."}
        """
        if not answer or not context:
            return {"score": 0.0, "details": "答案或上下文为空"}

        # 使用LLM评估
        if LLM_API_KEY:
            prompt = f"""评估以下答案对给定文档片段的忠实度（Faithfulness）。

文档片段:
{context[:1500]}

答案:
{answer}

请按以下标准打分（0-100分）:
- 如果答案中的所有信息都能在文档中找到依据 → 90-100
- 如果大部分信息有依据，少数信息无法确认 → 60-89
- 如果部分信息有依据，部分无依据 → 30-59
- 如果答案内容在文档中无法找到依据 → 0-29

只输出数字分数，不要其他文字。
"""
            result = call_llm(prompt)
            if result:
                try:
                    score = float(re.search(r'\d+', result).group()) / 100
                    return {"score": min(max(score, 0), 1), "details": "LLM评估"}
                except Exception:
                    pass

        # 降级：基于规则的简易评估
        return self._rule_based_faithfulness(answer, context)

    def _rule_based_faithfulness(self, answer: str, context: str) -> dict:
        """基于规则的忠实度评估（降级方案）"""
        if not context:
            return {"score": 0.0, "details": "无上下文"}

        answer_lower = answer.lower()
        context_lower = context.lower()

        # 提取答案中的数字（通常是关键信息）
        answer_numbers = set(re.findall(r'\d+[\.,\d]*', answer))
        context_numbers = set(re.findall(r'\d+[\.,\d]*', context))

        # 提取答案中的中文关键短语
        answer_phrases = set(re.findall(r'[\u4e00-\u9fa5]{2,8}', answer))
        context_phrases = set(re.findall(r'[\u4e00-\u9fa5]{2,8}', context))

        # 数字覆盖率
        number_overlap = 0
        if answer_numbers:
            number_overlap = len(answer_numbers & context_numbers) / len(answer_numbers)

        # 短语覆盖率
        phrase_overlap = 0
        if answer_phrases:
            phrase_overlap = len(answer_phrases & context_phrases) / len(answer_phrases)

        combined_score = number_overlap * 0.5 + phrase_overlap * 0.5
        
        return {
            "score": round(min(max(combined_score, 0), 1), 3),
            "details": f"基于规则（数字覆盖:{number_overlap:.2f}, 短语覆盖:{phrase_overlap:.2f}）",
        }

    def evaluate_relevancy(self, answer: str, question: str) -> dict:
        """
        评估答案相关性（Answer Relevancy）:
        答案是否直接回应了问题。
        
        返回:
            {"score": 0.0-1.0, "details": "..."}
        """
        if not answer:
            return {"score": 0.0, "details": "答案为空"}

        # 使用LLM评估
        if LLM_API_KEY:
            prompt = f"""评估以下答案对问题的相关性（Relevancy）。

问题: {question}

答案: {answer[:500]}

请按以下标准打分（0-100分）:
- 答案直接回答了问题，包含关键信息 → 90-100
- 答案回答了问题但不够具体 → 60-89
- 答案部分回答了问题 → 30-59
- 答案与问题无关 → 0-29

只输出数字分数，不要其他文字。
"""
            result = call_llm(prompt)
            if result:
                try:
                    score = float(re.search(r'\d+', result).group()) / 100
                    return {"score": min(max(score, 0), 1), "details": "LLM评估"}
                except Exception:
                    pass

        # 降级：基于规则
        return self._rule_based_relevancy(answer, question)

    def _rule_based_relevancy(self, answer: str, question: str) -> dict:
        """基于规则的答案相关性评估"""
        q_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', question))
        a_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', answer))

        if not q_keywords:
            return {"score": 0.5, "details": "规则评估（问题无关键词）"}

        overlap = len(q_keywords & a_keywords) / len(q_keywords)
        return {
            "score": round(min(overlap, 1), 3),
            "details": f"规则评估（关键词重叠: {overlap:.2f}）",
        }

    def evaluate_context_precision(self, results: list, question: str) -> dict:
        """
        评估上下文精度（Context Precision）:
        前k个检索结果中，真正相关的结果比例。
        
        使用关键词匹配作为相关性的粗略判断。
        """
        if not results:
            return {"score": 0.0, "details": "无检索结果"}

        q_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', question.lower()))

        relevant_count = 0
        for i, r in enumerate(results):
            text = r.get("text", "").lower()
            score = r.get("final_score", r.get("score", 0))
            
            # 判断是否相关：有关键词匹配 或 分数较高
            text_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', text))
            word_overlap = len(q_words & text_words) / max(len(q_words), 1)
            
            if word_overlap > 0.2 or score > 0.02:
                relevant_count += 1

        precision = relevant_count / len(results) if results else 0
        return {
            "score": round(precision, 3),
            "details": f"前{len(results)}个结果中{relevant_count}个相关",
        }

    def evaluate(self, question: str, context: str, answer: str,
                 results: list = None) -> dict:
        """综合评估"""
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevancy = self.evaluate_relevancy(answer, question)
        
        context_precision = {"score": 0, "details": "无检索结果"}
        if results:
            context_precision = self.evaluate_context_precision(results, question)

        overall = round((faithfulness["score"] * 0.4 + relevancy["score"] * 0.4
                        + context_precision["score"] * 0.2), 3)

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "context_precision": context_precision,
            "overall": overall,
        }


def main():
    """运行评估"""
    print("=" * 70)
    print("  RAG评估体系")
    print("  工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统")
    print("=" * 70)

    # 检查是否有API Key
    if not LLM_API_KEY:
        print("  [提示] 未配置API Key，使用基于规则的简易评估")
    else:
        print(f"  [信息] 使用 {LLM_MODEL} 进行评估")

    # 初始化系统
    sys.path.insert(0, os.path.expanduser("~/Desktop/工单2"))
    from qa_engine.orchestrator import RAGSystem

    pdf_path = os.path.expanduser("~/Desktop/工单2/data/招股说明书1.pdf")
    system = RAGSystem(use_cache=True)

    # 加载PDF（尝试从缓存加载）
    load_result = system.load_pdf(pdf_path)
    print(f"  加载结果: {load_result.get('status', 'unknown')}, {load_result.get('chunks', 0)} chunks\n")

    if load_result.get("status") == "error":
        print(f"  [错误] 加载失败: {load_result.get('error')}")
        return

    evaluator = RAGEvaluator()
    all_results = []

    for q in TEST_QUESTIONS[:5]:  # 先跑5个
        question = q["question"]
        qid = q["id"]
        expected = q.get("expected_answer", "")

        print(f"\n  评估问题 [{qid}]: {question[:50]}...")

        # 检索
        retrieval = system.retriever.retrieve(question, top_k=5, threshold=0.0)
        context = retrieval.get("context_text", "")
        results = retrieval.get("results", [])

        # 生成
        gen = system.generator.generate(question, context)
        answer = gen.get("answer", "")

        # 评估
        eval_result = evaluator.evaluate(question, context, answer, results)

        print(f"    忠实度(Faithfulness):    {eval_result['faithfulness']['score']:.3f}")
        print(f"    相关性(Relevancy):       {eval_result['relevancy']['score']:.3f}")
        print(f"    上下文精度(Context Prec): {eval_result['context_precision']['score']:.3f}")
        print(f"    综合分(Overall):          {eval_result['overall']:.3f}")

        all_results.append({
            "id": qid,
            "question": question,
            "evaluation": eval_result,
        })

    # 汇总
    if all_results:
        avg_faith = sum(r["evaluation"]["faithfulness"]["score"] for r in all_results) / len(all_results)
        avg_rel = sum(r["evaluation"]["relevancy"]["score"] for r in all_results) / len(all_results)
        avg_prec = sum(r["evaluation"]["context_precision"]["score"] for r in all_results) / len(all_results)
        avg_overall = sum(r["evaluation"]["overall"] for r in all_results) / len(all_results)

        print(f"\n{'='*70}")
        print(f"  评估汇总（{len(all_results)}个问题）")
        print(f"{'='*70}")
        print(f"  平均忠实度(Faithfulness):    {avg_faith:.3f} ({avg_faith*100:.1f}%)")
        print(f"  平均相关性(Relevancy):       {avg_rel:.3f} ({avg_rel*100:.1f}%)")
        print(f"  平均上下文精度(Context Prec): {avg_prec:.3f} ({avg_prec*100:.1f}%)")
        print(f"  综合评分(Overall):            {avg_overall:.3f} ({avg_overall*100:.1f}%)")

    # 保存
    output_path = os.path.expanduser("~/Desktop/工单2/evaluation/rag_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  评估结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
