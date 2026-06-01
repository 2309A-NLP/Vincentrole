"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
Query理解模块 - 意图识别、消歧、分解
"""

import re
from typing import Optional


class QueryUnderstanding:
    """
    Query理解引擎。

    功能:
    1. 意图识别 - 判断用户想了解什么类型的信息
    2. 消歧 - 处理模糊表述，提取关键实体
    3. 分解 - 将复杂问题拆解为子问题
    4. 查询重写 - 优化查询文本以提高检索效果
    """

    # 常见金融/招股书问题类型
    INTENT_PATTERNS = {
        "财务数据": [
            "收入", "利润", "净利润", "毛利率", "营收", "成本",
            "现金流", "资产", "负债", "净资产",
        ],
        "公司信息": [
            "注册资本", "法定代表人", "成立", "注册地址",
            "主营业务", "经营范围", "员工",
        ],
        "风险因素": [
            "风险", "不确定性", "挑战", "不利",
        ],
        "行业分析": [
            "行业", "市场", "竞争", "上下游", "产业链",
        ],
        "募资用途": [
            "募集资金", "募投", "投资", "项目",
        ],
        "股权结构": [
            "股东", "持股", "股权", "实际控制人",
        ],
    }

    # 金融实体提取模式
    ENTITY_PATTERNS = {
        "公司名": r'[\u4e00-\u9fa5]{2,}(?:股份|有限|集团|公司)',
        "金额": r'\d+(?:\.\d+)?\s*(?:万|亿|元|万元|亿元)',
        "年份": r'\d{4}\s*年',
        "比例": r'\d+(?:\.\d+)?\s*%',
    }

    # English intent patterns
    EN_INTENT_PATTERNS = {
        "financial_data": [
            "revenue", "profit", "net income", "gross margin", "cost",
            "cash flow", "assets", "liabilities", "equity", "earnings",
            "income", "expense", "EBITDA", "operating",
        ],
        "company_info": [
            "registered capital", "legal representative", "established",
            "registered address", "business scope", "employees",
            "incorporation", "founded",
        ],
        "risk_factors": [
            "risk", "uncertainty", "challenge", "adverse", "negative",
            "downside", "volatility",
        ],
        "industry_analysis": [
            "industry", "market", "competition", "supply chain",
            "market share", "sector", "segment",
        ],
        "fundraising": [
            "offering", "proceeds", "use of funds", "investment",
            "capital raising", "IPO", "listing",
        ],
        "equity_structure": [
            "shareholder", "ownership", "equity", "controlling",
            "stake", "share",
        ],
    }

    # English entity extraction patterns
    EN_ENTITY_PATTERNS = {
        "company_name": r'[A-Z][a-zA-Z]+(?:,\s*(?:Inc|Ltd|Corp|LLC|PLC|Group|Holdings))?(?:\s+(?:Inc|Ltd|Corp|LLC|PLC|Group|Holdings))?',
        "amount": r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?',
        "year": r'\b(?:19|20)\d{2}\b',
        "percentage": r'\d+(?:\.\d+)?\s*%',
    }

    def __init__(self):
        pass

    def detect_language(self, question: str) -> str:
        """
        Detect whether the question is in Chinese or English.

        Returns:
            str: "zh" for Chinese, "en" for English, "unknown" if cannot determine.
        """
        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', question))
        # Count English words (sequences of Latin letters)
        english_words = len(re.findall(r'[a-zA-Z]+', question))

        if chinese_chars > english_words:
            return "zh"
        elif english_words > chinese_chars and english_words >= 3:
            return "en"
        else:
            # Very short or ambiguous: check first substantive character
            if chinese_chars > 0:
                return "zh"
            return "en"

    def analyze(self, question: str) -> dict:
        """
        全面分析用户问题。

        Returns:
            dict: {
                "intent": "财务数据" / "风险因素" / ...,
                "entities": {"公司名": [...], "金额": [...], ...},
                "is_complex": True/False,  # 是否需要分解
                "sub_questions": [...],     # 子问题列表
                "rewritten_query": "...",   # 重写后的查询
            }
        """
        lang = self.detect_language(question)
        intent_patterns = self.EN_INTENT_PATTERNS if lang == "en" else self.INTENT_PATTERNS
        entity_patterns = self.EN_ENTITY_PATTERNS if lang == "en" else self.ENTITY_PATTERNS

        intent = self._identify_intent(question, intent_patterns)
        entities = self._extract_entities(question, entity_patterns)
        sub_questions = self._decompose(question)
        rewritten = self._rewrite_query(question, intent, entities)

        return {
            "intent": intent,
            "entities": entities,
            "is_complex": len(sub_questions) > 1,
            "sub_questions": sub_questions,
            "rewritten_query": rewritten,
            "original": question,
            "language": lang,
        }

    def _identify_intent(self, question: str, patterns: dict = None) -> str:
        """识别问题的主要意图"""
        if patterns is None:
            patterns = self.INTENT_PATTERNS
        scores = {}
        for intent, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in question)
            if score > 0:
                scores[intent] = score
        if scores:
            return max(scores, key=scores.get)
        return "通用"

    def _extract_entities(self, question: str, patterns: dict = None) -> dict:
        """提取问题中的关键实体"""
        if patterns is None:
            patterns = self.ENTITY_PATTERNS
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, question)
            if matches:
                entities[entity_type] = list(set(matches))
        return entities

    def _decompose(self, question: str) -> list[str]:
        """将复杂问题分解为子问题"""
        sub_questions = []

        # 包含"和"、"与"、"及"的多实体问题
        if re.search(r'[和与及]、', question):
            parts = re.split(r'[和与及]、', question)
            if len(parts) >= 2:
                sub_questions = [p.strip() for p in parts if p.strip()]

        # "分别" 通常意味着多值查询
        if "分别" in question:
            sub_questions.append(question)

        # 复杂问题：包含"为什么"/"如何"的推理型问题
        if any(w in question for w in ["为什么", "如何", "怎么"]):
            sub_questions.append(question)

        return sub_questions if sub_questions else [question]

    def _rewrite_query(self, question: str, intent: str, entities: dict) -> str:
        """重写查询，增强检索效果"""
        # 如果问题较短且包含代词，尝试补全
        rewritten = question

        # 移除疑问词但保留核心信息（对向量检索更友好）
        replacements = [
            (r'请问', ''),
            (r'我想知道', ''),
            (r'能否告诉[我]?', ''),
        ]
        for pattern, replacement in replacements:
            rewritten = re.sub(pattern, replacement, rewritten)

        # 补充intent标签
        if intent not in ["通用", rewritten]:
            rewritten = f"{intent}: {rewritten}"

        return rewritten.strip()

    def extract_key_info(self, question: str) -> dict:
        """
        快速提取问题的关键信息（简化版分析）。

        Returns:
            {"question": "...", "entity": "...", "metric": "..."}
        """
        entities = self._extract_entities(question)
        key_company = ""
        if "公司名" in entities:
            key_company = entities["公司名"][0]

        # 提取查询指标
        metrics = []
        for m in ["收入", "利润", "占比", "比重", "注册资本", "法定代表人"]:
            if m in question:
                metrics.append(m)

        return {
            "question": question,
            "公司": key_company,
            "指标": metrics,
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    qu = QueryUnderstanding()
    tests = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        "公司注册资本是多少？法定代表人是谁？",
    ]
    for q in tests:
        result = qu.analyze(q)
        print(f"问题: {q}")
        print(f"  意图: {result['intent']}")
        print(f"  实体: {result['entities']}")
        print(f"  重写: {result['rewritten_query']}")
        print()
