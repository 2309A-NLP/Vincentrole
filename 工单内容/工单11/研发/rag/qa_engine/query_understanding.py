"""工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
Query理解模块 - 意图识别、消歧、分解、多语言支持
"""

import re
from typing import Optional, List

from .utils import detect_language


class QueryUnderstanding:
    """
    Query理解引擎（支持中英文）。

    功能:
    1. 语言检测 - 自动检测中文/英文
    2. 意图识别 - 判断用户想了解什么类型的信息
    3. 消歧 - 处理模糊表述，提取关键实体
    4. 分解 - 将复杂问题拆解为子问题
    5. 查询重写 - 优化查询文本以提高检索效果
    """

    # 中文意图模式
    ZH_INTENT_PATTERNS = {
        "财务数据": ["收入", "利润", "净利润", "毛利率", "营收", "成本", "现金流", "资产", "负债", "净资产"],
        "公司信息": ["注册资本", "法定代表人", "成立", "注册地址", "主营业务", "经营范围", "员工"],
        "风险因素": ["风险", "不确定性", "挑战", "不利"],
        "行业分析": ["行业", "市场", "竞争", "上下游", "产业链", "上游", "下游"],
        "募资用途": ["募集资金", "募投", "投资", "项目", "筹资", "发行"],
        "股权结构": ["股东", "持股", "股权", "实际控制人"],
    }

    # 英文意图模式
    EN_INTENT_PATTERNS = {
        "financial_data": ["revenue", "profit", "net income", "gross margin", "income", "cost", "cash flow", "asset", "liability", "equity"],
        "company_info": ["registered capital", "legal representative", "founded", "established", "address", "business", "employee"],
        "risk_factors": ["risk", "uncertainty", "challenge", "adverse"],
        "industry_analysis": ["industry", "market", "competition", "supply chain", "upstream", "downstream"],
        "fundraising": ["raised", "proceeds", "investment", "project", "funding"],
        "shareholding": ["shareholder", "stake", "equity", "controller"],
    }

    # 中文实体提取模式
    ZH_ENTITY_PATTERNS = {
        "公司名": r'[\u4e00-\u9fa5]{2,}(?:股份|有限|集团|公司)',
        "金额": r'\d+(?:\.\d+)?\s*(?:万|亿|元|万元|亿元)',
        "年份": r'\d{4}\s*年',
        "比例": r'\d+(?:\.\d+)?\s*%',
    }

    # 英文实体提取模式
    EN_ENTITY_PATTERNS = {
        "company": r'[A-Z][A-Za-z\s,]+(?:Inc|Corp|Ltd|Limited|Co)',
        "amount": r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|yuan|dollars)?',
        "year": r'\d{4}',
        "percentage": r'\d+(?:\.\d+)?\s*%',
    }

    def __init__(self):
        pass

    def analyze(self, question: str) -> dict:
        """
        全面分析用户问题（自动检测语言）。

        Returns:
            dict: {
                "lang": "zh"/"en",
                "intent": "...",
                "entities": {...},
                "is_complex": True/False,
                "sub_questions": [...],
                "rewritten_query": "...",
            }
        """
        lang = detect_language(question)
        intent = self._identify_intent(question, lang)
        entities = self._extract_entities(question, lang)
        sub_questions = self._decompose(question)
        rewritten = self._rewrite_query(question, intent)

        return {
            "lang": lang,
            "intent": intent,
            "entities": entities,
            "is_complex": len(sub_questions) > 1,
            "sub_questions": sub_questions,
            "rewritten_query": rewritten,
            "original": question,
        }

    def _identify_intent(self, question: str, lang: str) -> str:
        """识别问题的主要意图（按语言）"""
        patterns = self.ZH_INTENT_PATTERNS if lang == "zh" else self.EN_INTENT_PATTERNS
        scores = {}
        for intent, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw.lower() in question.lower())
            if score > 0:
                scores[intent] = score
        if scores:
            return max(scores, key=scores.get)
        return "通用" if lang == "zh" else "general"

    def _extract_entities(self, question: str, lang: str) -> dict:
        """提取问题中的关键实体（按语言）"""
        patterns = self.ZH_ENTITY_PATTERNS if lang == "zh" else self.EN_ENTITY_PATTERNS
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, question)
            if matches:
                entities[entity_type] = list(set(matches))

        # 如果没匹配到公司名，尝试用通用模式
        if not entities:
            # 中文：取前6个字以上的连续中文
            if lang == "zh":
                names = re.findall(r'[\u4e00-\u9fa5]{4,}', question)
                if names:
                    entities["公司名_猜测"] = names[:2]
            else:
                # 英文：取大写首字母的词组
                names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', question)
                if names:
                    entities["company_guess"] = names[:2]
        return entities

    def _decompose(self, question: str) -> list[str]:
        """将复杂问题分解为子问题"""
        sub_questions = []

        # 包含"和"、"与"、"及"、"and"的多实体问题
        if re.search(r'[和与及]、', question) or ' and ' in question.lower():
            parts = re.split(r'[和与及]、| and ', question)
            if len(parts) >= 2:
                sub_questions = [p.strip() for p in parts if p.strip()]

        # "分别" / "respectively" 通常意味着多值查询
        if "分别" in question or "respectively" in question.lower():
            sub_questions.append(question)

        # 复杂问题：包含"为什么"/"如何"/"why"/"how"的推理型问题
        if any(w in question for w in ["为什么", "如何", "怎么", "why", "how"]):
            sub_questions.append(question)

        return sub_questions if sub_questions else [question]

    def _rewrite_query(self, question: str, intent: str) -> str:
        """重写查询，增强检索效果"""
        rewritten = question

        # 移除客套前缀
        replacements = [
            (r'请问', ''),
            (r'我想知道', ''),
            (r'能否告诉[我]?', ''),
            (r'(please\s+)?(tell me|can you)\s+', '', re.IGNORECASE),
        ]
        for pattern, replacement, *flags in replacements:
            flag = flags[0] if flags else 0
            rewritten = re.sub(pattern, replacement, rewritten, flags=flag)

        return rewritten.strip()

    def extract_key_info(self, question: str) -> dict:
        """
        快速提取问题的关键信息（简化版分析）。
        Returns:
            {"question": "...", "company": "...", "metrics": [...], "lang": "zh"/"en"}
        """
        lang = detect_language(question)
        entities = self._extract_entities(question, lang)

        key_company = ""
        if lang == "zh":
            for key in ["公司名", "公司名_猜测"]:
                if key in entities and entities[key]:
                    key_company = entities[key][0]
                    break
        else:
            for key in ["company", "company_guess"]:
                if key in entities and entities[key]:
                    key_company = entities[key][0]
                    break

        # 提取查询指标
        if lang == "zh":
            metrics = [m for m in ["收入", "利润", "占比", "比重", "注册资本", "法定代表人"] if m in question]
        else:
            metrics = [m for m in ["revenue", "profit", "ratio", "capital", "representative", "percentage"] if m.lower() in question.lower()]

        return {
            "question": question,
            "company": key_company,
            "metrics": metrics,
            "lang": lang,
        }
