"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
检索器 - 查询扩展 + 重排序 + 混合检索（支持中英文、容错）
"""

import re
from typing import List, Dict, Optional


def detect_language(text: str) -> str:
    """检测查询语言"""
    if not text:
        return "zh"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_chars = len(re.findall(r'[a-zA-Z]', text))
    total = max(chinese_chars + ascii_chars, 1)
    if chinese_chars / total >= 0.3:
        return "zh"
    return "en"



EN_COMPANY_MAP = {
    "china pacific insurance": "中国太保", "china pacific": "中国太保", "cpic": "中国太保",
    "baosteel": "宝钢", "baoshan iron": "宝钢", "fuli zhaocai": "涪陵榨菜",
    "sany heavy": "三一重工", "sany": "三一重工",
    "china southern airlines": "南方航空", "china southern": "南方航空",
    "midea group": "美的", "midea": "美的",
    "shuanghui": "双汇", "wh group": "双汇",
    "haitian flavouring": "海天味业", "haitian": "海天味业",
    "luzhou laojiao": "泸州老窖", "luzhou": "泸州老窖",
}

EN_TERM_MAP = {
    "operating revenue": "营业收入", "revenue": "营业收入", "net profit": "净利润",
    "gross margin": "毛利率", "net income": "净利润", "cash flow": "现金流",
    "total assets": "总资产", "total liabilities": "总负债", "shareholder": "股东",
    "dividend": "分红", "registered capital": "注册资本", "legal representative": "法定代表人",
    "business scope": "经营范围", "main business": "主营业务",
    "main business segments": "主营业务", "employees": "员工",
    "industry": "行业", "market share": "市场份额", "profitability": "盈利能力",
}


def translate_en_to_zh(query: str) -> str:
    """英文查询翻译为中文（基于映射表）"""
    zh_query = query
    for en_name, zh_name in EN_COMPANY_MAP.items():
        if en_name in zh_query.lower():
            zh_query = re.sub(re.escape(en_name), zh_name, zh_query, flags=re.IGNORECASE)
    for en_term, zh_term in EN_TERM_MAP.items():
        if en_term in zh_query.lower():
            zh_query = re.sub(re.escape(en_term), zh_term, zh_query, flags=re.IGNORECASE)
    return zh_query

class Retriever:
    """
    检索器 - 支持查询扩展、混合检索、重排序（中英文）

    流程:
    1. 查询分析（语言检测、意图识别、实体提取）
    2. 查询扩展（中英文同义词）
    3. 混合检索（向量 + BM25）
    4. 重排序（关键词 + 标题 + 表格加权）
    5. 上下文整合
    """

    def __init__(self, vector_store, embedding_model,
                 expand_query: bool = True,
                 rerank_top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.expand_query = expand_query
        self.rerank_top_k = rerank_top_k

        # 中文同义词扩展表
        self.zh_expansions = {
            "注册资本": ["注册资本", "注资金", "股本", "股份资本"],
            "营业收入": ["营业收入", "主营业务收入", "收入", "营收"],
            "上游": ["上游", "供应商", "原材料"],
            "下游": ["下游", "客户", "用户"],
            "核心竞争力": ["核心竞争力", "核心技术", "技术优势"],
            "风险": ["风险", "不确定性", "威胁"],
            "筹资": ["筹资", "募集资金", "融资", "募投"],
            "使用本次发行募集资金": ["使用本次发行募集资金", "募集资金用途", "资金用途", "投资项目"],
            "军用领域": ["军用领域", "国防", "军工", "军品", "军事"],
            "收入占比": ["收入占比", "比重", "主营业务收入占比", "收入比例"],
            "军用领域收入": ["军用领域收入", "军品销售收入", "军品销售额", "国防收入", "军品收入", "军队收入"],
            "收入分别是多少": ["收入分别是多少", "收入金额", "收入数据", "收入表", "各期收入"],
        }

        # 英文同义词扩展表
        self.en_expansions = {
            "registered capital": ["registered capital", "share capital", "equity", "capital stock"],
            "revenue": ["revenue", "income", "sales", "turnover", "operating income"],
            "upstream": ["upstream", "suppliers", "raw materials"],
            "downstream": ["downstream", "customers", "clients"],
            "risk": ["risk", "uncertainty", "threat"],
            "military": ["military", "defense", "defence", "armed forces"],
            "technology standard": ["technology standard", "technical standard", "specification"],
            "fundraising": ["fundraising", "proceeds", "funds raised", "capital raised"],
        }

    def retrieve(self, question: str, top_k: int = 5,
                 threshold: float = 0.0) -> Dict:
        """
        检索主入口（含容错）。

        流程:
        1. 查询分析
        2. 查询扩展
        3. 混合检索（向量 + BM25）
        4. 重排序
        5. 上下文整合
        """
        try:
            import time
            t0 = time.time()
            lang = detect_language(question)
            if lang == "en":
                search_query = translate_en_to_zh(question)
                print(f"  [检索] 英文翻译: {question[:40]} → {search_query[:40]}")
            else:
                search_query = question
            print(f"  [检索] 开始 | 语言={lang} | query={search_query[:50]}")

            # Step 1: 查询分析
            query_analysis = self._analyze_query(search_query, "zh" if lang == "en" else lang)

            # Step 2: 查询扩展
            expanded = self._expand_query(search_query, "zh" if lang == "en" else lang) if self.expand_query else search_query
            if expanded != search_query:
                print(f"  [检索] 查询扩展: → {expanded[:50]}")

            # Step 3: 向量化查询
            query_emb = self.embedding_model.encode([expanded], is_query=True)[0]

            # Step 4: 混合检索
            raw_results = self.vector_store.search(
                query_embedding=query_emb,
                query_text=expanded,
                top_k=top_k * 2,
            )
            print(f"  [检索] 检索完成: {len(raw_results)} 条结果")

            # Step 5: 重排序
            reranked = self._rerank(search_query, raw_results, "zh" if lang == "en" else lang)

            # 过滤低分结果
            filtered = [r for r in reranked if r.get("final_score", 0) >= threshold]
            if not filtered:
                filtered = reranked[:top_k]

            final_results = filtered[:top_k]

            # 构建上下文
            context_text = self._build_context(final_results, query_analysis)
            elapsed = time.time() - t0
            print(f"  [检索] 完成 | {len(final_results)} 条 | 耗时 {elapsed:.2f}s")

            return {
                "results": final_results,
                "context_text": context_text,
                "total_results": len(final_results),
                "query_analysis": query_analysis,
            }
        except Exception as e:
            # 容错：检索失败时返回空结果
            return {
                "results": [],
                "context_text": "",
                "total_results": 0,
                "query_analysis": {
                    "lang": detect_language(question),
                    "intent": "通用" if detect_language(question) == "zh" else "general",
                    "entities": [],
                    "keywords": [],
                    "error": str(e),
                },
            }

    # ============================================================
    # 查询分析
    # ============================================================

    def _analyze_query(self, question: str, lang: str) -> Dict:
        """分析查询意图和关键实体"""
        q = question.lower()

        # 意图分类（中英文）
        if lang == "zh":
            intents = {
                "数值查询": ["多少", "几", "百分比", "万元", "数量", "总额", "占比", "比重"],
                "实体识别": ["是谁", "什么", "哪个", "何人", "谁"],
                "时间查询": ["什么时候", "哪年", "时间", "期间"],
                "因果分析": ["为什么", "原因", "影响"],
            }
        else:
            intents = {
                "numeric_query": ["how much", "how many", "percentage", "amount", "total", "ratio"],
                "entity_query": ["who", "what", "which"],
                "time_query": ["when", "what year", "what time", "period"],
                "causal_query": ["why", "reason", "impact", "cause"],
            }

        detected_intent = "通用" if lang == "zh" else "general"
        for intent, keywords in intents.items():
            if any(kw in q for kw in keywords):
                detected_intent = intent
                break

        # 实体提取
        entities = self._extract_entities(question, lang)
        # 关键词提取
        keywords = self._extract_keywords(question)

        return {
            "lang": lang,
            "intent": detected_intent,
            "entities": entities,
            "keywords": keywords,
        }

    def _extract_entities(self, text: str, lang: str) -> List[str]:
        """提取命名实体"""
        entities = []

        if lang == "zh":
            # 公司名
            company_pattern = r'[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限公司|科技公司|集团公司)'
            companies = re.findall(company_pattern, text)
            entities.extend(companies)
            # 人名
            person_pattern = r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|总经|董事长)'
            persons = re.findall(person_pattern, text)
            entities.extend(persons)
            # 数值+单位
            number_pattern = r'\d+(?:\.\d+)?[％%万亿元人家]'
            numbers = re.findall(number_pattern, text)
            entities.extend(numbers)
        else:
            # English company names
            company_pattern = r'[A-Z][A-Za-z\s,]{2,}(?:Inc|Corp|Ltd|Limited|Co|Company)'
            companies = re.findall(company_pattern, text)
            entities.extend(companies)
            # Dollar amounts
            amount_pattern = r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|yuan)?'
            amounts = re.findall(amount_pattern, text)
            entities.extend(amounts)

        return list(set(entities))

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 中文停用词
        zh_stopwords = set("的是了在和与有从为不也对于其以及之中上去可但而或如何什么哪个哪些多少这那")
        # 英文停用词
        en_stopwords = set("the a an is are was were be been have has had do does did will would could should may might shall can must need dare ought used".split())

        # 中文词
        zh_words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        # 英文词
        en_words = re.findall(r'[a-zA-Z]{3,}', text.lower())

        all_stopwords = zh_stopwords | en_stopwords
        return [w for w in zh_words + en_words if w not in all_stopwords]

    # ============================================================
    # 查询扩展
    # ============================================================

    def _expand_query(self, question: str, lang: str) -> str:
        """查询扩展：添加同义词"""
        expansions = self.zh_expansions if lang == "zh" else self.en_expansions
        expanded_terms = []
        for key, synonyms in expansions.items():
            if key.lower() in question.lower():
                expanded_terms.extend(synonyms)
        if expanded_terms:
            return f"{question} {' '.join(set(expanded_terms))}"
        return question

    # ============================================================
    # 重排序
    # ============================================================

    def _rerank(self, question: str, results: List[Dict], lang: str) -> List[Dict]:
        """重排序：关键词匹配度 + 位置信息"""
        if not results:
            return results

        q_words = set(self._extract_keywords(question))

        reranked = []
        for r in results:
            chunk = r.get("chunk", {})
            text = chunk.get("text", "")
            heading = chunk.get("heading", "")
            page = chunk.get("page", 0)

            # 关键词匹配度
            text_words = set(self._extract_keywords(text + " " + heading))
            keyword_overlap = len(q_words & text_words) / max(len(q_words), 1)

            # 标题匹配奖励
            heading_bonus = 0.0
            if heading:
                heading_words = set(self._extract_keywords(heading))
                heading_bonus = len(q_words & heading_words) * 0.15

            # 页码位置因子
            page_factor = 1.0
            if 1 <= page <= 10:
                page_factor = 1.05

            # 表格奖励
            type_bonus = 0.1 if chunk.get("type") == "table" else 0.0

            # 综合分
            base_score = r.get("score", 0)
            final_score = base_score * (1 + keyword_overlap) * page_factor + heading_bonus + type_bonus

            reranked.append({
                "text": text,
                "page": page,
                "type": chunk.get("type", "unknown"),
                "heading": heading,
                "score": base_score,
                "keyword_overlap": keyword_overlap,
                "final_score": final_score,
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    # ============================================================
    # 上下文构建
    # ============================================================

    def _build_context(self, results: List[Dict], query_analysis: Dict) -> str:
        """构建LLM上下文"""
        parts = []
        for i, r in enumerate(results):
            text = r["text"]
            page = r["page"]
            heading = r.get("heading", "")

            if query_analysis.get("lang", "zh") == "zh":
                header = f"[片段 {i+1}] 第{page}页"
                if heading:
                    header += f" | 标题: {heading}"
            else:
                header = f"[Chunk {i+1}] Page {page}"
                if heading:
                    header += f" | Heading: {heading}"
            parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(parts)

    def format_context_for_llm(self, context: str, max_chars: int = 3000) -> str:
        """截断上下文以适应LLM token限制"""
        if len(context) <= max_chars:
            return context
        return context[:max_chars] + "\n... [上下文已截断/Context truncated]"
