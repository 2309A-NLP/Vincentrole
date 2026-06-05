"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
检索器 - 表格感知检索（列头匹配+表格类型识别+多文档智能路由）
"""

import re
from typing import List, Dict, Optional

try:
    from ..config import IMAGE_QUERY_EXPANSIONS, IMAGE_CONFIG
except ImportError:
    try:
        from config import IMAGE_QUERY_EXPANSIONS, IMAGE_CONFIG
    except ImportError:
        IMAGE_QUERY_EXPANSIONS = {}
        IMAGE_CONFIG = {}


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


# 招股说明书金融术语 → 表格类型映射
TABLE_TYPE_ZH_MAP = {
    "发行股数": ["股本结构", "发行概况", "发行前后", "股本变化", "本次发行"],
    "发行后总股本": ["股本结构", "发行概况", "总股本", "发行后"],
    "募集资金": ["募集资金用途", "募投项目", "投资项目", "资金运用", "募集资金投资"],
    "投资项目": ["募集资金用途", "募投项目", "投资项目", "资金运用", "募集资金投资"],
    "控制关系": ["关联方关系", "关联方", "关联关系", "持股"],
    "不存在控制关系": ["关联方关系", "关联方", "关联关系"],
    "关联方": ["关联方关系", "关联方", "关联关系", "持股比例"],
    "持股比例": ["关联方关系", "关联方", "持股比例", "控制关系"],
    "军用领域收入": ["主营业务收入", "收入构成", "收入结构", "军品收入"],
    "收入占比": ["主营业务收入占比", "收入结构", "收入构成", "军品收入占比"],
    "成本": ["主营业务成本", "成本构成", "营业成本"],
    "注册资本": ["公司概况", "公司基本信息", "注册资本"],
    "法定代表人": ["公司概况", "公司基本信息", "法定代表人"],
}


class Retriever:
    """
    检索器 - 表格感知检索优化（工单3核心优化）。

    优化点:
    1. 表格类型感知查询扩展（金融术语→表格类型映射）
    2. 列头匹配（查询词匹配到表格列名时高权重提升）
    3. 表格标题匹配（查询词匹配到表格标题时优先）
    4. 公司名称智能路由（不同公司的问题自动匹配相应PDF的chunk）
    5. 表格内容加权（显著高于文本chunk的权重）
    6. 数值查询专项优化（发行股数、募集资金等数值问题的表格命中优化）
    """

    def __init__(self, vector_store, embedding_model,
                 expand_query: bool = True,
                 rerank_top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.expand_query = expand_query
        self.rerank_top_k = rerank_top_k

        # 中文同义词扩展表（增强版含表格类型映射）
        self.zh_expansions = {
            "注册资本": ["注册资本", "注资金", "股本", "股份资本", "公司概况"],
            "营业收入": ["营业收入", "主营业务收入", "收入", "营收", "收入构成"],
            "上游": ["上游", "供应商", "原材料", "行业上游"],
            "下游": ["下游", "客户", "用户", "行业下游"],
            "核心竞争力": ["核心竞争力", "核心技术", "技术优势"],
            "风险": ["风险", "不确定性", "威胁"],
            "筹资": ["筹资", "募集资金", "融资", "募投", "投资"],
            "使用本次发行募集资金": ["使用本次发行募集资金", "募集资金用途", "资金用途", "投资项目", "募投项目"],
            "军用领域": ["军用领域", "国防", "军工", "军品", "军事"],
            "收入占比": ["收入占比", "比重", "主营业务收入占比", "收入比例", "收入结构"],
            "军用领域收入": ["军用领域收入", "军品销售收入", "军品销售额", "国防收入", "军品收入", "军队收入", "收入构成"],
            "发行股数": ["发行股数", "发行股份", "本次发行", "发行概况", "股本结构"],
            "发行后总股本": ["发行后总股本", "总股本", "发行后", "股本结构", "发行概况"],
            "控制关系": ["控制关系", "控制", "关联方", "关联关系", "持股比例"],
            "关联方": ["关联方", "关联企业", "关联关系", "关联公司", "持股"],
            "持股比例": ["持股比例", "持股", "股权比例", "控股比例"],
            "募集资金投资": ["募集资金投资", "投资项目", "募投项目", "资金运用", "投资计划"],
            "技术标准": ["技术标准", "军用标准", "国家标准", "行业标准", "视频标准"],
            "国家科技进步一等奖": ["国家科技进步一等奖", "科技进步一等奖", "技术发明", "获奖", "工程"],
            # 图像相关问题扩展（工单4优化）
            "组织结构图": ["组织结构图", "组织架构", "销售部", "大客户销售部", "部门构成",
                         "内部组织结构", "职能部门", "公司治理结构", "股东大会", "董事会"],
            "销售部": ["销售部", "销售部门", "大客户销售部", "渠道销售部",
                      "电话及网络销售部", "国际贸易部", "销售架构", "销售处"],
            "大客户销售部": ["大客户销售部", "大客户销售", "大客户", "销售处",
                           "分公司", "销售处经理"],
            "IC市场": ["IC市场", "集成电路市场", "半导体市场", "应用结构",
                      "增长率", "赛迪顾问", "2008年", "应用行业", "工业控制"],
            "应用结构与增长": ["应用结构与增长", "市场应用结构", "增长率", "增长图",
                             "IC市场应用", "中国IC市场"],
        }

        # 英文同义词扩展表
        self.en_expansions = {
            "registered capital": ["registered capital", "share capital", "equity", "capital stock", "company profile"],
            "revenue": ["revenue", "income", "sales", "turnover", "operating income", "revenue breakdown"],
            "upstream": ["upstream", "suppliers", "raw materials"],
            "downstream": ["downstream", "customers", "clients"],
            "risk": ["risk", "uncertainty", "threat"],
            "military": ["military", "defense", "defence", "armed forces", "defense industry"],
            "technology standard": ["technology standard", "technical standard", "specification", "military standard"],
            "fundraising": ["fundraising", "proceeds", "funds raised", "capital raised", "investment project"],
            "shares issued": ["shares issued", "share capital", "equity structure", "offering"],
            "controlling relationship": ["controlling relationship", "related party", "shareholding", "affiliate"],
        }

    def retrieve(self, question: str, top_k: int = 5,
                 threshold: float = 0.0) -> Dict:
        """
        检索主入口（含表格感知优化 + 关键词兜底）。

        流程:
        1. 查询分析（含公司名称检测）
        2. 表格感知查询扩展（金融术语→表格类型）
        3. 混合检索（向量 + BM25）
        4. 重排序（含列头匹配、表格标题匹配、公司路由）
        5. 关键词兜底（向量分数太低时，用关键词扫描全文补漏）
        6. 上下文整合
        """
        try:
            import time
            t0 = time.time()
            lang = detect_language(question)
            # 英文查询自动翻译为中文（数据全是中文，bge-base-zh-v1.5 嵌入模型也是中文的）
            if lang == "en":
                search_query = translate_en_to_zh(question)
                print(f"  [检索] 英文翻译: {question[:40]} → {search_query[:40]}")
            else:
                search_query = question
            print(f"  [检索] 开始 | 语言={lang} | query={search_query[:50]}")

            # Step 1: 查询分析（含公司名称检测）
            query_analysis = self._analyze_query(search_query, "zh" if lang == "en" else lang)
            company_name = query_analysis.get("company", "")
            if company_name:
                print(f"  [检索] 检测到公司: {company_name}")

            # Step 2: 表格感知查询扩展
            expanded = self._expand_query_table_aware(search_query, "zh" if lang == "en" else lang) if self.expand_query else search_query

            # Step 2b: 图像感知查询扩展
            if self.expand_query:
                expanded = self._expand_query_image_aware(expanded, "zh" if lang == "en" else lang)

            # Step 3: 向量化查询
            query_emb = self.embedding_model.encode([expanded], is_query=True)[0]

            # Step 4: 混合检索
            raw_results = self.vector_store.search(
                query_embedding=query_emb,
                query_text=expanded,
                top_k=top_k * 2,
            )
            print(f"  [检索] 检索完成: {len(raw_results)} 条结果")

            # Step 5: 表格感知重排序
            reranked = self._rerank_table_aware(
                search_query, raw_results, "zh" if lang == "en" else lang,
                company_name=company_name,
            )

            # Step 5b: 关键词兜底检索（始终运行，与向量结果合并）
            if self.vector_store.chunks:
                keyword_results = self._keyword_fallback_search(search_query, "zh" if lang == "en" else lang)
                if keyword_results:
                    print(f"  [检索] 关键词兜底: +{len(keyword_results)} 条")
                    # 合并：去重后加入结果池（保留分数更高的版本）
                    existing_keys = {}
                    for i, r in enumerate(reranked):
                        key = (r.get("page"), r.get("text", "")[:100])
                        existing_keys[key] = i  # 记录位置
                    
                    for kr in keyword_results:
                        key = (kr.get("page"), kr.get("text", "")[:100])
                        if key in existing_keys:
                            # 重复：如果关键词分数更高，替换原结果
                            idx = existing_keys[key]
                            if kr.get("final_score", 0) > reranked[idx].get("final_score", 0):
                                reranked[idx] = kr
                        else:
                            # 新结果：直接添加
                            reranked.append(kr)
                            existing_keys[key] = len(reranked) - 1
                    reranked.sort(key=lambda x: x.get("final_score", 0), reverse=True)

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
            return {
                "results": [],
                "context_text": "",
                "total_results": 0,
                "query_analysis": {
                    "lang": detect_language(question),
                    "intent": "通用" if detect_language(question) == "zh" else "general",
                    "entities": [],
                    "keywords": [],
                    "company": "",
                    "table_type": "",
                    "error": str(e),
                },
            }

    # ============================================================
    # 查询分析（含公司名+表格类型检测）
    # ============================================================
    def _analyze_query(self, question: str, lang: str) -> Dict:
        """分析查询：意图、实体、关键词、公司名、表格类型"""
        q = question.lower()

        # 意图分类
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
        keywords = self._extract_keywords(question)

        # 公司名检测（用于多文档路由）
        company = self._detect_company(question, lang)

        # 表格类型检测（问题属于哪种表格数据查询）
        table_type = self._detect_table_type(question, lang)

        return {
            "lang": lang,
            "intent": detected_intent,
            "entities": entities,
            "keywords": keywords,
            "company": company,
            "table_type": table_type,
        }

    def _detect_company(self, text: str, lang: str) -> str:
        """检测问题中的公司名"""
        if lang == "zh":
            pattern = r'([\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限公司|集团公司|科技公司))'
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
            # 简写模式
            short = re.findall(r'(武汉[\u4e00-\u9fa5]{2,10}(?:股份|有限|电子|信息|科技))', text)
            if short:
                return short[0]
        else:
            pattern = r'([A-Z][A-Za-z\s,]{2,}(?:Inc|Corp|Ltd|Limited|Co|Company))'
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        return ""

    def _detect_table_type(self, text: str, lang: str) -> str:
        """检测问题涉及的表格类型"""
        if lang == "zh":
            # 发行相关
            if "发行股数" in text or "发行后总股本" in text or "本次发行" in text:
                return "股本结构"
            if "募集资金" in text or "投资项目" in text or "募投" in text or "补充流动资金" in text:
                return "募投项目"
            if "控制关系" in text or "关联方" in text or "持股比例" in text or "关联关系" in text:
                return "关联方"
            if "收入" in text or "营收" in text or "销售" in text:
                if "占比" in text or "比重" in text:
                    return "收入结构"
                return "收入数据"
            if "成本" in text:
                return "成本数据"
            if "技术标准" in text:
                return "技术标准"
        else:
            if "shares" in text.lower() or "offering" in text.lower() or "equity" in text.lower():
                return "股本结构"
            if "raised" in text.lower() or "proceeds" in text.lower() or "investment" in text.lower():
                return "募投项目"
            if "related" in text.lower() or "affiliate" in text.lower() or "shareholder" in text.lower():
                return "关联方"
            if "revenue" in text.lower() or "income" in text.lower():
                return "收入数据"
        return ""

    def _extract_entities(self, text: str, lang: str) -> List[str]:
        entities = []
        if lang == "zh":
            company_pattern = r'[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限公司|科技公司|集团公司)'
            companies = re.findall(company_pattern, text)
            entities.extend(companies)
            person_pattern = r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|总经|董事长)'
            persons = re.findall(person_pattern, text)
            entities.extend(persons)
            number_pattern = r'\d+(?:\.\d+)?[％%万亿元人家]'
            numbers = re.findall(number_pattern, text)
            entities.extend(numbers)
        else:
            company_pattern = r'[A-Z][A-Za-z\s,]{2,}(?:Inc|Corp|Ltd|Limited|Co|Company)'
            companies = re.findall(company_pattern, text)
            entities.extend(companies)
            amount_pattern = r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|yuan)?'
            amounts = re.findall(amount_pattern, text)
            entities.extend(amounts)
        return list(set(entities))

    def _extract_keywords(self, text: str) -> List[str]:
        zh_stopwords = set("的是了在和与有从为不也对于其以及之中上去可但而或如何什么哪个哪些多少这那")
        en_stopwords = set("the a an is are was were be been have has had do does did will would could should may might shall can must need dare ought used".split())
        zh_words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        en_words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        all_stopwords = zh_stopwords | en_stopwords
        return [w for w in zh_words + en_words if w not in all_stopwords]

    # ============================================================
    # 表格感知查询扩展（核心优化）
    # ============================================================
    def _expand_query_table_aware(self, question: str, lang: str) -> str:
        """
        增强查询扩展：同义词 + 表格类型映射。

        例如 "发行股数是多少" → "发行股数 发行股份 本次发行 发行概况 股本结构"
        这样 BM25 可以命中表格的列名和标题。
        """
        expansions = self.zh_expansions if lang == "zh" else self.en_expansions
        expanded_terms = []

        for key, synonyms in expansions.items():
            if key.lower() in question.lower():
                expanded_terms.extend(synonyms)

        # 额外表格类型扩展
        if lang == "zh":
            table_type = self._detect_table_type(question, lang)
            if table_type:
                # 添加表格类型的全部同义词
                for key, table_types in TABLE_TYPE_ZH_MAP.items():
                    if key.lower() in question.lower():
                        expanded_terms.extend(table_types)

        if expanded_terms:
            result = f"{question} {' '.join(set(expanded_terms))}"
            return result
        return question

    # ============================================================
    # 图像感知查询扩展
    # ============================================================
    def _expand_query_image_aware(self, question: str, lang: str) -> str:
        """图像感知查询扩展：当查询包含图像相关关键词时，添加图像类型扩展词"""
        if lang != "zh":
            return question

        expanded_terms = []
        for key, expansions in IMAGE_QUERY_EXPANSIONS.items():
            if key in question:
                expanded_terms.extend(expansions)

        if expanded_terms:
            return f"{question} {' '.join(set(expanded_terms))}"
        return question

    # ============================================================
    # 关键词兜底检索（向量分数太低时的补漏机制）
    # ============================================================
    def _keyword_fallback_search(self, question: str, lang: str) -> List[Dict]:
        """
        关键词兜底检索：当向量检索分数太低时，用关键词直接扫描全文chunk。
        
        触发条件: 向量最高分 < 0.5
        适用场景: 查询包含特定术语（如"销售部"、"部门构成"）但向量语义匹配弱
        """
        if lang != "zh" or not self.vector_store.chunks:
            return []

        # 提取短关键词（2-4个字的中文词，避免过长的短语）
        short_keywords = set()
        # 从问题中提取2-4字的关键词
        for match in re.findall(r'[\u4e00-\u9fa5]{2,4}', question):
            if len(match) <= 4:  # 只要短词
                short_keywords.add(match)
        
        # 特殊模式匹配：添加额外的短关键词
        pattern_keywords = {
            '销售部': ['销售部', '渠道销售', '电话', '网络销售', '大客户', '国际贸易'],
            '组织结构': ['部门', '职能', '构成', '下设'],
            '部门构成': ['部门', '下设', '构成', '设置'],
            '大客户': ['大客户', '销售处', '分公司'],
            '技术标准': ['技术标准', '国家标准', '行业标准', '军用标准'],
            '上游': ['上游', '供应商', '原材料'],
            '下游': ['下游', '客户', '行业'],
            '注册资本': ['注册资本', '注资金'],
            '法定代表人': ['法定代表人', '法人'],
            '收入': ['收入', '营收', '主营业务'],
            '关联方': ['关联方', '关联企业', '控制关系'],
        }
        
        matched_keywords = set(short_keywords)
        for pattern_key, extra_kws in pattern_keywords.items():
            if pattern_key in question:
                matched_keywords.update(extra_kws)
        
        # 过滤掉太长的关键词和停用词
        zh_stopwords = set("的是了在和与有从为不也对于其以及之中上去可但而或如何什么哪个多少这那其")
        matched_keywords = {kw for kw in matched_keywords if kw not in zh_stopwords and len(kw) <= 6}
        
        if not matched_keywords:
            return []

        # 扫描所有chunk
        results = []
        for chunk in self.vector_store.chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            # 计算关键词命中数
            hit_count = sum(1 for kw in matched_keywords if kw in text)
            if hit_count == 0:
                continue

            # 计算关键词分数：命中越多分越高，短文本命中更精准
            keyword_score = hit_count / max(len(matched_keywords), 1)
            # 短文本加权（更精准的答案往往在短chunk中）
            length_bonus = max(0, 1 - len(text) / 2000) * 0.3
            final_score = keyword_score + length_bonus

            results.append({
                "text": text,
                "page": chunk.get("page", 0),
                "type": chunk.get("type", "unknown"),
                "heading": chunk.get("heading", ""),
                "source_file": chunk.get("source_file", ""),
                "score": 0,  # 无向量分
                "keyword_overlap": keyword_score,
                "final_score": final_score,
                "table_info": chunk.get("table_info", {}),
                "image_info": chunk.get("image_info", {}),
            })

        # 按分数排序，取top-5
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:5]

    # ============================================================
    # 表格感知重排序（核心优化）
    # ============================================================
    def _rerank_table_aware(self, question: str, results: List[Dict],
                             lang: str, company_name: str = "") -> List[Dict]:
        """
        表格感知重排序：
        1. 表头列名匹配 → 高权重提升
        2. 表格标题匹配 → 最高权重提升
        3. 表格chunk类型加权（>普通文本）
        4. 公司名匹配 → 多文档路由
        5. 关键词+表格+公司综合分
        """
        if not results:
            return results

        q_words = set(self._extract_keywords(question))
        q_lower = question.lower()
        table_type = self._detect_table_type(question, lang)

        reranked = []
        for r in results:
            chunk = r.get("chunk", {})
            text = chunk.get("text", "")
            heading = chunk.get("heading", "")
            chunk_type = chunk.get("type", "unknown")
            source_file = chunk.get("source_file", "")
            table_info = chunk.get("table_info", {})
            image_info = chunk.get("image_info", {})

            # === 基础得分：向量相似度 ===
            base_score = r.get("score", 0)

            # === 关键词匹配度 ===
            text_words = set(self._extract_keywords(text + " " + heading))
            keyword_overlap = len(q_words & text_words) / max(len(q_words), 1)

            # === 表格标题匹配（最高权重） ===
            title_bonus = 0.0
            if table_info:
                title = table_info.get("title", "")
                if title:
                    title_lower = title.lower()
                    # 表格标题直接命中 → 高权重
                    for qw in q_words:
                        if qw in title_lower:
                            title_bonus += 0.5
                    # 表格类型匹配
                    if table_type and table_type in title:
                        title_bonus += 0.8

            # === 列头匹配（高权重） ===
            header_bonus = 0.0
            if table_info:
                headers = table_info.get("headers", [])
                for h in headers:
                    h_str = str(h).lower()
                    for qw in q_words:
                        if qw in h_str:
                            header_bonus += 0.4

            # === 公司名匹配（多文档路由） ===
            company_bonus = 0.0
            if company_name and source_file:
                # 提取PDF文件名中的公司关键词
                if "兴图" in source_file and "兴图" in company_name:
                    company_bonus = 0.3
                elif "力源" in source_file and "力源" in company_name:
                    company_bonus = 0.3
                elif "兴图" in source_file and ("兴图" in text or "兴图" in heading):
                    company_bonus = 0.2
                elif "力源" in source_file and ("力源" in text or "力源" in heading):
                    company_bonus = 0.2

            # === 表格类型加权 ===
            type_bonus = 0.0
            if chunk_type in ("table", "table_numeric"):
                type_bonus = 0.3  # 基础表格加权
                # 如果问题类型匹配_额外加成
                if table_type and table_info:
                    title = table_info.get("title", "")
                    if table_type in title:
                        type_bonus += 0.4

            # === 表格是否有数值数据 ===
            numeric_bonus = 0.0
            if table_info and table_info.get("has_numbers", False):
                if table_type:
                    numeric_bonus = 0.2

            # === 图像chunk加权 ===
            image_bonus = 0.0
            if chunk_type == "image":
                image_bonus = IMAGE_CONFIG.get("图像检索权重", 0.3)
                # 查询词匹配图像描述/类型额外提升
                if image_info:
                    img_type = image_info.get("image_type", "")
                    caption = image_info.get("caption_text", "")
                    for qw in q_words:
                        if qw in img_type.lower() or qw in caption.lower():
                            image_bonus += IMAGE_CONFIG.get("图像指令命中提升", 0.4)
                            break

            # === 综合得分 ===
            final_score = (
                base_score * (1 + keyword_overlap) +
                title_bonus +
                header_bonus +
                type_bonus +
                company_bonus +
                numeric_bonus +
                image_bonus
            )

            reranked.append({
                "text": text,
                "page": chunk.get("page", 0),
                "type": chunk_type,
                "heading": heading,
                "source_file": source_file,
                "score": base_score,
                "keyword_overlap": keyword_overlap,
                "final_score": final_score,
                "table_info": table_info,
                "image_info": image_info,
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    # ============================================================
    # 上下文构建
    # ============================================================
    def _build_context(self, results: List[Dict], query_analysis: Dict) -> str:
        """构建LLM上下文（含表格提示）"""
        parts = []
        seen_tables = set()

        for i, r in enumerate(results):
            text = r["text"]
            page = r["page"]
            heading = r.get("heading", "")
            chunk_type = r.get("type", "unknown")
            source_file = r.get("source_file", "")
            table_info = r.get("table_info", {})

            if query_analysis.get("lang", "zh") == "zh":
                # 来源信息
                src = f"第{page}页" if isinstance(page, (int, str)) else f"第{page}页"
                if source_file:
                    src += f" ({source_file})"

                header = f"[片段 {i+1}] {src}"
                if heading:
                    header += f" | 标题: {heading}"

                # 表格特殊标记
                if chunk_type in ("table", "table_numeric"):
                    title = table_info.get("title", "")
                    if title:
                        header += f" | 📊表格: {title}"

                    # 避免重复表格内容
                    table_key = f"{page}_{title}_{table_info.get('headers', [])}"
                    if table_key in seen_tables:
                        continue
                    seen_tables.add(table_key)
            else:
                src = f"Page {page}"
                if source_file:
                    src += f" ({source_file})"
                header = f"[Chunk {i+1}] {src}"
                if heading:
                    header += f" | Heading: {heading}"
                if chunk_type in ("table", "table_numeric"):
                    title = table_info.get("title", "")
                    if title:
                        header += f" | 📊Table: {title}"

            parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(parts)

    def format_context_for_llm(self, context: str, max_chars: int = 4000) -> str:
        """截断上下文以适应LLM token限制"""
        if len(context) <= max_chars:
            return context
        return context[:max_chars] + "\n... [上下文已截断/Context truncated]"
