"""
工单编号: 人工智能NLP-RAG-混合检索任务
检索器 - 支持向量检索/全文检索/混合检索三种模式 + 多重排器
"""
import re
from typing import List, Dict, Optional

try:
    from config import IMAGE_CONFIG
except ImportError:
    IMAGE_CONFIG = {}


def detect_language(text: str) -> str:
    """检测查询语言"""
    if not text:
        return "zh"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_chars = len(re.findall(r'[a-zA-Z]', text))
    total = max(chinese_chars + ascii_chars, 1)
    return "zh" if chinese_chars / total >= 0.3 else "en"


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

TABLE_TYPE_EN_MAP = {}


class Retriever:
    """
    检索器 - 支持三种检索模式（工单6核心升级）

    检索模式：
    1. vector  — 纯向量检索（Milvus + BM25混合）
    2. fulltext — 纯全文检索（Whoosh倒排索引）
    3. hybrid   — 混合检索（向量+全文融合）

    重排（可选）：
    - tfidf / llm / adaptive / crossencoder
    """

    def __init__(self, vector_store=None, embedding_model=None,
                 fulltext_store=None, reranker=None,
                 expand_query: bool = True, rerank_top_k: int = 5,
                 hybrid_config: dict = None,
                 retrieval_mode: str = "hybrid"):
        """
        Args:
            vector_store: 向量存储实例（Milvus/FAISS）
            embedding_model: 嵌入模型实例
            fulltext_store: 全文检索实例（Whoosh FullTextStore）
            reranker: 重排器实例
            expand_query: 是否查询扩展
            rerank_top_k: 重排返回数量
            hybrid_config: 混合检索配置（权重/融合算法）
            retrieval_mode: 默认检索模式（vector/fulltext/hybrid）
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.fulltext_store = fulltext_store
        self.reranker = reranker
        self.expand_query = expand_query
        self.rerank_top_k = rerank_top_k
        self.retrieval_mode = retrieval_mode
        self.hybrid_config = hybrid_config or {}

        # 重排器（可外部设置）
        self._reranker = reranker

        # 中文同义词扩展表
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
            "组织结构图": ["组织结构图", "组织架构", "销售部", "大客户销售部", "部门构成",
                         "内部组织结构", "职能部门", "公司治理结构", "股东大会", "董事会"],
            "销售部": ["销售部", "销售部门", "大客户销售部", "渠道销售部",
                      "电话及网络销售部", "国际贸易部", "销售架构", "销售处"],
            "大客户销售部": ["大客户销售部", "大客户销售", "大客户", "销售处", "分公司", "销售处经理"],
            "IC市场": ["IC市场", "集成电路市场", "半导体市场", "应用结构",
                      "增长率", "赛迪顾问", "2008年", "应用行业", "工业控制"],
            "应用结构与增长": ["应用结构与增长", "市场应用结构", "增长率", "增长图", "IC市场应用", "中国IC市场"],
        }

    def set_retrieval_mode(self, mode: str):
        """切换检索模式"""
        if mode in ("vector", "fulltext", "hybrid"):
            self.retrieval_mode = mode
            print(f"[Retriever] 检索模式切换为: {mode}")

    def set_reranker(self, reranker):
        """设置重排器"""
        self._reranker = reranker

    # ============================================================
    # 主检索入口
    # ============================================================
    def retrieve(self, question: str, top_k: int = 5,
                 threshold: float = 0.0, mode: str = None) -> Dict:
        """
        检索主入口（支持三种模式）

        Args:
            question: 用户问题
            top_k: 返回数量
            threshold: 相关性阈值
            mode: 检索模式（覆盖默认）: vector/fulltext/hybrid

        Returns:
            Dict: {results, context_text, total_results, query_analysis}
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
            print(f"  [检索] 开始 | 语言={lang} | 模式={mode or self.retrieval_mode} | query={search_query[:50]}")
            query_analysis = self._analyze_query(search_query, "zh" if lang == "en" else lang)
            company_name = query_analysis.get("company", "")
            if company_name:
                print(f"  [检索] 检测到公司: {company_name}")

            # 查询扩展
            expanded = self._expand_query(search_query, "zh" if lang == "en" else lang)
            if expanded != search_query:
                print(f"  [检索] 查询扩展: → {expanded[:50]}")

            # 选择检索模式
            mode = mode or self.retrieval_mode

            if mode == "fulltext":
                results = self._retrieve_fulltext(expanded, top_k)
            elif mode == "hybrid":
                results = self._retrieve_hybrid(expanded, "zh" if lang == "en" else lang, top_k, threshold)
            else:  # vector (default)
                results = self._retrieve_vector(expanded, "zh" if lang == "en" else lang, top_k, threshold)
            print(f"  [检索] 检索完成: {len(results)} 条结果")

            # 重排
            if self._reranker and results:
                results = self._reranker.rerank(expanded, results, top_k=self.rerank_top_k or top_k)
                print(f"  [检索] 重排完成: {len(results)} 条")

            # 表格感知重排序（基础版）
            results = self._rerank_table_aware(search_query, results, "zh" if lang == "en" else lang, company_name=company_name)

            # 关键词兜底
            if self.vector_store and hasattr(self.vector_store, 'chunks') and self.vector_store.chunks:
                keyword_results = self._keyword_fallback_search(search_query, "zh" if lang == "en" else lang)
                if keyword_results:
                    print(f"  [检索] 关键词兜底: +{len(keyword_results)} 条")
                    results = self._merge_results(results, keyword_results)

            # 过滤低分
            filtered = [r for r in results if r.get("final_score", 0) >= threshold]
            if not filtered:
                filtered = results[:top_k]
            final_results = filtered[:top_k]

            context_text = self._build_context(final_results, query_analysis)
            elapsed = time.time() - t0
            print(f"  [检索] 完成 | {len(final_results)} 条 | 耗时 {elapsed:.2f}s")

            return {
                "results": final_results,
                "context_text": context_text,
                "total_results": len(final_results),
                "query_analysis": query_analysis,
                "retrieval_mode": mode,
            }
        except Exception as e:
            import traceback
            print(f"  [检索] ❌ 异常: {e}")
            traceback.print_exc()
            return {
                "results": [],
                "context_text": "",
                "total_results": 0,
                "query_analysis": {
                    "lang": detect_language(question),
                    "intent": "通用" if detect_language(question) == "zh" else "general",
                    "entities": [], "keywords": [], "company": "",
                    "table_type": "", "error": str(e),
                },
                "retrieval_mode": mode if mode else self.retrieval_mode,
            }

    # ============================================================
    # 三种检索实现
    # ============================================================

    def _retrieve_vector(self, query: str, lang: str, top_k: int, threshold: float) -> List[Dict]:
        """向量检索"""
        if not self.vector_store or not self.embedding_model:
            return []
        query_emb = self.embedding_model.encode([query], is_query=True)[0]
        raw_results = self.vector_store.search(
            query_embedding=query_emb, query_text=query,
            top_k=top_k * 2,
        )
        # 统一格式为 flat dict（兼容后续处理）
        return self._normalize_results(raw_results)

    def _retrieve_fulltext(self, query: str, top_k: int) -> List[Dict]:
        """全文检索（Whoosh）"""
        if not self.fulltext_store:
            return []
        raw = self.fulltext_store.search(query, top_k=top_k * 2)
        results = []
        for r in raw:
            results.append({
                "text": r.get("text", ""),
                "page": r.get("page", 0),
                "type": r.get("type", "text"),
                "heading": r.get("heading", ""),
                "source_file": r.get("source_file", ""),
                "score": r.get("fulltext_score", 0),
                "final_score": r.get("fulltext_score", 0),
                "chunk_id": r.get("chunk_id", ""),
                "_source": "fulltext",
            })
        return results

    def _retrieve_hybrid(self, query: str, lang: str, top_k: int, threshold: float) -> List[Dict]:
        """
        混合检索：向量 + 全文 融合

        融合算法（配置）:
        - weighted_avg: 加权平均
        - rrf: Reciprocal Rank Fusion
        - vote: 投票机制
        """
        fusion_algo = self.hybrid_config.get("融合算法", "rrf")
        vector_weight = self.hybrid_config.get("向量权重", 0.5)
        fulltext_weight = self.hybrid_config.get("全文权重", 0.5)
        rrf_k = self.hybrid_config.get("rrf常数", 60)

        # 1. 向量检索
        vector_results = []
        if self.vector_store and self.embedding_model:
            query_emb = self.embedding_model.encode([query], is_query=True)[0]
            vec_raw = self.vector_store.search(
                query_embedding=query_emb, query_text=query,
                top_k=top_k * 3,
            )
            vector_results = self._normalize_results(vec_raw)

        # 2. 全文检索
        fulltext_results = []
        if self.fulltext_store:
            ft_raw = self.fulltext_store.search(query, top_k=top_k * 3)
            for i, r in enumerate(ft_raw):
                fulltext_results.append({
                    "text": r.get("text", ""),
                    "page": r.get("page", 0),
                    "type": r.get("type", "text"),
                    "heading": r.get("heading", ""),
                    "source_file": r.get("source_file", ""),
                    "score": float(r.get("fulltext_score", 0)),
                    "final_score": float(r.get("fulltext_score", 0)),
                    "chunk_id": r.get("chunk_id", i),
                    "_source": "fulltext",
                })

        # 3. 融合
        if not vector_results and not fulltext_results:
            return []
        if not fulltext_results:
            return vector_results[:top_k]
        if not vector_results:
            return fulltext_results[:top_k]

        if fusion_algo == "weighted_avg":
            return self._fuse_weighted_avg(vector_results, fulltext_results,
                                           vector_weight, fulltext_weight, top_k)
        elif fusion_algo == "vote":
            return self._fuse_vote(vector_results, fulltext_results, top_k)
        else:  # rrf
            return self._fuse_rrf(vector_results, fulltext_results, rrf_k, top_k)

    def _fuse_rrf(self, vec_results: List[Dict], ft_results: List[Dict],
                  k: int = 60, top_k: int = 5) -> List[Dict]:
        """RRF融合"""
        rrf_scores = {}
        results_map = {}

        def get_key(r):
            return hash(r.get("text", "")[:200])

        for rank, r in enumerate(vec_results):
            key = get_key(r)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
            results_map[key] = r

        for rank, r in enumerate(ft_results):
            key = get_key(r)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in results_map:
                results_map[key] = r

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            r = dict(results_map[key])
            r["rrf_score"] = rrf_scores[key]
            r["final_score"] = rrf_scores[key]
            r["_source"] = "hybrid_rrf"
            results.append(r)
        return results

    def _fuse_weighted_avg(self, vec_results, ft_results,
                           vec_w, ft_w, top_k) -> List[Dict]:
        """加权平均融合"""
        # 归一化分数
        all_results = {}
        if vec_results:
            max_v = max(r.get("score", 0) for r in vec_results) or 1.0
            for r in vec_results:
                r["_norm_vec_score"] = r.get("score", 0) / max_v

        if ft_results:
            max_f = max(r.get("score", 0) for r in ft_results) or 1.0
            for r in ft_results:
                r["_norm_ft_score"] = r.get("score", 0) / max_f

        # 合并
        def get_key(r):
            return hash(r.get("text", "")[:200])

        for r in vec_results:
            key = get_key(r)
            all_results[key] = r
            all_results[key]["_norm_ft_score"] = 0.0
            all_results[key]["_vec_rank"] = True

        for r in ft_results:
            key = get_key(r)
            if key in all_results:
                all_results[key]["_norm_ft_score"] = r.get("_norm_ft_score", 0)
            else:
                r["_norm_vec_score"] = 0.0
                r["_vec_rank"] = False
                all_results[key] = r

        results = []
        for key, r in all_results.items():
            r["final_score"] = vec_w * r.get("_norm_vec_score", 0) + ft_w * r.get("_norm_ft_score", 0)
            r["_source"] = "hybrid_weighted"
            results.append(r)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def _fuse_vote(self, vec_results, ft_results, top_k) -> List[Dict]:
        """投票机制融合"""
        def get_key(r):
            return hash(r.get("text", "")[:200])

        vote_counts = {}
        results_map = {}
        for r in vec_results:
            key = get_key(r)
            vote_counts[key] = vote_counts.get(key, 0) + 1
            results_map[key] = r
        for r in ft_results:
            key = get_key(r)
            vote_counts[key] = vote_counts.get(key, 0) + 1
            if key not in results_map:
                results_map[key] = r

        sorted_keys = sorted(vote_counts.keys(), key=lambda x: vote_counts[x], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            r = dict(results_map[key])
            r["vote_count"] = vote_counts[key]
            r["final_score"] = vote_counts[key]
            r["_source"] = "hybrid_vote"
            results.append(r)
        return results

    def _normalize_results(self, raw: List[Dict]) -> List[Dict]:
        """统一结果格式为 flat dict"""
        normalized = []
        for r in raw:
            chunk = r.get("chunk", {})
            normalized.append({
                "text": chunk.get("text", r.get("text", "")),
                "page": chunk.get("page", r.get("page", 0)),
                "type": chunk.get("type", r.get("type", "text")),
                "heading": chunk.get("heading", r.get("heading", "")),
                "source_file": chunk.get("source_file", r.get("source_file", "")),
                "score": float(r.get("score", 0)),
                "final_score": float(r.get("score", 0)),
                "chunk_id": chunk.get("chunk_id", r.get("chunk_id", -1)),
                "table_info": chunk.get("table_info", r.get("table_info", {})),
                "image_info": chunk.get("image_info", r.get("image_info", {})),
                "_source": "vector",
            })
        return normalized

    # ============================================================
    # 查询扩展
    # ============================================================
    def _expand_query(self, question: str, lang: str) -> str:
        """查询扩展：同义词 + 表格类型 + 图像感知"""
        expanded = question
        if self.expand_query:
            expanded = self._expand_query_table_aware(expanded, lang)
            expanded = self._expand_query_image_aware(expanded, lang)
        return expanded

    def _expand_query_table_aware(self, question: str, lang: str) -> str:
        if lang != "zh":
            return question
        expansions = self.zh_expansions
        expanded_terms = []
        for key, synonyms in expansions.items():
            if key.lower() in question.lower():
                expanded_terms.extend(synonyms)
        if lang == "zh":
            table_type = self._detect_table_type(question, lang)
            if table_type:
                for key, table_types in TABLE_TYPE_ZH_MAP.items():
                    if key.lower() in question.lower():
                        expanded_terms.extend(table_types)
        if expanded_terms:
            return f"{question} {' '.join(set(expanded_terms))}"
        return question

    def _expand_query_image_aware(self, question: str, lang: str) -> str:
        return question

    # ============================================================
    # 查询分析
    # ============================================================
    def _analyze_query(self, question: str, lang: str) -> Dict:
        q = question.lower()
        intents = {}
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
        return {
            "lang": lang,
            "intent": detected_intent,
            "entities": self._extract_entities(question, lang),
            "keywords": self._extract_keywords(question),
            "company": self._detect_company(question, lang),
            "table_type": self._detect_table_type(question, lang),
        }

    def _detect_company(self, text: str, lang: str) -> str:
        if lang == "zh":
            pattern = r'([\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限公司|集团公司|科技公司))'
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
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
        if lang == "zh":
            if "发行股数" in text or "发行后总股本" in text or "本次发行" in text:
                return "股本结构"
            if "募集资金" in text or "投资项目" in text or "募投" in text or "补充流动资金" in text:
                return "募投项目"
            if "控制关系" in text or "关联方" in text or "持股比例" in text or "关联关系" in text:
                return "关联方"
            if "收入" in text or "营收" in text or "销售" in text:
                return "收入结构" if ("占比" in text or "比重" in text) else "收入数据"
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
            companies = re.findall(r'[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限公司|科技公司|集团公司)', text)
            entities.extend(companies)
            persons = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|总经|董事长)', text)
            entities.extend(persons)
            numbers = re.findall(r'\d+(?:\.\d+)?[％%万亿元人家]', text)
            entities.extend(numbers)
        else:
            companies = re.findall(r'[A-Z][A-Za-z\s,]{2,}(?:Inc|Corp|Ltd|Limited|Co|Company)', text)
            entities.extend(companies)
            amounts = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|yuan)?', text)
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
    # 关键词兜底
    # ============================================================
    def _keyword_fallback_search(self, question: str, lang: str) -> List[Dict]:
        """关键词兜底检索"""
        if lang != "zh" or not self.vector_store or not hasattr(self.vector_store, 'chunks'):
            return []
        if not self.vector_store.chunks:
            return []

        short_keywords = set()
        for match in re.findall(r'[\u4e00-\u9fa5]{2,4}', question):
            if len(match) <= 4:
                short_keywords.add(match)

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

        zh_stopwords = set("的是了在和与有从为不也对于其以及之中上去可但而或如何什么哪个多少这那其")
        matched_keywords = {kw for kw in matched_keywords if kw not in zh_stopwords and len(kw) <= 6}
        if not matched_keywords:
            return []

        results = []
        for chunk in self.vector_store.chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            hit_count = sum(1 for kw in matched_keywords if kw in text)
            if hit_count == 0:
                continue
            keyword_score = hit_count / max(len(matched_keywords), 1)
            length_bonus = max(0, 1 - len(text) / 2000) * 0.3
            final_score = keyword_score + length_bonus
            results.append({
                "text": text,
                "page": chunk.get("page", 0),
                "type": chunk.get("type", "unknown"),
                "heading": chunk.get("heading", ""),
                "source_file": chunk.get("source_file", ""),
                "score": 0,
                "keyword_overlap": keyword_score,
                "final_score": final_score,
                "table_info": chunk.get("table_info", {}),
                "image_info": chunk.get("image_info", {}),
                "_source": "keyword_fallback",
            })
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:5]

    def _merge_results(self, primary: List[Dict], supplement: List[Dict]) -> List[Dict]:
        """合并检索结果（去重保留高分）"""
        seen = {}
        for r in primary:
            key = (r.get("page"), r.get("text", "")[:100])
            seen[key] = r
        for r in supplement:
            key = (r.get("page"), r.get("text", "")[:100])
            if key in seen:
                if r.get("final_score", 0) > seen[key].get("final_score", 0):
                    seen[key] = r
            else:
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x.get("final_score", 0), reverse=True)

    # ============================================================
    # 表格感知重排序
    # ============================================================
    def _rerank_table_aware(self, question: str, results: List[Dict],
                             lang: str, company_name: str = "") -> List[Dict]:
        if not results:
            return results
        q_words = set(self._extract_keywords(question))
        q_lower = question.lower()
        table_type = self._detect_table_type(question, lang)

        reranked = []
        for r in results:
            text = r.get("text", "")
            heading = r.get("heading", "")
            chunk_type = r.get("type", "unknown")
            source_file = r.get("source_file", "")
            table_info = r.get("table_info", {})
            image_info = r.get("image_info", {})

            # 基础分
            base_score = r.get("final_score", r.get("score", 0))
            boost = 0.0

            # 表格chunk加分
            if chunk_type == "table":
                boost += 0.3
            # 图像chunk加分
            if chunk_type == "image":
                boost += 0.3
            # 列名匹配
            if table_info:
                col_names = table_info.get("column_names", [])
                if any(q_word in " ".join(col_names) for q_word in q_words):
                    boost += 0.4
            # 标题匹配
            if heading and any(q_word in heading for q_word in q_words):
                boost += 0.5
            # 公司名匹配
            if company_name and source_file and company_name in source_file:
                boost += 0.2

            result = dict(r)
            result["final_score"] = base_score + boost
            reranked.append(result)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    # ============================================================
    # 上下文构建
    # ============================================================
    def _build_context(self, results: List[Dict], query_analysis: Dict) -> str:
        """构建供LLM生成的上下文"""
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results):
            text = r.get("text", "")
            if not text:
                continue
            heading = r.get("heading", "")
            page = r.get("page", 0)
            source = r.get("source_file", "")
            score = r.get("final_score", 0)
            chunk_type = r.get("type", "text")

            header = f"[片段{i+1}]"
            if heading:
                header += f" 标题: {heading}"
            if page:
                header += f" (第{page}页)"
            if source:
                header += f" [{source}]"
            if chunk_type:
                header += f" 类型:{chunk_type}"

            parts.append(f"{header}\n{text.strip()[:800]}")

        return "\n\n---\n\n".join(parts)

    def format_context_for_llm(self, context: str, max_chars: int = 6000) -> str:
        """截断上下文并添加格式化提示"""
        if not context:
            return ""
        truncated = context[:max_chars]
        return (
            "以下是来自招股说明书的参考内容，请据此回答用户的问题。\n"
            "注意：部分内容可能包含表格数据，请仔细阅读。\n\n"
            f"{truncated}"
        )
