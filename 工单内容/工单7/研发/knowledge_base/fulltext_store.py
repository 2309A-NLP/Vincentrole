"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
全文检索存储 - Whoosh倒排索引
"""

import os
import json
from typing import List, Dict, Optional

from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import MultifieldParser, OrGroup, AndGroup
from whoosh.query import Term, And, Or, Not, Phrase, FuzzyTerm
from whoosh.analysis import StandardAnalyzer, Tokenizer, LowercaseFilter, Token
from whoosh.writing import AsyncWriter
import jieba
import re


class JiebaTokenizer(Tokenizer):
    """基于jieba的中文分词器"""
    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, mode='',
                 **kwargs):
        assert isinstance(value, str), f"Expected str, got {type(value)}"
        pos = start_pos
        for word in jieba.cut(value, cut_all=False):
            if len(word.strip()) == 0:
                continue
            # 只保留中文词(1字起)和英文词(2字母起)
            if re.match(r'^[\u4e00-\u9fff]+$', word) or \
               re.match(r'^[a-zA-Z]{2,}$', word):
                token = Token()
                token.text = word.lower()
                if positions:
                    token.pos = pos
                    pos += 1
                if chars:
                    token.startchar = start_char
                    token.endchar = start_char + len(word)
                    start_char += len(word)
                yield token

ChineseAnalyzer = JiebaTokenizer() | LowercaseFilter()


class FullTextStore:
    """
    Whoosh全文检索存储

    提供基于倒排索引的多字段搜索能力，支持：
    - 多字段检索（text, heading, source_file）
    - 布尔查询（AND, OR, NOT）
    - 短语匹配（双引号）
    - 模糊匹配（~ 后缀）
    - 索引构建与增量更新

    Schema:
        chunk_id (STORED, 唯一标识)
        text (TEXT, 全文索引字段)
        heading (TEXT, 标题索引字段)
        source_file (TEXT, 来源文件索引字段)
        page (STORED, 页码)
        type (STORED, 类型)
    """

    SCHEMA = Schema(
        chunk_id=ID(stored=True, unique=True),
        text=TEXT(stored=True, analyzer=ChineseAnalyzer),
        heading=TEXT(stored=True, analyzer=ChineseAnalyzer),
        source_file=TEXT(stored=True, analyzer=StandardAnalyzer()),
        page=STORED,
        type=STORED,
    )

    def __init__(self, index_dir: str):
        """
        初始化全文检索存储

        Args:
            index_dir: Whoosh索引目录路径，不存在时自动创建
        """
        self.index_dir = index_dir
        self._ensure_dir()
        self.index = self._open_or_create_index()

    def _ensure_dir(self):
        """确保索引目录存在"""
        os.makedirs(self.index_dir, exist_ok=True)

    def _open_or_create_index(self):
        """打开已有索引，或创建新索引"""
        if exists_in(self.index_dir):
            print(f"[FullTextStore] 打开已有索引: {self.index_dir}")
            return open_dir(self.index_dir)
        else:
            print(f"[FullTextStore] 创建新索引: {self.index_dir}")
            return create_in(self.index_dir, self.SCHEMA)

    # ------------------------------------------------------------------ #
    # 索引构建
    # ------------------------------------------------------------------ #

    def build_index(self, chunks: List[Dict]):
        """
        从chunk列表构建索引（增量更新）

        如果chunk_id已存在则更新，不存在则新增。

        Args:
            chunks: 文档块列表，每个元素为字典，需包含以下字段:
                - chunk_id: 唯一标识（int或str）
                - text: 文本内容（str）
                - heading: 标题（str，可选）
                - source_file: 来源文件名（str，可选）
                - page: 页码（int，可选）
                - type: 类型（str，可选）

        Raises:
            ValueError: 如果chunks为空或缺少必要字段
        """
        if not chunks:
            print("[FullTextStore] 警告: chunks为空，跳过索引构建")
            return

        count = 0
        writer = AsyncWriter(self.index)

        for chunk in chunks:
            if not isinstance(chunk, dict):
                print(f"[FullTextStore] 跳过非字典元素: {type(chunk)}")
                continue

            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                print("[FullTextStore] 跳过缺少chunk_id的条目")
                continue

            text = chunk.get("text", "") or ""
            heading = chunk.get("heading", "") or ""
            source_file = chunk.get("source_file", "") or ""
            page = chunk.get("page")
            ctype = chunk.get("type", "")

            writer.update_document(
                chunk_id=chunk_id,
                text=text,
                heading=heading,
                source_file=source_file,
                page=page,
                type=ctype,
            )
            count += 1

        writer.commit()
        print(f"[FullTextStore] 索引构建完成: 共处理 {count} 个文档块")

    # ------------------------------------------------------------------ #
    # 搜索
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_boolean: bool = True,
        use_phrase: bool = True,
        use_fuzzy: bool = True,
    ) -> List[Dict]:
        """
        执行全文检索

        自动检测查询类型:
        - 包含AND/OR/NOT → 布尔查询（当use_boolean=True）
        - 包含双引号 → 短语匹配（当use_phrase=True）
        - 包含~后缀 → 模糊匹配（当use_fuzzy=True）
        - 默认 → 多字段普通检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量上限
            use_boolean: 是否启用布尔查询解析
            use_phrase: 是否启用短语匹配
            use_fuzzy: 是否启用模糊匹配

        Returns:
            List[Dict]: 检索结果列表，每个元素包含:
                - chunk_id
                - text
                - page
                - type
                - heading
                - source_file
                - fulltext_score
        """
        if not query or not query.strip():
            return []

        results_list = []

        with self.index.searcher() as searcher:
            q = self._parse_query(query, use_boolean, use_phrase, use_fuzzy)

            if q is None:
                return []

            whoosh_results = searcher.search(q, limit=top_k)

            for hit in whoosh_results:
                results_list.append({
                    "chunk_id": hit.get("chunk_id"),
                    "text": hit.get("text", ""),
                    "page": hit.get("page"),
                    "type": hit.get("type", ""),
                    "heading": hit.get("heading", ""),
                    "source_file": hit.get("source_file", ""),
                    "fulltext_score": hit.score,
                })

        return results_list

    def _parse_query(
        self,
        query: str,
        use_boolean: bool,
        use_phrase: bool,
        use_fuzzy: bool,
    ):
        """
        解析查询字符串，构造Whoosh查询对象

        策略优先级:
        1. 布尔查询（use_boolean=True 且查询含 AND/OR/NOT）
        2. 短语匹配（use_phrase=True 且查询含双引号）
        3. 模糊匹配（use_fuzzy=True 且查询含~）
        4. 默认多字段搜索
        """
        query = query.strip()

        # --- 策略1: 布尔查询 ---
        if use_boolean and self._contains_boolean_op(query):
            return self._build_boolean_query(query)

        # --- 策略2: 短语匹配 ---
        if use_phrase and '"' in query:
            return self._build_phrase_query(query)

        # --- 策略3: 模糊匹配 ---
        if use_fuzzy and "~" in query:
            return self._build_fuzzy_query(query)

        # --- 策略4: 默认多字段搜索 ---
        return self._build_default_query(query)

    @staticmethod
    def _contains_boolean_op(query: str) -> bool:
        """检查查询字符串是否包含布尔操作符"""
        tokens = query.upper().split()
        return any(op in tokens for op in ("AND", "OR", "NOT"))

    def _build_boolean_query(self, query: str) -> Optional["whoosh.query.Query"]:
        """
        构建布尔查询

        支持 AND / OR / NOT 操作符。
        分词后用 MultifieldParser 解析每个子查询，
        然后手动组合为 And / Or / Not 节点。
        """
        tokens = self._tokenize_boolean(query)
        if not tokens:
            return None

        return self._assemble_boolean(tokens)

    @staticmethod
    def _tokenize_boolean(query: str) -> List:
        """
        将布尔查询字符串拆分为token序列

        返回格式: [(type, value), ...]
            type: "term" | "op"
            term: 普通搜索词
            op: AND | OR | NOT
        """
        parts = query.split()
        tokens = []
        for part in parts:
            upper = part.upper()
            if upper in ("AND", "OR", "NOT"):
                tokens.append(("op", upper))
            else:
                tokens.append(("term", part))
        return tokens

    def _assemble_boolean(self, tokens: List) -> Optional["whoosh.query.Query"]:
        """
        根据token序列组装布尔查询树

        支持的操作:
        - NOT term → Not(Term("text", term))
        - a AND b → And([...])
        - a OR b  → Or([...])
        """
        # 处理 NOT: 将 "NOT term" 合并为一个带 Not 的 term
        processed = []
        i = 0
        while i < len(tokens):
            ttype, tval = tokens[i]
            if ttype == "op" and tval == "NOT":
                if i + 1 < len(tokens) and tokens[i + 1][0] == "term":
                    processed.append(("not_term", tokens[i + 1][1]))
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            processed.append((ttype, tval))
            i += 1

        # 提取所有 term 和 not_term 作为叶子节点
        terms = []
        not_terms = []
        for ttype, tval in processed:
            if ttype == "term":
                terms.append(self._multifield_term(tval))
            elif ttype == "not_term":
                not_terms.append(self._multifield_term(tval))

        # 确定连接词（AND 或 OR），默认 AND
        join_op = "AND"
        for ttype, tval in processed:
            if ttype == "op":
                join_op = tval
                break

        if not terms and not not_terms:
            return None

        # 组合正项
        if join_op == "OR":
            combined = Or(terms) if terms else None
        else:
            combined = And(terms) if terms else None

        # 附加 NOT 项
        for nt in not_terms:
            combined = And([combined, Not(nt)]) if combined else Not(nt)

        return combined

    @staticmethod
    def _multifield_term(term_str: str) -> "whoosh.query.Query":
        """在多个字段上创建 Term 查询（OR 组合）"""
        from whoosh.query import Or as QOr

        field_clauses = []
        for field in ("text", "heading", "source_file"):
            field_clauses.append(Term(field, term_str))
        return QOr(field_clauses)

    def _build_phrase_query(self, query: str):
        """
        构建短语匹配查询

        提取双引号内的内容作为短语，在text字段上做Phrase匹配。
        双引号外的词作为普通查询补充。
        """
        import re

        phrases = re.findall(r'"([^"]+)"', query)
        remainder = re.sub(r'"[^"]+"', "", query).strip()

        from whoosh.query import And as QAnd

        clause_list = []

        for phrase in phrases:
            clause_list.append(Phrase("text", phrase.split()))

        if remainder:
            clause_list.append(
                self._build_default_query(remainder)
            )

        if not clause_list:
            return None
        if len(clause_list) == 1:
            return clause_list[0]
        return QAnd(clause_list)

    def _build_fuzzy_query(self, query: str):
        """
        构建模糊匹配查询

        对每个词条尝试 ~N 后缀解析（默认模糊度2），
        在多个字段上构建 FuzzyTerm，OR 组合。
        """
        import re

        terms = query.split()
        fuzzy_clauses = []

        for term in terms:
            m = re.match(r"(.+)~(\d*)$", term)
            if m:
                word = m.group(1)
                maxdist = int(m.group(2)) if m.group(2) else 2
            else:
                word = term
                maxdist = 2

            field_clauses = []
            for field in ("text", "heading", "source_file"):
                field_clauses.append(FuzzyTerm(field, word, maxdist=maxdist))

            fuzzy_clauses.append(Or(field_clauses))

        if not fuzzy_clauses:
            return None
        if len(fuzzy_clauses) == 1:
            return fuzzy_clauses[0]
        return And(fuzzy_clauses)

    def _build_default_query(self, query: str):
        """默认多字段搜索"""
        parser = MultifieldParser(
            ["text", "heading", "source_file"],
            schema=self.SCHEMA,
            group=OrGroup,
        )
        return parser.parse(query)
