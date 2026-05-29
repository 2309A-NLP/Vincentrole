# -*- coding: utf-8 -*-
"""
混合检索工具

提供轻量级关键词索引、BM25 召回、查询变体生成和多路召回结果融合。
"""

import json
import logging
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import settings


logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_PAGE_MARK_PATTERN = re.compile(r"【第 \d+ 页】")


def tokenize_for_retrieval(text: str) -> List[str]:
    """
    面向中文和中英混合文本的保守分词。

    不依赖 jieba：中文用单字 + 相邻 bigram，英文/数字按词切分。
    """
    raw_tokens = _TOKEN_PATTERN.findall((text or "").lower())
    tokens: List[str] = []
    chinese_buffer: List[str] = []

    def flush_chinese_buffer() -> None:
        if not chinese_buffer:
            return
        # 中文没有天然空格分词；单字保证召回率，bigram 提升短语匹配质量。
        tokens.extend(chinese_buffer)
        tokens.extend(
            "".join(chinese_buffer[index:index + 2])
            for index in range(len(chinese_buffer) - 1)
        )
        chinese_buffer.clear()

    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            chinese_buffer.append(token)
            continue

        flush_chinese_buffer()
        tokens.append(token)

    flush_chinese_buffer()
    return [token for token in tokens if token.strip()]


def build_query_variants(query: str) -> List[str]:
    """
    生成多路稠密召回的查询变体。
    """
    normalized = " ".join((query or "").split())
    without_page_marks = _PAGE_MARK_PATTERN.sub(" ", normalized)
    without_punctuation = re.sub(r"[^\w\u4e00-\u9fff]+", " ", without_page_marks)
    # 关键词式 query 对长句、PDF 页码和标点噪声更不敏感。
    keyword_query = " ".join(tokenize_for_retrieval(without_punctuation)[:24])

    variants = []
    for item in (normalized, without_punctuation.strip(), keyword_query.strip()):
        if item and item not in variants:
            variants.append(item)
    logger.debug(
        "Query variants built: input_length=%s variants=%s",
        len(query or ""),
        len(variants or [query]),
    )
    return variants or [query]


def merge_retrieval_results(
    result_groups: Iterable[Iterable[Dict]],
    limit: int = 20,
) -> List[Dict]:
    """
    使用简化 RRF 融合多路召回结果，并按文本去重。
    """
    merged: Dict[str, Dict] = {}
    rrf_k = 60.0
    total_items = 0
    group_count = 0

    for group in result_groups:
        group_count += 1
        for rank, item in enumerate(group or [], start=1):
            total_items += 1
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            meta = str(item.get("meta", ""))
            key = f"{meta}\n{text}"
            route = item.get("route", "unknown")
            original_score = float(item.get("score", 0.0))
            # RRF 只依赖各路召回排名，适合融合 BM25 分数和向量相似度这种不同量纲的分数。
            fused_score = 1.0 / (rrf_k + rank)

            if key not in merged:
                merged[key] = {
                    "text": text,
                    "meta": meta,
                    "score": original_score,
                    "fusion_score": fused_score,
                    "routes": {route},
                }
            else:
                merged[key]["score"] = max(merged[key]["score"], original_score)
                merged[key]["fusion_score"] += fused_score
                merged[key]["routes"].add(route)

    ranked = sorted(
        merged.values(),
        key=lambda item: (item["fusion_score"], item["score"]),
        reverse=True,
    )

    for item in ranked:
        item["route"] = "+".join(sorted(item.pop("routes")))

    logger.debug(
        "Retrieval results merged: groups=%s raw_items=%s unique_items=%s returned=%s",
        group_count,
        total_items,
        len(ranked),
        min(len(ranked), limit),
    )
    return ranked[:limit]


class KeywordBM25Index:
    """
    本地轻量 BM25 索引。

    这个索引作为 Milvus 的关键词召回补充，随 insert_data 同步追加。
    """

    def __init__(self, index_path: str = None):
        self.index_path = Path(
            index_path
            or os.getenv(
                "ROLEPLAY_KEYWORD_INDEX_PATH",
                os.path.join(settings.BASE_DIR, "data", "keyword_index.jsonl"),
            )
        )
        self._lock = threading.RLock()
        self.documents: List[Dict] = []
        self.doc_freqs: Counter = Counter()
        self.avg_doc_len = 0.0
        self._loaded = False
        logger.info("Keyword BM25 index initialized: path=%s", self.index_path)

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return

            # 延迟加载：避免应用启动时因为关键词索引较大而拖慢首屏初始化。
            self.documents = []
            self.doc_freqs = Counter()
            total_length = 0
            skipped_invalid_json = 0
            skipped_empty_tokens = 0

            if self.index_path.exists():
                logger.info("Loading keyword index: path=%s", self.index_path)
                with self.index_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            skipped_invalid_json += 1
                            continue

                        tokens = item.get("tokens") or tokenize_for_retrieval(
                            item.get("text", "")
                        )
                        if not tokens:
                            skipped_empty_tokens += 1
                            continue

                        # 查询阶段会反复计算 BM25，提前缓存词频能减少每次搜索的 CPU 开销。
                        token_counts = Counter(tokens)
                        document = {
                            "text": item.get("text", ""),
                            "meta": item.get("meta", ""),
                            "tokens": tokens,
                            "token_counts": token_counts,
                            "length": len(tokens),
                        }
                        self.documents.append(document)
                        self.doc_freqs.update(token_counts.keys())
                        total_length += len(tokens)
            else:
                logger.info("Keyword index file does not exist yet: path=%s", self.index_path)

            self.avg_doc_len = total_length / len(self.documents) if self.documents else 0.0
            self._loaded = True
            logger.info(
                "Keyword index loaded: path=%s documents=%s terms=%s avg_doc_len=%.2f skipped_json=%s skipped_empty=%s",
                self.index_path,
                len(self.documents),
                len(self.doc_freqs),
                self.avg_doc_len,
                skipped_invalid_json,
                skipped_empty_tokens,
            )

    @staticmethod
    def _is_visible_to_user(meta: str, user_id: str = None) -> bool:
        if not user_id:
            return True

        # 用户上传资料必须隔离；公共知识库内容不带 user_pdf/user_image 前缀，所有用户可见。
        is_user_upload = meta.startswith("user_pdf:") or meta.startswith("user_image:")
        if not is_user_upload:
            return True

        return (
            meta.startswith(f"user_pdf:{user_id}/")
            or meta.startswith(f"user_image:{user_id}/")
        )

    def add_documents(self, texts: List[str], metas: List[str]) -> None:
        """
        追加索引文档并持久化到 JSONL。
        """
        if not texts:
            logger.debug("Keyword index add skipped: empty text batch.")
            return

        self._ensure_loaded()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        new_items: List[Tuple[Dict, Dict]] = []
        skipped_empty_text = 0
        skipped_empty_tokens = 0
        for text, meta in zip(texts, metas):
            clean_text = str(text or "").strip()
            if not clean_text:
                skipped_empty_text += 1
                continue

            tokens = tokenize_for_retrieval(clean_text)
            if not tokens:
                skipped_empty_tokens += 1
                continue

            # JSONL 存原始 tokens，进程重启后可直接恢复索引，避免重新分词全部 chunk。
            serializable = {
                "text": clean_text,
                "meta": str(meta or ""),
                "tokens": tokens,
            }
            document = {
                **serializable,
                "token_counts": Counter(tokens),
                "length": len(tokens),
            }
            new_items.append((serializable, document))

        if not new_items:
            logger.debug(
                "Keyword index add skipped: batch=%s skipped_empty_text=%s skipped_empty_tokens=%s",
                len(texts),
                skipped_empty_text,
                skipped_empty_tokens,
            )
            return

        with self._lock:
            with self.index_path.open("a", encoding="utf-8") as file:
                for serializable, document in new_items:
                    file.write(json.dumps(serializable, ensure_ascii=False) + "\n")
                    self.documents.append(document)
                    self.doc_freqs.update(document["token_counts"].keys())

            total_length = sum(document["length"] for document in self.documents)
            self.avg_doc_len = total_length / len(self.documents)
            logger.info(
                "Keyword index updated: added=%s total_documents=%s avg_doc_len=%.2f path=%s",
                len(new_items),
                len(self.documents),
                self.avg_doc_len,
                self.index_path,
            )

    def search(self, query: str, limit: int = 10, user_id: str = None) -> List[Dict]:
        """
        BM25 关键词召回。
        """
        self._ensure_loaded()
        query_terms = tokenize_for_retrieval(query)
        if not query_terms or not self.documents:
            logger.debug(
                "Keyword search skipped: query_terms=%s documents=%s user_filtered=%s",
                len(query_terms),
                len(self.documents),
                bool(user_id),
            )
            return []

        query_counts = Counter(query_terms)
        doc_count = len(self.documents)
        avg_doc_len = self.avg_doc_len or 1.0
        # k1/b 使用 BM25 常见默认值；当前索引规模较小时足够稳定，后续可配置化。
        k1 = 1.5
        b = 0.75

        scored = []
        visible_documents = 0
        with self._lock:
            for document in self.documents:
                meta = document["meta"]
                if not self._is_visible_to_user(meta, user_id=user_id):
                    continue

                visible_documents += 1
                score = 0.0
                doc_len = document["length"] or 1
                token_counts = document["token_counts"]

                for term, query_count in query_counts.items():
                    freq = token_counts.get(term, 0)
                    if freq <= 0:
                        continue

                    doc_freq = self.doc_freqs.get(term, 0)
                    idf = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                    denominator = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
                    score += query_count * idf * (freq * (k1 + 1)) / denominator

                if score > 0:
                    scored.append({
                        "text": document["text"],
                        "score": score,
                        "meta": meta,
                        "route": "keyword",
                    })

        scored.sort(key=lambda item: item["score"], reverse=True)
        logger.debug(
            "Keyword search completed: query_terms=%s visible_documents=%s matched=%s returned=%s user_filtered=%s",
            len(query_terms),
            visible_documents,
            len(scored),
            min(len(scored), limit),
            bool(user_id),
        )
        return scored[:limit]


keyword_index = KeywordBM25Index()
