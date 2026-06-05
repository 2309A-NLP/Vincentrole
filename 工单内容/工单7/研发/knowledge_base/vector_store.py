"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
向量存储 - FAISS + BM25混合索引
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple


class VectorStore:
    """
    向量存储 - 支持向量检索 + BM25关键词检索
    """

    def __init__(self, dimension: int,
                 index_path: str = "data/vector_index.faiss",
                 metadata_path: str = "data/chunk_metadata.json"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunks: List[Dict] = []
        self.index = None
        self.bm25 = None
        self._ensure_dirs()

    def _ensure_dirs(self):
        for p in [self.index_path, self.metadata_path]:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def build_index(self, chunks: List[Dict], embeddings: np.ndarray):
        """构建索引（向量 + BM25）"""
        import faiss
        n = len(chunks)
        if n == 0:
            return

        self.chunks = chunks
        emb = np.array(embeddings, dtype="float32")

        # FAISS索引（IP + 归一化 = cosine）
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(emb)
        self.index.add(emb)

        # BM25索引
        self._build_bm25()

    def _build_bm25(self):
        """构建BM25关键词索引"""
        try:
            from rank_bm25 import BM25Okapi
            texts = [c["text"] for c in self.chunks]
            tokenized = [self._tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(tokenized)
        except ImportError:
            print("[VectorStore] 未安装 rank-bm25，关键词检索不可用")
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """简单中英文分词"""
        import re
        # 中文按字符分割，英文按单词
        tokens = []
        for word in re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text.lower()):
            tokens.append(word)
        return tokens

    def search(self, query_embedding: np.ndarray,
               query_text: str,
               top_k: int = 5,
               alpha: float = 0.7) -> List[Dict]:
        """
        混合检索：向量 + BM25，RRF融合

        Args:
            query_embedding: 查询向量
            query_text: 查询文本
            top_k: 返回最多几个结果
            alpha: 向量权重（BM25权重为 1-alpha）

        Returns:
            List[Dict]: [{"chunk": ..., "score": ..., "rank": ...}, ...]
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        # 向量检索
        vec_results = self._vector_search(query_embedding, top_k * 3)

        # BM25检索
        bm25_results = self._bm25_search(query_text, top_k * 3)

        # RRF融合
        fused = self._rrf_fuse(vec_results, bm25_results, k=60)

        return fused[:top_k]

    def _vector_search(self, query_embedding: np.ndarray,
                       top_k: int) -> List[Tuple[int, float]]:
        """向量检索，返回[(chunk_idx, score)]"""
        import faiss
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > -1:
                results.append((int(idx), float(score)))
        return results

    def _bm25_search(self, query_text: str,
                     top_k: int) -> List[Tuple[int, float]]:
        """BM25关键词检索，返回[(chunk_idx, score)]"""
        if self.bm25 is None:
            return []

        tokenized = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized)

        # 取top_k个最高分
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results

    def _rrf_fuse(self, vec_results: List[Tuple[int, float]],
                  bm25_results: List[Tuple[int, float]],
                  k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) 融合向量和BM25结果

        RRF分数 = sum(1 / (k + rank))
        """
        rrf_scores = {}

        # 向量结果
        for rank, (idx, score) in enumerate(vec_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        # BM25结果
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        # 按RRF分排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in sorted_items:
            chunk = self.chunks[idx].copy()
            chunk["_idx"] = idx
            results.append({
                "chunk": chunk,
                "score": score,
            })
        return results

    # ============================================================
    # 持久化
    # ============================================================
    def save(self):
        if self.index is None:
            return
        import faiss
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        # BM25也存起来
        bm25_path = self.metadata_path.replace(".json", "_bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self) -> bool:
        import faiss
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return False
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            # 加载BM25
            bm25_path = self.metadata_path.replace(".json", "_bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
            else:
                self._build_bm25()
            return True
        except Exception as e:
            print(f"[VectorStore] 加载失败: {e}")
            return False


if __name__ == "__main__":
    store = VectorStore(dimension=5)
    chunks = [
        {"text": "武汉兴图新科电子股份有限公司注册资本6000万元", "page": 1},
        {"text": "公司主营业务收入主要来自军用领域", "page": 2},
        {"text": "下游客户主要为军工研究所", "page": 3},
    ]
    embs = np.random.randn(3, 5).astype("float32")
    store.build_index(chunks, embs)
    print(f"存储了 {store.total_chunks} 个chunk")
