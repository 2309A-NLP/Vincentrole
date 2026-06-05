"""
工单编号: 人工智能NLP-RAG-混合检索任务
Embedding 模块 - 支持多模型切换（bge-small/bge-base/m3e/bge-m3）
"""
import os
import hashlib
from typing import List, Optional, Dict


class EmbeddingModel:
    """文本嵌入模型封装，支持多种本地模型。"""

    def __init__(self, model_name: str = "bge-base-zh-v1.5",
                 device: str = "cpu", normalize: bool = True,
                 use_local_fallback: bool = True,
                 model_registry: Optional[Dict] = None):
        """
        Args:
            model_name: EMBEDDING_MODELS 中的 key
            device: cpu / mps (Apple Silicon)
            normalize: 是否对向量做L2归一化
            use_local_fallback: 是否启用本地回退
            model_registry: 模型注册表（注入 config.EMBEDDING_MODELS）
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.use_local_fallback = use_local_fallback
        self._model = None
        self._fallback_mode = False

        # 从注册表中读取模型信息
        self._registry = model_registry or {}
        self._model_info = self._registry.get(model_name, {})
        self._local_path = self._model_info.get("path", "")
        self._query_prefix = self._model_info.get("query_prefix", "")

        # 离线模式
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            load_path = self._local_path if self._local_path else self._model_info.get("hf_name", self.model_name)
            print(f"[Embedding] 加载模型: {self.model_name} ({load_path}) ...")
            self._model = SentenceTransformer(load_path, device=self.device)
            print(f"[Embedding] 模型加载完成，向量维度: {self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[Embedding] 模型加载失败: {e}")
            if not self.use_local_fallback:
                raise
            # 回退到 TF-IDF
            print("[Embedding] 使用 TF-IDF 回退模式")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(max_features=512, analyzer='char', ngram_range=(2, 4), lowercase=False)
            self._fallback_mode = True
            self._query_prefix = ""

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        info_dim = self._model_info.get("dimension", 768)
        if self._fallback_mode and hasattr(self, '_tfidf'):
            try:
                return len(self._tfidf.get_feature_names_out())
            except Exception:
                return info_dim
        # 尝试加载
        self._load_model()
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return info_dim

    def encode(self, texts: list, is_query: bool = False) -> list:
        """将文本列表转为向量列表"""
        self._load_model()

        if self._fallback_mode:
            return self._encode_tfidf(texts)

        processed = []
        for t in texts:
            if is_query and self._query_prefix:
                processed.append(self._query_prefix + t)
            else:
                processed.append(t)

        embeddings = self._model.encode(
            processed,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def _encode_tfidf(self, texts: list) -> list:
        """TF-IDF 回退"""
        self._load_model()
        if not self._fallback_mode:
            return self.encode(texts)
        # 确保 fit 过
        if not hasattr(self, '_vocab_built') or not self._vocab_built:
            self._tfidf.fit(texts)
            self._vocab_built = True
        else:
            self._tfidf.fit(list(self._tfidf.get_feature_names_out()) + texts)
        matrix = self._tfidf.transform(texts)
        import numpy as np
        dense = matrix.toarray()
        if self.normalize:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            dense = dense / norms
        return dense.tolist()

    def encode_single(self, text: str, is_query: bool = False) -> list:
        return self.encode([text], is_query=is_query)[0]

    def get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
