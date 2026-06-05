"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
Embedding 模块 - 将文本转换为向量表示
"""

import os
import hashlib
import json


class EmbeddingModel:
    """文本嵌入模型封装，支持 sentence-transformers 和 TF-IDF 回退两种方式。

    在无法连接 HuggingFace 时，自动回退到 TF-IDF 向量化，
    确保项目可离线运行。
    """

    def __init__(self, model_name: str = None,
                 device: str = "cpu", normalize: bool = True,
                 use_local_fallback: bool = True):
        """
        Args:
            model_name: 模型名称，默认自动选择
            device: cpu / mps (Apple Silicon)
            normalize: 是否对向量做L2归一化
            use_local_fallback: 无法下载模型时是否回退到 TF-IDF
        """
        self.model_name = model_name or "BAAI/bge-small-zh-v1.5"
        self.device = device
        self.normalize = normalize
        self.use_local_fallback = use_local_fallback
        self._model = None
        self._fallback_mode = False
        # 尝试使用镜像站
        import os
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    def _load_model(self):
        """延迟加载模型，避免启动时加载"""
        if self._model is not None:
            return
        # 尝试加载 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] 加载模型: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if "bge" in self.model_name.lower():
                self._query_prefix = "为这个句子生成表示以用于检索相关文章："
            else:
                self._query_prefix = ""
            print(f"[Embedding] 模型加载完成，向量维度: {self._model.get_sentence_embedding_dimension()}")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[Embedding] 模型下载失败: {e}")
            if not self.use_local_fallback:
                raise

        # 回退到 TF-IDF
        print(f"[Embedding] 使用 TF-IDF 回退模式（本地无需下载）")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(
            max_features=512,
            analyzer='char',
            ngram_range=(2, 4),
            lowercase=False,
        )
        self._fallback_mode = True
        self._vocab_built = False
        self._query_prefix = ""

    def encode(self, texts: list, is_query: bool = False) -> list:
        """
        将文本列表转为向量。

        Args:
            texts: 文本列表
            is_query: 是否为查询文本（BGE模型需要添加query前缀）

        Returns:
            list[list[float]]: 向量列表
        """
        self._load_model()

        if self._fallback_mode:
            return self._encode_tfidf(texts)

        if is_query and self._query_prefix:
            texts = [self._query_prefix + t for t in texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _encode_tfidf(self, texts: list) -> list:
        """使用 TF-IDF 进行向量化"""
        if not self._vocab_built:
            self._tfidf.fit(texts)
            self._vocab_built = True
        else:
            # 增量更新词汇表
            self._tfidf.fit(list(self._tfidf.get_feature_names_out()) + texts)
        matrix = self._tfidf.transform(texts)
        # 转为 dense list
        import numpy as np
        dense = matrix.toarray()
        if self.normalize:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            dense = dense / norms
        return dense.tolist()

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        self._load_model()
        if self._fallback_mode:
            return len(self._tfidf.get_feature_names_out()) if self._vocab_built else 512
        return self._model.get_sentence_embedding_dimension()

    def encode_single(self, text: str, is_query: bool = False) -> list[float]:
        """编码单条文本"""
        return self.encode([text], is_query=is_query)[0]

    def get_cache_key(self, text: str) -> str:
        """生成文本的缓存键（用于加速重复文本的编码）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    model = EmbeddingModel("BAAI/bge-small-zh-v1.5")  # 轻量版本用于测试
    vec = model.encode(["武汉兴图新科电子股份有限公司"])
    print(f"向量维度: {len(vec[0])}")
    print(f"前5个值: {vec[0][:5]}")
