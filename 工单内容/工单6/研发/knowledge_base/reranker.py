"""
工单编号: 人工智能NLP-RAG-混合检索任务
重排器模块 - 提供3种重排算法（LLM/TF-IDF/CrossEncoder）
"""
import json
import re
import requests
from typing import List, Dict, Optional


class BaseReranker:
    """重排器基类"""

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        raise NotImplementedError


# ============================================================
# 重排器1: 基于LLM的重排器
# ============================================================
class LLMReranker(BaseReranker):
    """
    基于LLM的重排器（工单6新增）

    对每个chunk，让Kimi评估其与query的相关性（0-10分），
    然后按LLM评分重新排序。

    策略：为避免大量API调用，采用分批并行评估，
    每个chunk提取前200字符作为摘要送评。
    """

    def __init__(self, generator=None, llm_config=None):
        """
        Args:
            generator: LLMGenerator实例（当llm_config未提供时使用）
            llm_config: dict with api_key, api_url, model, timeout
        """
        self.generator = generator
        self._score_cache = {}

        # Kimi API配置（优先使用）
        self._llm_config = llm_config or {}
        self._api_key = self._llm_config.get("API密钥", "")
        self._api_url = self._llm_config.get("API地址", "https://api.moonshot.cn/v1").rstrip("/") + "/chat/completions"
        self._model = self._llm_config.get("模型", "moonshot-v1-8k")
        self._timeout = self._llm_config.get("超时", 30)

        # LLM评分提示模板
        self.prompt_template_zh = (
            "你是一个文档相关性评估专家。请评估以下文档片段与用户问题的相关性。\n\n"
            "用户问题：{query}\n\n"
            "文档片段（摘要）：\n{chunk_text}\n\n"
            "请从0-10分给出相关性评分（0=完全不相关，10=完全相关）。"
            "只需返回一个数字，不要有其他文字。\n评分："
        )

    def _call_llm(self, prompt: str) -> str:
        """调用Kimi/OpenAI兼容API获取评分"""
        if self._api_key:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 16,
            }
            try:
                resp = requests.post(
                    self._api_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"  [LLMReranker] Kimi API调用失败: {e}")
                return ""
        elif self.generator:
            result = self.generator.generate(prompt, "")
            return result.get("answer", "").strip()
        return ""

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """使用LLM对chunk进行重排评分"""
        if not results:
            return results[:top_k]
        if not self._api_key and not self.generator:
            print("  [LLMReranker] 无可用LLM配置，跳过重排")
            return results[:top_k]

        from ..qa_engine.generator import detect_language
        lang = detect_language(query)
        template = self.prompt_template_zh

        scored = []
        for r in results:
            chunk = r.get("chunk", r)
            text = chunk.get("text", "")
            if not text:
                text = r.get("text", "")
            chunk_text = text[:300]  # 取前300字符作为摘要

            # 检查缓存
            cache_key = f"{hash(query)}:{hash(chunk_text[:100])}"
            if cache_key in self._score_cache:
                score = self._score_cache[cache_key]
            else:
                prompt = template.format(query=query, chunk_text=chunk_text)
                try:
                    answer = self._call_llm(prompt)
                    # 提取数字
                    nums = re.findall(r"\d+(?:\.\d+)?", answer)
                    score = float(nums[0]) / 10.0 if nums else 0.5
                    score = max(0.0, min(1.0, score))
                except Exception:
                    score = r.get("final_score", r.get("score", 0))
                self._score_cache[cache_key] = score

            entry = dict(r)
            entry["llm_score"] = score
            entry["final_score"] = score
            scored.append(entry)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]


# ============================================================
# 重排器2: 基于TF-IDF的重排器
# ============================================================
class TFIDFReranker(BaseReranker):
    """
    基于TF-IDF的重排器（工单6新增）

    对检索结果的chunk文本计算与query的TF-IDF余弦相似度，
    以此重新排序。

    优势：快速、无外部依赖、可解释性强
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer='char',
            ngram_range=(2, 4),
            lowercase=False,
        )
        self._fitted = False

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """使用TF-IDF余弦相似度重排"""
        if not results:
            return results[:top_k]

        # 收集所有文本
        texts = [query]
        for r in results:
            chunk = r.get("chunk", r)
            texts.append(chunk.get("text", r.get("text", "")))

        # 向量化
        if not self._fitted:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            tfidf_matrix = self.vectorizer.transform(texts)

        # 计算余弦相似度（query vs each chunk）
        query_vec = tfidf_matrix[0:1]
        chunk_vecs = tfidf_matrix[1:]
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, chunk_vecs).flatten()

        scored = []
        for i, r in enumerate(results):
            entry = dict(r)
            entry["tfidf_score"] = float(similarities[i]) if i < len(similarities) else 0.0
            entry["final_score"] = entry["tfidf_score"]
            scored.append(entry)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]


# ============================================================
# 重排器3: 基于CrossEncoder的重排器（bge-reranker-v2-m3）
# ============================================================
class CrossEncoderReranker(BaseReranker):
    """
    基于CrossEncoder的重排器（工单6新增）

    使用 bge-reranker-v2-m3 模型对 query 和每个 chunk 进行深度语义相关性评分。
    CrossEncoder 比 BiEncoder（向量检索）更准确，因为 query 和 chunk 在编码时
    会相互注意力交互。
    """

    def __init__(self, model_path: str = ""):
        """
        Args:
            model_path: CrossEncoder模型本地路径
        """
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            load_path = self.model_path or "BAAI/bge-reranker-v2-m3"
            print(f"[CrossEncoder] 加载重排模型: {load_path}")
            self._model = CrossEncoder(load_path, device="cpu", max_length=512)
            print("[CrossEncoder] 模型加载完成")
        except Exception as e:
            print(f"[CrossEncoder] 模型加载失败: {e}")
            self._model = None

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """使用CrossEncoder重排"""
        if not results:
            return results[:top_k]

        self._load_model()
        if self._model is None:
            # 降级：保持原顺序
            return results[:top_k]

        # 准备(query, chunk) pair
        pairs = []
        for r in results:
            chunk = r.get("chunk", r)
            text = chunk.get("text", r.get("text", ""))
            pairs.append((query, text[:512]))

        try:
            # CrossEncoder评分
            scores = self._model.predict(pairs)
        except Exception as e:
            print(f"[CrossEncoder] 评分失败: {e}")
            return results[:top_k]

        scored = []
        for i, r in enumerate(results):
            entry = dict(r)
            entry["crossencoder_score"] = float(scores[i]) if i < len(scores) else 0.0
            entry["final_score"] = entry["crossencoder_score"]
            scored.append(entry)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]


# ============================================================
# 重排器工厂
# ============================================================
def create_reranker(reranker_type: str = "tfidf",
                    generator=None,
                    model_path: str = "",
                    llm_config: dict = None) -> BaseReranker:
    """
    创建重排器实例

    Args:
        reranker_type: "llm" / "tfidf" / "crossencoder"
        generator: LLMGenerator实例（LLM重排器需要）
        model_path: CrossEncoder模型路径
        llm_config: LLM API配置（LLM重排器使用Kimi）

    Returns:
        BaseReranker子类实例
    """
    if reranker_type == "llm":
        return LLMReranker(generator=generator, llm_config=llm_config)
    elif reranker_type == "tfidf":
        return TFIDFReranker()
    elif reranker_type == "crossencoder":
        return CrossEncoderReranker(model_path=model_path)
    else:
        print(f"[Reranker] 未知重排器类型: {reranker_type}，使用TF-IDF")
        return TFIDFReranker()
