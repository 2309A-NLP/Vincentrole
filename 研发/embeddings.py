# -*- coding: utf-8 -*-
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from config import settings


logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self):
        # 检测设备：Mac 使用 mps，有 NVIDIA 显卡用 cuda，否则 cpu
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Embedding device selected: device=%s", self.device)

        # 1. 加载 Embedding 模型 (BGE-m3) 维度(1024)
        logger.info("Loading BGE-m3 embedding model: path=%s", settings.LOCAL_BGE_M3_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LOCAL_BGE_M3_PATH,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            settings.LOCAL_BGE_M3_PATH,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # 2. 加载 Reranker 模型 (BGE-reranker-v2-m3)
        logger.info("Loading BGE reranker model: path=%s", settings.LOCAL_BGE_RERANK_PATH)
        self.reranker = CrossEncoder(
            settings.LOCAL_BGE_RERANK_PATH,
            max_length=512,  # 这里的长度可以根据显存调整
            device=self.device,
            default_activation_function=torch.nn.functional.sigmoid
        )
        logger.info("Embedding and reranker models loaded.")

    def encode(self, texts, max_length=512):
        """将文本转化为向量"""
        logger.debug("Encoding texts: count=%s max_length=%s", len(texts), max_length)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            # 获取 [CLS] token 的输出作为句向量
            embeddings = outputs.last_hidden_state[:, 0]
            # 归一化 (对于余弦相似度至关重要)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        vectors = embeddings.cpu().numpy()
        logger.debug("Texts encoded: count=%s vector_dim=%s", len(texts), vectors.shape[-1] if len(vectors) else 0)
        return vectors

    def rerank(self, query, passages):
        """
        重排序：根据相关性对检索到的片段进行打分排序
        """
        if not passages:
            logger.debug("Rerank skipped because no passages were provided.")
            return []

        normalized_passages = []
        for passage in passages:
            if isinstance(passage, dict):
                text = str(passage.get("text", "")).strip()
                score = float(passage.get("score", 0.0))
            else:
                text = str(passage).strip()
                score = 0.0

            if text:
                normalized_passages.append({"text": text, "score": score})

        if not normalized_passages:
            logger.debug("Rerank skipped because passages were empty after normalization.")
            return []

        # 构造 [query, passage] 对
        pairs = [[query, item["text"]] for item in normalized_passages]

        # 预测分数
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # 关联分数和内容
        scored_passages = []
        for item, rerank_score in zip(normalized_passages, scores):
            item["rerank_score"] = float(rerank_score)
            scored_passages.append(item)

        # 优先按重排分数排序，分数相同则保留原始检索相似度较高的内容
        scored_passages.sort(key=lambda item: (item["rerank_score"], item["score"]), reverse=True)

        # 返回得分最高的前 3 个片段
        logger.debug("Rerank completed: query_length=%s candidates=%s returned=%s", len(query or ""), len(scored_passages), 3)
        return [item["text"] for item in scored_passages[:3]]


# 全局单例
embed_engine = EmbeddingEngine()

#复杂pdf的解析
