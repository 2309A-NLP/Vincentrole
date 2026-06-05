"""
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
图像编码器 - CLIP多模态接口（支持CLIP模型+BGE文本回退）
"""

import os
from typing import List, Optional, Union, Dict
import numpy as np


class ImageEmbeddingEncoder:
    """
    图像编码器 - 多模态Embedding接口。

    设计目标:
    1. 提供统一的 encode_image / encode_text 接口（兼容CLIP风格）
    2. 优先使用CLIP模型（多模态对齐）
    3. CLIP不可用时回退到BGE文本编码（图像描述文本编码）

    使用方式:
        当CLIP模型可用时（权重文件在缓存中）:
            encoder = ImageEmbeddingEncoder(use_clip=True)
            img_emb = encoder.encode_image(image_array)    # 图像→向量
            txt_emb = encoder.encode_text(["描述文本"])    # 文本→向量

        当CLIP不可用时:
            encoder = ImageEmbeddingEncoder(use_clip=False)
            # 仅支持文本编码（图像描述作为文本编码）
            txt_emb = encoder.encode_text(["图像描述文本"])
    """

    def __init__(self, use_clip: bool = False, device: str = "cpu",
                 bge_encoder=None):
        """
        Args:
            use_clip: 是否尝试加载CLIP模型
            device: 运行设备 (cpu / mps)
            bge_encoder: BGE文本编码器实例（用于回退）
        """
        self.device = device
        self.use_clip = use_clip
        self.clip_model = None
        self.bge_encoder = bge_encoder

        if use_clip:
            self._init_clip()

    def _init_clip(self):
        """尝试加载CLIP模型"""
        try:
            # 尝试从本地缓存加载CLIP
            import sentence_transformers
            from sentence_transformers import SentenceTransformer

            # 尝试查找本地缓存
            cache_dir = os.path.expanduser(
                "~/.cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32"
            )
            if os.path.exists(cache_dir):
                snapshots = os.path.join(cache_dir, "snapshots")
                if os.path.exists(snapshots):
                    snaps = os.listdir(snapshots)
                    if snaps:
                        model_path = os.path.join(snapshots, snaps[0])
                        if os.path.exists(os.path.join(model_path, "0_CLIPModel")):
                            model_path = os.path.join(model_path, "0_CLIPModel")
                        self.clip_model = SentenceTransformer(model_path)
                        print(f"CLIP模型已加载（本地缓存）")
                        return

            # 如果本地没有，尝试在线加载（国内可能超时）
            model_name = "sentence-transformers/clip-ViT-B-32"
            self.clip_model = SentenceTransformer(model_name)
            print(f"CLIP模型已加载（在线）")

        except Exception as e:
            print(f"CLIP模型加载失败: {e}")
            print("回退到BGE文本编码模式")
            self.use_clip = False
            self.clip_model = None

    def encode_image(self, image: Union[str, np.ndarray, "PIL.Image"]) -> np.ndarray:
        """
        编码图像为向量。

        Args:
            image: 图像路径(str)或numpy数组或PIL Image

        Returns:
            np.ndarray: 图像向量 (512维或CLIP维度)
        """
        if self.clip_model and self.use_clip:
            return self.clip_model.encode(image, normalize_embeddings=True)
        else:
            # CLIP不可用，返回零向量（表示没有图像编码能力）
            dim = 512
            return np.zeros(dim, dtype=np.float32)

    def encode_images(self, images: List) -> np.ndarray:
        """批量编码图像"""
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        if self.clip_model and self.use_clip:
            return self.clip_model.encode(images, normalize_embeddings=True)
        else:
            return np.zeros((len(images), 512), dtype=np.float32)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为向量（与图像向量在同一空间）。

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: 文本向量
        """
        if self.clip_model and self.use_clip:
            return self.clip_model.encode(texts, normalize_embeddings=True)
        elif self.bge_encoder:
            # 回退到BGE编码（通过图像描述文本）
            emb = self.bge_encoder.encode(texts, is_query=True)
            # 手动归一化（EmbeddingModel.encode不支持normalize_embeddings参数）
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return emb / norms
        else:
            # 无编码器，返回零向量
            dim = 512
            return np.zeros((len(texts), dim), dtype=np.float32)

    def compute_similarity(self, query_vec: np.ndarray,
                           image_vecs: np.ndarray) -> np.ndarray:
        """计算查询向量与图像向量的余弦相似度"""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        image_norms = image_vecs / (np.linalg.norm(image_vecs, axis=1, keepdims=True) + 1e-10)
        return np.dot(image_norms, query_norm)

    @property
    def embedding_dim(self) -> int:
        """返回嵌入维度"""
        if self.clip_model:
            return self.clip_model.get_sentence_embedding_dimension()
        return 512

    def is_clip_available(self) -> bool:
        """检查CLIP是否可用"""
        return self.use_clip and self.clip_model is not None
