"""
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
图像向量存储与检索模块
存储图像向量索引，支持图文跨模态检索
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class ImageIndexEntry:
    """图像索引条目"""
    image_id: str                    # 唯一标识
    image_path: str                  # 图像文件路径
    page_num: int                    # 页码
    source_file: str                 # 来源PDF
    image_type: str                  # 图像类型
    description: str                 # 文本描述
    caption_text: str = ""           # 图标题
    embedding: Optional[np.ndarray] = None  # 向量（不序列化）
    image_width: int = 0             # 宽度
    image_height: int = 0            # 高度

    def to_metadata(self) -> Dict:
        """转为可序列化元数据"""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "page_num": self.page_num,
            "source_file": self.source_file,
            "image_type": self.image_type,
            "description": self.description[:200],
            "caption_text": self.caption_text,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


class ImageStore:
    """
    图像向量存储 - 图文跨模态检索。

    存储图像嵌入向量，支持:
    1. 文本→图像 的跨模态检索
    2. 图像→文本 的关联检索（CLIP可用时）
    3. 按PDF来源分组检索
    """

    def __init__(self, encoder: 'ImageEmbeddingEncoder'):
        self.encoder = encoder
        self.entries: List[ImageIndexEntry] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D) 矩阵
        self.metadata_path = ""
        self.index_path = ""

    def add_images(self, images: List[Dict], source_file: str = "") -> int:
        """
        添加图像到索引。

        Args:
            images: 图像信息列表 [{image_path, page_num, description, ...}]
            source_file: 来源PDF文件名

        Returns:
            int: 添加数量
        """
        added = 0
        for img in images:
            # 跳过已有（按路径去重）
            if any(e.image_path == img.get("image_path", "") for e in self.entries):
                continue

            image_id = f"{source_file}_p{img.get('page_num', 0)}_{added}"
            entry = ImageIndexEntry(
                image_id=image_id,
                image_path=img.get("image_path", ""),
                page_num=img.get("page_num", 0),
                source_file=source_file or img.get("source_file", ""),
                image_type=img.get("image_type", "未知"),
                description=img.get("description", img.get("surrounding_text", "")),
                caption_text=img.get("caption_text", ""),
                image_width=img.get("width", 0),
                image_height=img.get("height", 0),
            )

            # 编码描述文本（作为图像表示）
            desc_text = entry.description or entry.caption_text or f"第{entry.page_num}页的图像"
            emb = self.encoder.encode_text([desc_text])[0]
            entry.embedding = emb
            self.entries.append(entry)
            added += 1

        # 重建向量矩阵
        self._rebuild_embeddings()
        return added

    def _rebuild_embeddings(self):
        """重建完整的向量矩阵"""
        if not self.entries:
            self.embeddings = None
            return
        vecs = []
        for e in self.entries:
            if e.embedding is not None:
                vecs.append(e.embedding)
        if vecs:
            self.embeddings = np.stack(vecs, axis=0)
        else:
            self.embeddings = None

    def search(self, query_text: str, top_k: int = 3,
               source_filter: Optional[str] = None) -> List[Dict]:
        """
        文本查询→检索最相关图像。

        Args:
            query_text: 查询文本
            top_k: 返回数量
            source_filter: 按来源PDF筛选

        Returns:
            List[Dict]: 排序后的结果 [{
                "image_path": str, "page_num": int,
                "score": float, "description": str, ...
            }]
        """
        if not self.entries or self.embeddings is None:
            return []

        # 编码查询文本
        query_emb = self.encoder.encode_text([query_text])[0]

        # 计算相似度
        scores = self.encoder.compute_similarity(query_emb, self.embeddings)

        # 排序
        indices = np.argsort(-scores)  # 降序

        results = []
        for idx in indices:
            entry = self.entries[idx]
            score = float(scores[idx])

            # 来源过滤
            if source_filter and entry.source_file != source_filter:
                continue

            results.append({
                "image_path": entry.image_path,
                "page_num": entry.page_num,
                "source_file": entry.source_file,
                "image_type": entry.image_type,
                "description": entry.description[:300],
                "caption_text": entry.caption_text,
                "score": round(score, 4),
                "image_width": entry.image_width,
                "image_height": entry.image_height,
            })

            if len(results) >= top_k:
                break

        return results

    def search_by_page(self, page_num: int,
                       source_file: str = "") -> List[Dict]:
        """按页码查询该页的所有图像"""
        results = []
        for entry in self.entries:
            if entry.page_num == page_num:
                if source_file and entry.source_file != source_file:
                    continue
                results.append({
                    "image_path": entry.image_path,
                    "page_num": entry.page_num,
                    "source_file": entry.source_file,
                    "image_type": entry.image_type,
                    "description": entry.description[:300],
                    "caption_text": entry.caption_text,
                })
        return results

    def get_image_count(self, source_file: Optional[str] = None) -> int:
        """获取图像数量"""
        if source_file:
            return sum(1 for e in self.entries if e.source_file == source_file)
        return len(self.entries)

    def clear(self):
        """清空索引"""
        self.entries.clear()
        self.embeddings = None

    def save(self, index_path: str, metadata_path: str):
        """
        保存索引（元数据JSON + 向量npy）。

        Args:
            index_path: 向量保存路径 (.faiss 兼容)
            metadata_path: 元数据保存路径
        """
        self.metadata_path = metadata_path
        self.index_path = index_path

        # 保存元数据
        metadata = [e.to_metadata() for e in self.entries]
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 保存向量为npy
        if self.embeddings is not None:
            npy_path = index_path.replace(".faiss", ".images.npy")
            np.save(npy_path, self.embeddings)
            print(f"图像索引已保存: {len(self.entries)} 个向量 → {npy_path}")

    def load(self, index_path: str, metadata_path: str) -> bool:
        """
        加载已保存的索引。

        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(metadata_path):
            return False

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 加载向量
        npy_path = index_path.replace(".faiss", ".images.npy")
        if os.path.exists(npy_path):
            self.embeddings = np.load(npy_path)
        else:
            return False

        # 重建条目
        self.entries.clear()
        for m in metadata:
            entry = ImageIndexEntry(
                image_id=m.get("image_id", ""),
                image_path=m.get("image_path", ""),
                page_num=m.get("page_num", 0),
                source_file=m.get("source_file", ""),
                image_type=m.get("image_type", "未知"),
                description=m.get("description", ""),
                caption_text=m.get("caption_text", ""),
                image_width=m.get("image_width", 0),
                image_height=m.get("image_height", 0),
            )
            self.entries.append(entry)

        print(f"图像索引已加载: {len(self.entries)} 个条目")
        return True

    def get_page_image_map(self) -> Dict[str, List[Dict]]:
        """获取(文件名_页码)→图像的映射，用于UI显示"""
        page_map = {}
        for entry in self.entries:
            key = f"{entry.source_file}_{entry.page_num}"
            if key not in page_map:
                page_map[key] = []
            page_map[key].append({
                "image_path": entry.image_path,
                "image_type": entry.image_type,
                "caption": entry.caption_text,
                "description": entry.description[:200],
            })
        return page_map
