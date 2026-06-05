"""
Milvus向量存储 - 替代FAISS，支持大规模数据扩展

功能:
1. Milvus连接管理
2. 向量存储与检索
3. BM25混合检索（保留关键词检索能力）
4. 元数据管理（与本地JSON同步）
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

import config


class MilvusVectorStore:
    """
    Milvus向量存储实现
    
    设计:
    - Milvus只存向量和元数据，完整文本chunk保留在本地JSON
    - 支持BM25混合检索（本地实现）
    - 元数据与Milvus同步，支持增量加载
    """

    def __init__(self, dimension: int, collection_name: str = "rag_chunks",
                 metadata_path: str = "", milvus_host: str = "127.0.0.1", milvus_port: int = 19530):
        self.dimension = dimension
        self.collection_name = collection_name
        self.metadata_path = metadata_path
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        
        # 本地数据
        self.chunks = []  # 完整chunk文本列表
        self.chunk_metadata = []  # 元数据列表（与Milvus同步）
        
        # Milvus客户端
        self.client = None
        self._connect()
        
        # BM25相关
        self.bm25_index = None
        self.bm25_tokenizer = None
        
        # 统计信息
        self.total_chunks = 0

    def _connect(self):
        """连接Milvus服务器"""
        try:
            uri = f"http://{self.milvus_host}:{self.milvus_port}"
            self.client = MilvusClient(uri=uri)
            print(f"[Milvus] 连接成功: {uri}")
            
            # 检查collection是否存在
            if self.client.has_collection(self.collection_name):
                print(f"[Milvus] Collection '{self.collection_name}' 已存在")
            else:
                print(f"[Milvus] Collection '{self.collection_name}' 不存在，需要创建")
                
        except Exception as e:
            print(f"[Milvus] 连接失败: {e}")
            self.client = None

    def build_index(self, chunks: List[dict], embeddings: np.ndarray):
        """
        构建向量索引
        
        Args:
            chunks: chunk列表，每个chunk包含text, page, type等字段
            embeddings: 向量数组，shape=(n_chunks, dimension)
        """
        if not self.client:
            print("[Milvus] 客户端未连接，无法构建索引")
            return
        
        try:
            # 1. 删除旧collection（如果存在）
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                print(f"[Milvus] 删除旧collection: {self.collection_name}")
            
            # 2. 创建schema（auto_id=True，不手动管理ID）
            schema = CollectionSchema(fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
            ], enable_dynamic_field=True)
            
            # 3. 创建collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
            )
            print(f"[Milvus] 创建collection: {self.collection_name}")
            
            # 4. 准备数据
            self.chunks = chunks
            self.chunk_metadata = []
            
            # 5. 批量插入（Milvus要求）
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                # 准备插入数据
                data = []
                for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
                    chunk_id = i + j
                    text_hash = str(hash(chunk.get("text", "")))
                    
                    # 动态字段：存储元数据
                    dynamic_fields = {
                        "text": chunk.get("text", "")[:1000],  # 限制长度
                        "page": chunk.get("page", 0),
                        "type": chunk.get("type", "text"),
                        "heading": chunk.get("heading", ""),
                        "source_file": chunk.get("source_file", ""),
                    }
                    
                    vector = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    data.append({
                        "vector": vector,
                        "chunk_id": chunk_id,
                        "text_hash": text_hash,
                        **dynamic_fields,
                    })
                
                # 插入数据
                self.client.insert(
                    collection_name=self.collection_name,
                    data=data,
                )
                total_inserted += len(data)
                print(f"[Milvus] 插入批次 {i//batch_size + 1}: {len(data)} 条")
            
            # 6. 创建索引（IVF_FLAT）
            from pymilvus.milvus_client.index import IndexParams
            
            index_params = IndexParams([
                {"key": "index_type", "value": "IVF_FLAT"},
                {"key": "metric_type", "value": "COSINE"},
                {"key": "params", "value": {"nlist": 128}},
            ])
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
            print(f"[Milvus] 创建IVF_FLAT索引")
            
            # 7. 加载collection到内存
            self.client.load_collection(self.collection_name)
            print(f"[Milvus] Collection加载完成")
            
            # 8. 保存元数据到本地
            self.save()
            
            self.total_chunks = total_inserted
            print(f"[Milvus] 索引构建完成: {total_inserted} 个chunk")
            
        except Exception as e:
            print(f"[Milvus] 索引构建失败: {e}")
            import traceback
            traceback.print_exc()

    def load(self) -> bool:
        """从本地文件加载元数据，检查Milvus连接"""
        if not self.client:
            self._connect()
        
        if not self.client:
            return False
        
        # 检查collection是否存在
        if not self.client.has_collection(self.collection_name):
            print(f"[Milvus] Collection '{self.collection_name}' 不存在")
            return False
        
        # 加载collection
        try:
            self.client.load_collection(self.collection_name)
            print(f"[Milvus] Collection '{self.collection_name}' 加载成功")
            
            # 获取chunk数量
            stats = self.client.get_collection_stats(self.collection_name)
            self.total_chunks = stats.get("row_count", 0)
            print(f"[Milvus] 当前chunk数量: {self.total_chunks}")
            
            # 加载本地元数据
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self.chunk_metadata = data.get("metadata", [])
                print(f"[Milvus] 本地元数据加载: {len(self.chunks)} 个chunk")
            
            return True
            
        except Exception as e:
            print(f"[Milvus] 加载失败: {e}")
            return False

    def save(self):
        """保存元数据到本地文件"""
        if not self.metadata_path:
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            data = {
                "chunks": self.chunks,
                "metadata": self.chunk_metadata,
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "total_chunks": self.total_chunks,
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"[Milvus] 元数据保存到: {self.metadata_path}")
            
        except Exception as e:
            print(f"[Milvus] 保存失败: {e}")

    def search(self, query_embedding: np.ndarray = None, query_text: str = "",
               top_k: int = 5, threshold: float = 0.0, query_vector: np.ndarray = None,
               alpha: float = 0.7) -> List[dict]:
        """
        向量检索 - 兼容FAISS VectorStore接口
        
        Args:
            query_embedding: 查询向量（FAISS接口）
            query_text: 查询文本（用于BM25，Milvus暂未实现）
            top_k: 返回数量
            threshold: 相似度阈值
            query_vector: 查询向量（兼容旧接口）
            alpha: 混合检索权重（暂未使用）
            
        Returns:
            检索结果列表
        """
        # 兼容两种参数名
        vec = query_embedding if query_embedding is not None else query_vector
        if vec is None:
            print("[Milvus] search() 缺少查询向量")
            return []
        if not self.client:
            print("[Milvus] 客户端未连接")
            return []
        
        try:
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16},
            }
            
            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vec.tolist() if hasattr(vec, 'tolist') else list(vec)],
                limit=top_k,
                search_params=search_params,
                output_fields=["text", "page", "type", "heading", "source_file", "chunk_id"],
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    score = hit.get("distance", 0)
                    if score < threshold:
                        continue
                    
                    entity = hit.get("entity", {})
                    result = {
                        "chunk": {
                            "text": entity.get("text", ""),
                            "page": entity.get("page", 0),
                            "type": entity.get("type", "text"),
                            "heading": entity.get("heading", ""),
                            "source_file": entity.get("source_file", ""),
                            "chunk_id": entity.get("chunk_id", -1),
                        },
                        "score": float(score),
                        "rank": len(search_results) + 1,
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"[Milvus] 搜索失败: {e}")
            return []

    def hybrid_search(self, query_text: str, query_vector: np.ndarray, top_k: int = 5,
                     vector_weight: float = 0.7, bm25_weight: float = 0.3) -> List[dict]:
        """
        混合检索：向量检索 + BM25关键词检索
        
        Args:
            query_text: 查询文本
            query_vector: 查询向量
            top_k: 返回数量
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            
        Returns:
            混合检索结果
        """
        # 1. 向量检索
        vector_results = self.search(query_vector, top_k=top_k*2)
        
        # 2. BM25检索（如果可用）
        bm25_results = []
        if self.bm25_index:
            bm25_results = self._bm25_search(query_text, top_k=top_k*2)
        
        # 3. 融合RRF（Reciprocal Rank Fusion）
        if not bm25_results:
            return vector_results[:top_k]
        
        # RRF融合
        fused_results = {}
        k = 60  # RRF常数
        
        # 向量结果排名
        for rank, result in enumerate(vector_results):
            chunk_id = result.get("chunk_id", -1)
            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "result": result,
                    "rrf_score": 0,
                }
            fused_results[chunk_id]["rrf_score"] += vector_weight / (k + rank + 1)
        
        # BM25结果排名
        for rank, result in enumerate(bm25_results):
            chunk_id = result.get("chunk_id", -1)
            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "result": result,
                    "rrf_score": 0,
                }
            fused_results[chunk_id]["rrf_score"] += bm25_weight / (k + rank + 1)
        
        # 按RRF分数排序
        sorted_results = sorted(fused_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        # 返回top_k结果
        final_results = []
        for item in sorted_results[:top_k]:
            result = item["result"]
            result["rrf_score"] = item["rrf_score"]
            final_results.append(result)
        
        return final_results

    def _bm25_search(self, query: str, top_k: int = 5) -> List[dict]:
        """BM25关键词检索"""
        if not self.bm25_index or not self.chunks:
            return []
        
        try:
            # 分词（简单空格分词，中文需要分词器）
            query_tokens = query.lower().split()
            
            # BM25搜索
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 获取top_k结果
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks) and scores[idx] > 0:
                    chunk = self.chunks[idx]
                    result = {
                        "text": chunk.get("text", ""),
                        "page": chunk.get("page", 0),
                        "type": chunk.get("type", "text"),
                        "heading": chunk.get("heading", ""),
                        "source_file": chunk.get("source_file", ""),
                        "chunk_id": idx,
                        "bm25_score": float(scores[idx]),
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"[BM25] 搜索失败: {e}")
            return []

    def build_bm25_index(self):
        """构建BM25索引"""
        if not self.chunks:
            print("[BM25] 没有chunk数据，无法构建索引")
            return
        
        try:
            from rank_bm25 import BM25Okapi
            
            # 提取文本
            texts = [chunk.get("text", "").lower() for chunk in self.chunks]
            
            # 分词（简单空格分词）
            tokenized_texts = [text.split() for text in texts]
            
            # 构建BM25索引
            self.bm25_index = BM25Okapi(tokenized_texts)
            print(f"[BM25] 索引构建完成: {len(texts)} 个文档")
            
        except ImportError:
            print("[BM25] rank_bm25 未安装，跳过BM25索引构建")
        except Exception as e:
            print(f"[BM25] 索引构建失败: {e}")

    def get_stats(self) -> dict:
        """获取存储统计信息"""
        stats = {
            "backend": "milvus",
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "total_chunks": self.total_chunks,
            "milvus_connected": self.client is not None,
            "bm25_enabled": self.bm25_index is not None,
        }
        
        if self.client:
            try:
                collection_stats = self.client.get_collection_stats(self.collection_name)
                stats["milvus_row_count"] = collection_stats.get("row_count", 0)
            except:
                pass
        
        return stats