# -*- coding: utf-8 -*-
"""
存储模块（Storage Layer）

功能概述：
- Milvus：长期知识库（向量 + 文本 + 元数据）
- Redis：短期对话记忆（Session Memory）
- 为 RAG 系统提供检索与上下文支持
"""

import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import redis
import json
import uuid
from config import settings
from hybrid_retrieval import keyword_index


logger = logging.getLogger(__name__)


# ============================================================
# Milvus 向量数据库管理器
# ============================================================

class MilvusManager:
    """
    Milvus 管理器

    职责：
    - 连接 Milvus
    - 创建/重建 Collection
    - 插入向量数据
    - 执行向量检索
    """

    def __init__(self):
        """
        初始化 Milvus 连接并创建集合
        """
        try:
            connections.connect(uri=settings.MILVUS_URI)
            logger.info("Milvus connected: uri=%s", settings.MILVUS_URI)
        except Exception as e:
            logger.exception("Milvus connection failed: uri=%s", settings.MILVUS_URI)
            raise e

        self.collection_name = settings.MILVUS_COLLECTION
        self._create_collection()

    def _create_collection(self):
        """
        创建或重建 Milvus Collection

        说明：
        - 如果集合已存在，会先删除再重建，确保字段定义最新
        """
        if utility.has_collection(self.collection_name):
            logger.warning(
                "Milvus collection exists and will be dropped: collection=%s",
                self.collection_name,
            )
            utility.drop_collection(self.collection_name)
            logger.info(
                "Milvus collection dropped: collection=%s",
                self.collection_name,
            )

        logger.info(
            "Creating Milvus collection: collection=%s",
            self.collection_name,
        )

        # 定义字段结构
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=1024,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="meta",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
        ]

        schema = CollectionSchema(fields, "Character RAG Storage")
        self.collection = Collection(self.collection_name, schema)

        # 创建向量索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 200},
        }
        self.collection.create_index("vector", index_params)

        # 加载集合
        self.collection.load()
        logger.info(
            "Milvus collection created and loaded: collection=%s",
            self.collection_name,
        )

    def insert_data(self, texts, vectors, metas):
        """
        向 Milvus 插入数据

        :param texts: 文本列表（对应 content 字段）
        :param vectors: 向量列表
        :param metas: 元数据列表
        """
        logger.debug(
            "Milvus insert started: collection=%s count=%s vector_count=%s meta_count=%s",
            self.collection_name,
            len(texts),
            len(vectors) if vectors is not None else 0,
            len(metas),
        )

        # 为每个文本生成唯一 ID
        ids = [f"chunk_{uuid.uuid4().hex}" for _ in texts]

        # 构造插入实体（顺序必须与 schema 一致）
        entities = [
            ids,
            vectors,
            texts,   # 对应 content 字段
            metas,   # 对应 meta 字段
        ]

        self.collection.insert(entities)
        logger.debug(
            "Milvus insert completed: collection=%s count=%s",
            self.collection_name,
            len(texts),
        )
        try:
            keyword_index.add_documents(texts=texts, metas=metas)
        except Exception:
            logger.exception("Keyword index update failed after Milvus insert.")

    def search(self, vector, limit=5, user_id=None):
        """
        向量相似度检索

        :param vector: 查询向量
        :param limit: 返回结果数量
        :param user_id: 用户 ID（用于过滤用户上传的数据）
        """
        search_params = {"metric_type": "COSINE", "params": {"radius": 0.0}}

        # 用户数据优先，扩大检索范围
        search_limit = limit * 5 if user_id else limit
        logger.debug(
            "Milvus search started: collection=%s limit=%s search_limit=%s user_filtered=%s",
            self.collection_name,
            limit,
            search_limit,
            bool(user_id),
        )

        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param=search_params,
            limit=search_limit,
            output_fields=["content", "meta"],
        )

        hits = []
        for hit in results[0]:
            meta = hit.entity.get("meta") or ""

            # 用户数据隔离
            if user_id:
                is_user_upload = meta.startswith("user_pdf:") or meta.startswith("user_image:")
                is_current_user_upload = (
                    meta.startswith(f"user_pdf:{user_id}/")
                    or meta.startswith(f"user_image:{user_id}/")
                )
                if is_user_upload and not is_current_user_upload:
                    continue

            hits.append({
                "text": hit.entity.get("content"),
                "score": hit.score,
                "meta": meta,
            })

            if len(hits) >= limit:
                break

        logger.debug(
            "Milvus search completed: collection=%s requested_limit=%s returned=%s user_filtered=%s",
            self.collection_name,
            limit,
            len(hits),
            bool(user_id),
        )
        return hits


# 全局 Milvus 实例
milvus_db = MilvusManager()


# ============================================================
# Redis 短期记忆管理器
# ============================================================

class RedisMemory:
    """
    Redis 短期记忆管理器

    职责：
    - 保存最近几轮对话
    - 为 RAG 提供短期上下文
    """

    def __init__(self):
        """
        初始化 Redis 连接
        """
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
            self.client.ping()
            logger.info(
                "Redis connected: host=%s port=%s db=%s",
                settings.REDIS_HOST,
                settings.REDIS_PORT,
                settings.REDIS_DB,
            )
        except Exception as e:
            logger.exception(
                "Redis connection failed: host=%s port=%s db=%s",
                settings.REDIS_HOST,
                settings.REDIS_PORT,
                settings.REDIS_DB,
            )
            raise e

        self.max_history = 10

    def add_message(self, user_id, char_id, role, content):
        """
        添加一条对话记录
        """
        key = f"session:{user_id}:{char_id}:history"
        msg_json = json.dumps({"role": role, "content": content})
        self.client.lpush(key, msg_json)
        self.client.ltrim(key, 0, self.max_history - 1)

    def get_history(self, user_id, char_id):
        """
        获取最近的对话历史
        """
        key = f"session:{user_id}:{char_id}:history"
        messages = self.client.lrange(key, 0, -1)
        # Redis 中是倒序存储，需要反转
        return [json.loads(msg) for msg in reversed(messages)]


# 全局 Redis 实例
redis_mem = RedisMemory()
