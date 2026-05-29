# -*- coding: utf-8 -*-
# config.py
import os


class Settings:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- 本地模型路径配置 (根据你的实际路径填写) ---
    # BGE-m3 向量化模型路径
    LOCAL_BGE_M3_PATH = "/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-m3"

    # BGE-Reranker 重排序模型路径
    LOCAL_BGE_RERANK_PATH = "/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3"

    # --- Milvus 配置 ---
    MILVUS_URI = "http://localhost:19530"
    MILVUS_COLLECTION = "character_knowledge_v1"

    # --- Redis 配置 ---
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0

    # --- 大模型配置 ---
    # 原 Ollama 本地模型配置：
    # LLM_API_BASE = "http://localhost:11434/v1"
    # LLM_MODEL_NAME = "deepseek-r1:7b"
    # LLM_API_KEY = "EMPTY"

    # DeepSeek 在线模型配置：
    LLM_API_BASE = "https://api.deepseek.com"
    LLM_MODEL_NAME = "deepseek-chat"
    # 必须通过环境变量传入，禁止硬编码：
    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

    # --- 应用配置 ---
    APP_DB_PATH = os.getenv("ROLEPLAY_APP_DB_PATH", os.path.join(BASE_DIR, "data", "app_data.db"))
    SECRET_KEY = os.getenv("ROLEPLAY_SECRET_KEY", "replace-this-secret-before-production")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ROLEPLAY_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    ADMIN_TOKEN = os.getenv("ROLEPLAY_ADMIN_TOKEN", "").strip()

    # --- MinerU 在线 API 配置 ---
    # token 通过环境变量 MINERU_API_TOKEN 注入，避免硬编码
    MINERU_API_BASE = os.getenv("MINERU_API_BASE", "https://mineru.net/api/v4").rstrip("/")
    MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "").strip()
    MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "600"))


settings = Settings()
