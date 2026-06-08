"""
工单13 - 优化版 RAG 系统配置
基于对比测试结果：TF-IDF重排 + qwen-turbo + top5 最优（595ms）
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_PDFS = [
    os.path.join(DATA_DIR, "招股说明书1.pdf"),
    os.path.join(DATA_DIR, "招股说明书2.pdf"),
]

PDF_CONFIG = {"使用引擎": "pymupdf"}

# ===== 优化1: 分块大小从 800 → 500（减少LLM上下文） =====
CHUNK_CONFIG = {
    "分块大小": 500,
    "分块重叠": 80,
}

EMBEDDING_MODELS = {
    "bge-small-zh-v1.5": {
        "hf_name": "BAAI/bge-small-zh-v1.5",
        "path": "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
        "dimension": 512,
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
    },
    "bge-base-zh-v1.5": {
        "hf_name": "BAAI/bge-base-zh-v1.5",
        "path": "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65",
        "dimension": 768,
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
    },
}
EMBEDDING_CONFIG = {
    "模型名称": "bge-base-zh-v1.5",
    "设备": "cpu",
    "归一化": True,
    "维度": 768,
}

# ===== FAISS（不用Milvus，省去网络I/O） =====
VECTOR_STORE_BACKEND = "faiss"
MILVUS_CONFIG = {
    "host": "127.0.0.1",
    "port": 19530,
    "collection_name": "rag_chunks",
    "metadata_path": os.path.join(DATA_DIR, "chunk_metadata.json"),
}

# ===== 优化2: 全文检索保持启用（速度快，114ms） =====
FULLTEXT_CONFIG = {
    "启用": True,
    "索引目录": os.path.join(DATA_DIR, "fulltext_index"),
    "支持布尔查询": True,
    "支持短语匹配": True,
    "支持模糊匹配": True,
    "多字段检索": ["text", "heading", "source_file"],
}

# ===== 优化3: 重排器从 CrossEncoder(2s) → TF-IDF(~10ms) =====
RERANKER_CONFIG = {
    "启用重排": True,
    "默认重排器": "tfidf",
    "重排数量": 5,
    "可用重排器": {
        "llm": {"启用": False, "说明": "太慢，已禁用"},
        "tfidf": {"启用": True, "说明": "轻量级TF-IDF重排，~10ms"},
        "crossencoder": {"启用": False, "说明": "太慢(bge-reranker-v2-m3 ~2s)，按需启用"},
    },
    "reranker_model_path": "/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3",
}

# ===== 优化4: 默认混合检索模式 =====
HYBRID_SEARCH_CONFIG = {
    "默认检索模式": "hybrid",
    "向量权重": 0.5,
    "全文权重": 0.5,
    "融合算法": "rrf",
    "rrf常数": 60,
}

# ===== 保留多轮对话 =====
CONVERSATION_CONFIG = {
    "启用多轮对话": True,
    "历史轮数": 3,
    "上下文窗口": 2000,
    "指代消解": True,
}

IMAGE_CONFIG = {
    "启用图像提取": True,
    "提取最小尺寸": (100, 100),
    "保存格式": "png",
    "输出目录": os.path.join(DATA_DIR, "extracted_images"),
    "最大图像数": 200,
    "图像质量": 95,
}

# ===== 优化5: top_k从8→5，加快检索和重排 =====
RETRIEVAL_CONFIG = {
    "检索数量": 5,
    "相关性阈值": 0.0,
    "查询扩展": True,
    "重排序数量": 5,
    "表格权重提升": 0.3,
    "列头匹配提升": 0.4,
    "表格标题匹配提升": 0.5,
    "图像权重提升": 0.3,
    "图像描述匹配提升": 0.4,
}

# ===== 优化6: 用 qwen-turbo（更快、更便宜）替代 qwen-plus =====
LLM_CONFIG = {
    "提供商": "openai",
    "模型": "qwen-turbo",
    "API地址": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "API密钥": "sk-your-api-key-here",
    "最大Token数": 512,
    "温度": 0.1,
    "超时": 10,
    "重试次数": 1,
}

KIMI_CONFIG = {
    "模型": "moonshot-v1-8k-vision-preview",
    "API地址": "https://api.moonshot.cn/v1",
    "API密钥": "sk-bZ5...fIKx",
    "最大Token数": 2048,
    "温度": 0.1,
    "超时": 30,
    "重试次数": 2,
}

UI_CONFIG = {
    "页面标题": "PDF智能问答系统（混合检索优化版）",
    "页面图标": "🔍",
    "布局": "wide",
    "支持语音": True,
    "支持多文档": True,
    "显示图像预览": True,
    "版本号": "v6.0",
}
