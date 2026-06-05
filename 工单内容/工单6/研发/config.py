"""
工单编号: 人工智能NLP-RAG-混合检索任务
系统配置 - 支持多种检索模式（向量检索+全文检索+混合检索）+ 多模型嵌入 + 3种重排算法
"""

import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 默认PDF路径 - 支持多文档
DEFAULT_PDFS = [
    os.path.join(DATA_DIR, "招股说明书1.pdf"),
    os.path.join(DATA_DIR, "招股说明书2.pdf"),
]

# ============================================================
# PDF解析配置
# ============================================================
PDF_CONFIG = {
    "使用引擎": "pymupdf",
}

# ============================================================
# 分块配置
# ============================================================
CHUNK_CONFIG = {
    "分块大小": 800,
    "分块重叠": 150,
}

# ============================================================
# Embedding配置 - 支持多模型切换
# ============================================================
# 可选的嵌入模型
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
    "m3e-base": {
        "hf_name": "moka-ai/m3e-base",
        "path": "/Users/suwente/Desktop/m3e-base",
        "dimension": 768,
        "query_prefix": "",
    },
    "bge-m3": {
        "hf_name": "BAAI/bge-m3",
        "path": "/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-m3",
        "dimension": 1024,
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
    },
}

# 当前使用的嵌入模型（修改此项切换模型）
EMBEDDING_CONFIG = {
    "模型名称": "bge-base-zh-v1.5",        # EMBEDDING_MODELS中的key
    "设备": "cpu",
    "归一化": True,
    "维度": 768,
}

# ============================================================
# 向量存储配置 - Milvus
# ============================================================
VECTOR_STORE_BACKEND = "milvus"

# Milvus配置
MILVUS_CONFIG = {
    "host": "127.0.0.1",
    "port": 19530,
    "collection_name": "rag_chunks",
    "metadata_path": os.path.join(DATA_DIR, "chunk_metadata.json"),
}

# ============================================================
# 全文检索配置（工单6新增 - Whoosh倒排索引）
# ============================================================
FULLTEXT_CONFIG = {
    "启用": True,
    "索引目录": os.path.join(DATA_DIR, "fulltext_index"),
    "支持布尔查询": True,
    "支持短语匹配": True,
    "支持模糊匹配": True,
    "多字段检索": ["text", "heading", "source_file"],
}

# ============================================================
# 重排器配置（工单6新增 - 3种重排算法）
# ============================================================
RERANKER_CONFIG = {
    "启用重排": True,
    "默认重排器": "crossencoder",  # llm / tfidf / adaptive / crossencoder
    "重排数量": 10,

    # 3种重排器
    "可用重排器": {
        "llm": {
            "启用": True,
            "说明": "基于LLM的重排器，使用Kimi评估chunk与query的相关性",
        },
        "tfidf": {
            "启用": True,
            "说明": "基于TF-IDF的重排器，计算chunk文本与query的TF-IDF余弦相似度",
        },
        "crossencoder": {
            "启用": True,
            "说明": "基于CrossEncoder(bge-reranker-v2-m3)的本地重排器，无需联网",
        },
    },

    # bge-reranker-v2-m3 本地路径（ModelScope下载）
    "reranker_model_path": "/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3",
}

# ============================================================
# 混合检索配置（工单6新增）
# ============================================================
HYBRID_SEARCH_CONFIG = {
    "默认检索模式": "hybrid",          # vector / fulltext / hybrid
    "向量权重": 0.5,                   # hybrid模式下向量检索权重
    "全文权重": 0.5,                   # hybrid模式下全文检索权重
    "融合算法": "rrf",                # weighted_avg / rrf / vote
    "rrf常数": 60,                    # RRF融合常数k
}

# ============================================================
# 多轮对话配置
# ============================================================
CONVERSATION_CONFIG = {
    "启用多轮对话": True,
    "历史轮数": 3,
    "上下文窗口": 2000,
    "指代消解": True,
}

# ============================================================
# 图像处理配置
# ============================================================
IMAGE_CONFIG = {
    "启用图像提取": True,
    "提取最小尺寸": (100, 100),
    "保存格式": "png",
    "输出目录": os.path.join(DATA_DIR, "extracted_images"),
    "最大图像数": 200,
    "图像质量": 95,
    "多模态模型": "clip_vit_b_32",
    "模型路径": "",
    "可用设备": "cpu",
    "图像描述方式": "提取上下文文本",
    "上下文窗口": 1500,
    "图像检索权重": 0.3,
    "图像指令命中提升": 0.4,
    "检索图像上限": 3,
}

# ============================================================
# 检索配置
# ============================================================
RETRIEVAL_CONFIG = {
    "检索数量": 8,
    "相关性阈值": 0.0,
    "查询扩展": True,
    "重排序数量": 10,
    "表格权重提升": 0.3,
    "列头匹配提升": 0.4,
    "表格标题匹配提升": 0.5,
    "图像权重提升": 0.3,
    "图像描述匹配提升": 0.4,
}

# ============================================================
# LLM配置
# ============================================================
LLM_CONFIG = {
    "提供商": "deepseek",
    "模型": "deepseek-v4-flash",
    "API地址": "https://api.deepseek.com/v1",
    "API密钥": "sk-13cd4f0504954bc19c24b7f6dd2f6164",
    "最大Token数": 1024,
    "温度": 0.1,
    "超时": 15,
    "重试次数": 2,
}

# ============================================================
# Kimi配置（多模态回退）
# ============================================================
KIMI_CONFIG = {
    "模型": "moonshot-v1-8k-vision-preview",
    "API地址": "https://api.moonshot.cn/v1",
    "API密钥": "sk-bZ5OnuinbjLgqwvByNbaoxprV1zZAbXprHdrEULdwrX8fIKx",
    "最大Token数": 2048,
    "温度": 0.1,
    "超时": 30,
    "重试次数": 2,
}

# ============================================================
# UI配置（工单6：混合检索版）
# ============================================================
UI_CONFIG = {
    "页面标题": "PDF智能问答系统（混合检索优化版）",
    "页面图标": "🔍",
    "布局": "wide",
    "支持语音": True,
    "支持多文档": True,
    "显示图像预览": True,
    "版本号": "v6.0",
}
