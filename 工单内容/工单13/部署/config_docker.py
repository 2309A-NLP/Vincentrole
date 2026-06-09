"""
工单编号: 人工智能NLP-RAG-金融问答系统部署
Docker 容器配置文件 - 生产环境用通义千问 API（需通过环境变量传入）
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_PDFS = [
    os.path.join(DATA_DIR, "招股说明书1.pdf"),
]

PDF_CONFIG = {"使用引擎": "pymupdf"}

CHUNK_CONFIG = {"分块大小": 500, "分块重叠": 80}

EMBEDDING_MODELS = {
    "bge-small-zh-v1.5": {
        "hf_name": "BAAI/bge-small-zh-v1.5",
        "path": "/app/models/bge-small-zh-v1.5",
        "dimension": 512,
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
    },
}
EMBEDDING_CONFIG = {
    "模型名称": "bge-small-zh-v1.5",
    "设备": "cpu",
    "归一化": True,
    "维度": 512,
}

VECTOR_STORE_BACKEND = "faiss"
MILVUS_CONFIG = {}
VECTOR_STORE_CONFIG = {
    "索引路径": os.path.join(DATA_DIR, "vector_index.faiss"),
    "元数据路径": os.path.join(DATA_DIR, "chunk_metadata.json"),
}

FULLTEXT_CONFIG = {
    "启用": True,
    "索引目录": os.path.join(DATA_DIR, "fulltext_index"),
    "支持布尔查询": True,
    "支持短语匹配": True,
    "支持模糊匹配": True,
    "多字段检索": ["text", "heading", "source_file"],
}

RERANKER_CONFIG = {
    "启用重排": True,
    "默认重排器": "tfidf",
    "重排数量": 5,
    "可用重排器": {
        "llm": {"启用": False},
        "tfidf": {"启用": True},
        "crossencoder": {"启用": False},
    },
    "reranker_model_path": "",
}

HYBRID_SEARCH_CONFIG = {
    "默认检索模式": "hybrid",
    "向量权重": 0.5,
    "全文权重": 0.5,
    "融合算法": "rrf",
    "rrf常数": 60,
}

CONVERSATION_CONFIG = {
    "启用多轮对话": True,
    "历史轮数": 3,
    "上下文窗口": 2000,
    "指代消解": True,
}

IMAGE_CONFIG = {
    "启用图像提取": False,
}

RETRIEVAL_CONFIG = {
    "检索数量": 5,
    "相关性阈值": 0.0,
    "查询扩展": True,
    "重排序数量": 5,
    "表格权重提升": 0.3,
}

# LLM配置 - API密钥通过环境变量 DASHSCOPE_API_KEY 传入
LLM_CONFIG = {
    "提供商": "openai",
    "模型": os.getenv("LLM_MODEL", "qwen-turbo"),
    "API地址": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "API密钥": os.getenv("DASHSCOPE_API_KEY", ""),
    "最大Token数": 512,
    "温度": 0.1,
    "超时": 10,
    "重试次数": 1,
}

KIMI_CONFIG = {}

UI_CONFIG = {
    "页面标题": "金融问答系统（工单13优化版）",
    "页面图标": "⚡",
    "布局": "wide",
    "支持语音": False,
    "支持多文档": True,
    "显示图像预览": False,
    "版本号": "v13.0-docker",
}
