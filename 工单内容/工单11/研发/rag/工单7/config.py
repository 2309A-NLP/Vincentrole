"""
工单编号: 人工智能NLP-RAG-功能测试及评估
系统配置 - 支持多种文件格式（PDF/TXT）+ 混合检索 + 多种重排算法 + 评估功能
"""

import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/suwente/Desktop/专高六学习资料/RAG 工单/附件/ccf_competition 2"

# 支持的文件格式（只解析PDF）
SUPPORTED_FORMATS = ['.pdf']

# 获取data目录下所有支持的文件
def get_supported_files(directory=None):
    """获取指定目录下所有支持的文件（PDF/TXT）"""
    if directory is None:
        directory = DATA_DIR
    
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
                files.append(os.path.join(root, filename))
    return sorted(files)

# 默认加载的文件（自动扫描data目录）
DEFAULT_FILES = get_supported_files()

# ============================================================
# 文件解析配置
# ============================================================
FILE_CONFIG = {
    "PDF引擎": "pymupdf",
    "TXT编码": "utf-8",
    "TXT编码回退": ["gbk", "gb2312", "latin-1"],
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

EMBEDDING_CONFIG = {
    "模型名称": "bge-base-zh-v1.5",
    "设备": "cpu",
    "归一化": True,
    "维度": 768,
}

# ============================================================
# 向量存储配置 - Milvus
# ============================================================
VECTOR_STORE_BACKEND = "milvus"

MILVUS_CONFIG = {
    "host": "127.0.0.1",
    "port": 19530,
    "collection_name": "rag_chunks",
    "metadata_path": os.path.join(DATA_DIR, "chunk_metadata.json"),
}

# ============================================================
# 全文检索配置
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
# 重排器配置
# ============================================================
RERANKER_CONFIG = {
    "启用重排": True,
    "默认重排器": "crossencoder",
    "重排数量": 10,
    "可用重排器": {
        "llm": {"启用": True, "说明": "基于Kimi的LLM重排器"},
        "tfidf": {"启用": True, "说明": "基于TF-IDF的重排器"},
        "crossencoder": {"启用": True, "说明": "基于bge-reranker-v2-m3的本地重排器"},
    },
    "reranker_model_path": "/Users/suwente/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3",
}

# ============================================================
# 混合检索配置
# ============================================================
HYBRID_SEARCH_CONFIG = {
    "默认检索模式": "hybrid",
    "向量权重": 0.5,
    "全文权重": 0.5,
    "融合算法": "rrf",
    "rrf常数": 60,
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
}

# ============================================================
# 检索配置
# ============================================================
RETRIEVAL_CONFIG = {
    "检索数量": 8,
    "相关性阈值": 0.0,
    "查询扩展": True,
    "重排序数量": 10,
}

# ============================================================
# LLM配置
# ============================================================
LLM_CONFIG = {
    "提供商": "deepseek",
    "模型": "deepseek-v4-flash",
    "API地址": "https://api.deepseek.com/v1",
    "API密钥": "sk-your-api-key-here",
    "最大Token数": 1024,
    "温度": 0.1,
    "超时": 15,
    "重试次数": 2,
}

# ============================================================
# Kimi配置
# ============================================================
KIMI_CONFIG = {
    "模型": "moonshot-v1-8k-vision-preview",
    "API地址": "https://api.moonshot.cn/v1",
    "API密钥": "sk-your-api-key-here",
    "最大Token数": 2048,
    "温度": 0.1,
    "超时": 30,
    "重试次数": 2,
}

# ============================================================
# 评估配置
# ============================================================
EVALUATION_CONFIG = {
    "启用评估": True,
    "评估指标": ["recall", "precision", "f1", "mrr", "ndcg"],
    "样本问题路径": os.path.join(BASE_DIR, "sample_questions.pdf"),
    "评估输出目录": os.path.join(DATA_DIR, "evaluation_results"),
}

# ============================================================
# UI配置
# ============================================================
UI_CONFIG = {
    "页面标题": "PDF智能问答RAG系统",
    "页面图标": "📊",
    "布局": "wide",
    "支持语音": True,
    "支持多文档": True,
    "显示图像预览": True,
    "显示评估面板": True,
    "版本号": "v7.0",
}
