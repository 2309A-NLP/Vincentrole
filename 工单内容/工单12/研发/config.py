"""
工单13 - LightRAG优化任务
配置文件 - v2: 使用Dashscope qwen-plus
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/suwente/Desktop/专高六学习资料/RAG 工单/附件"
PDF_FILES = [
    os.path.join(DATA_DIR, "招股说明书1.pdf"),
    os.path.join(DATA_DIR, "招股说明书2.pdf"),
]

# LightRAG配置
LIGHTRAG_WORKING_DIR = os.path.join(BASE_DIR, "output")

# Dashscope API配置 (qwen-plus, OpenAI兼容模式)
API_KEY = "sk-your-api-key-here"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"

# BGE embedding 本地模型路径
BGE_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5"
    "/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
)

# 测试问题
TEST_QUESTIONS = [
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

# Backward-compatible aliases for main.py
KIMI_API_KEY = API_KEY
KIMI_BASE_URL = API_URL
KIMI_MODEL = LLM_MODEL

# Real API key
REAL_KEY = "sk-your-api-key-here"
REAL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REAL_MODEL = "qwen-plus"
