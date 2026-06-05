"""
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
测试问题集 - 15个工单规定的测试问题（含两份招股说明书 + 2个新增图像问题）
"""

# ============================================================
# 武汉兴图新科电子股份有限公司（招股说明书1.pdf） - 10个问题
# ============================================================
TEST_QUESTIONS_XINGTU = [
    {
        "id": 260,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "category": "财务数据",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 95,
        "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        "category": "公司信息",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 33,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
        "category": "财务数据",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 34,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
        "category": "行业分析",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 957,
        "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
        "category": "公司信息",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 793,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
        "category": "行业分析",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 795,
        "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
        "category": "公司信息",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 543,
        "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
        "category": "公司信息",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 531,
        "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
        "category": "公司信息",
        "source_pdf": "招股说明书1.pdf",
    },
    {
        "id": 207,
        "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
        "category": "募资用途",
        "source_pdf": "招股说明书1.pdf",
    },
]

# ============================================================
# 武汉力源信息技术股份有限公司（招股说明书2.pdf） - 5个问题（含2个新增图像问题）
# ============================================================
TEST_QUESTIONS_LIYUAN = [
    {
        "id": 1,
        "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
        "category": "股本结构",
        "source_pdf": "招股说明书2.pdf",
    },
    {
        "id": 2,
        "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
        "category": "募投项目",
        "source_pdf": "招股说明书2.pdf",
    },
    {
        "id": 3,
        "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
        "category": "关联方",
        "source_pdf": "招股说明书2.pdf",
    },
    {
        "id": 4,
        "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
        "category": "关联方",
        "source_pdf": "招股说明书2.pdf",
    },
    {
        "id": 5,
        "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
        "category": "图像问题",
        "source_pdf": "招股说明书2.pdf",
        "image_type": "组织结构图",
    },
    {
        "id": 6,
        "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
        "category": "图像问题",
        "source_pdf": "招股说明书2.pdf",
        "image_type": "IC市场应用结构与增长图",
    },
]

# 全部测试问题（合并）
TEST_QUESTIONS = TEST_QUESTIONS_XINGTU + TEST_QUESTIONS_LIYUAN


def get_all_questions() -> list:
    """获取全部测试问题"""
    return TEST_QUESTIONS


def get_questions_by_category(category: str) -> list:
    """按类别筛选问题"""
    return [q for q in TEST_QUESTIONS if q["category"] == category]


def get_questions_by_source(source_pdf: str) -> list:
    """按PDF来源筛选问题"""
    return [q for q in TEST_QUESTIONS if q.get("source_pdf") == source_pdf]


def get_image_questions() -> list:
    """获取所有图像相关问题"""
    return [q for q in TEST_QUESTIONS if q.get("image_type") is not None]
