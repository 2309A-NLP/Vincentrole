"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
测试问题集 - 工单规定的10个测试问题
"""

# 工单规定的测试问题（基于武汉兴图新科电子股份有限公司的招股说明书）
TEST_QUESTIONS = [
    {
        "id": 260,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "category": "财务数据",
    },
    {
        "id": 95,
        "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        "category": "公司信息",
    },
    {
        "id": 33,
        "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
        "category": "财务数据",
    },
    {
        "id": 34,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
        "category": "行业分析",
    },
    {
        "id": 957,
        "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
        "category": "公司信息",
    },
    {
        "id": 793,
        "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
        "category": "行业分析",
    },
    {
        "id": 795,
        "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
        "category": "公司信息",
    },
    {
        "id": 543,
        "question": "武汉兴图新科电子股份有限公司注册资本是多少？",
        "category": "公司信息",
    },
    {
        "id": 531,
        "question": "武汉兴图新科电子股份有限公司法定代表人是谁？",
        "category": "公司信息",
    },
    {
        "id": 207,
        "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
        "category": "募资用途",
    },
]

# 额外的扩展测试问题
EXTRA_QUESTIONS = [
    {
        "id": "extra_1",
        "question": "武汉兴图新科电子股份有限公司的主要客户是谁？",
        "category": "公司信息",
    },
    {
        "id": "extra_2",
        "question": "本次发行的保荐机构是哪家证券公司？",
        "category": "公司信息",
    },
    {
        "id": "extra_3",
        "question": "报告期内公司面临的主要风险有哪些？",
        "category": "风险因素",
    },
    {
        "id": "extra_4",
        "question": "公司2018年度的营业收入和净利润分别是多少？",
        "category": "财务数据",
    },
]


def get_all_questions() -> list:
    """获取全部测试问题"""
    return TEST_QUESTIONS + EXTRA_QUESTIONS


def get_questions_by_category(category: str) -> list:
    """按类别筛选问题"""
    return [q for q in TEST_QUESTIONS if q["category"] == category]
