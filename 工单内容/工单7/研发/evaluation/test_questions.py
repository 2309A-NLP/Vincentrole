"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
测试问题集 - 14个工单规定的测试问题（包含两份招股说明书）
"""

# 工单规定的测试问题
# 武汉兴图新科电子股份有限公司（招股说明书1.pdf） - 10个问题
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

# 武汉力源信息技术股份有限公司（招股说明书2.pdf） - 4个新增问题
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
]

# 全部测试问题（合并）
TEST_QUESTIONS = TEST_QUESTIONS_XINGTU + TEST_QUESTIONS_LIYUAN

# 英文测试问题
EN_TEST_QUESTIONS = [
    {"id": 260, "question": "What was the military revenue of Wuhan Xingtu Xinke Electronics during the reporting period?"},
    {"id": 95, "question": "What technology standard did Wuhan Xingtu Xinke Electronics participate in formulating?"},
    {"id": 543, "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics?"},
    {"id": 531, "question": "Who is the legal representative of Wuhan Xingtu Xinke Electronics?"},
    {"id": 1, "question": "How many shares will Wuhan Liyuan Information Technology issue, and what percentage of total shares will it represent?"},
    {"id": 2, "question": "What projects will Wuhan Liyuan Information Technology invest in with the raised funds?"},
    {"id": 3, "question": "Who are the related parties with controlling relationships of Wuhan Liyuan Information Technology?"},
    {"id": 4, "question": "What are the related party enterprises without controlling relationships of Wuhan Liyuan Information Technology?"},
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


def get_questions_by_source(source_pdf: str) -> list:
    """按PDF来源筛选问题"""
    return [q for q in TEST_QUESTIONS if q.get("source_pdf") == source_pdf]
