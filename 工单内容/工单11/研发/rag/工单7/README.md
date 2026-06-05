# 工单7 - RAG功能测试及评估系统

## 工单编号
人工智能NLP-RAG-功能测试及评估

## 功能概述
基于工单6的混合检索RAG系统，扩展支持多种文件格式（PDF/TXT），并添加RAG评估功能。

## 主要特性
1. **多格式文件支持** - 支持PDF和TXT文件解析
2. **混合检索** - 向量检索 + 全文检索（Whoosh倒排索引）
3. **多重排算法** - TF-IDF / LLM(Kimi) / CrossEncoder(bge-reranker-v2-m3)
4. **多轮对话** - 上下文维护 + 指代消解
5. **RAG评估** - 自动评估检索效果，生成评估报告

## 项目结构
```
工单7/
├── config.py                 # 配置文件
├── file_parser.py           # 多格式文件解析器（PDF/TXT）
├── evaluation.py            # RAG评估模块
├── test_evaluation.py       # 评估测试脚本
├── test_quick.py            # 快速功能测试
├── sample_questions.pdf     # 测试问题集
├── app/
│   └── ui.py               # Streamlit前端界面
├── qa_engine/
│   ├── orchestrator.py     # RAG系统主控制器
│   ├── retriever.py        # 检索器
│   ├── generator.py        # LLM生成器
│   └── conversation.py     # 多轮对话管理
├── knowledge_base/
│   ├── embeddings.py       # 嵌入模型
│   ├── milvus_store.py     # Milvus向量存储
│   ├── fulltext_store.py   # Whoosh全文索引
│   └── reranker.py         # 重排算法
├── pdf_parser/
│   ├── parser.py           # PDF解析器
│   ├── chunker.py          # 文本分块器
│   └── image_extractor.py  # 图像提取器
└── data/
    └── ccf_competition/     # 测试数据
        ├── pdf/             # PDF文件
        └── txt/             # TXT文件
```

## 快速开始

### 1. 启动前端界面
```bash
cd ~/Desktop/PDF智能问答RAG项目合集/工单7
streamlit run app/ui.py
```

### 2. 运行快速测试
```bash
cd ~/Desktop/PDF智能问答RAG项目合集/工单7
python3 test_quick.py
```

### 3. 运行RAG评估
```bash
cd ~/Desktop/PDF智能问答RAG项目合集/工单7
python3 test_evaluation.py --questions 10
```

## 使用说明

### 文件上传
1. 在前端界面点击"上传文档"
2. 支持PDF和TXT格式
3. 可以从data目录选择已有文件

### 检索模式
- **向量检索** - 基于语义相似度
- **全文检索** - 基于关键词匹配
- **混合检索** - 结合向量和全文检索（推荐）

### 重排算法
- **CrossEncoder** - 本地模型，无需联网（默认）
- **TF-IDF** - 快速，适合简单场景
- **LLM** - 使用Kimi API，效果最好但需要联网

## 评估功能

### 评估指标
- **Recall** - 召回率
- **Precision** - 精确率
- **F1** - F1分数
- **MRR** - 平均倒数排名
- **NDCG** - 归一化折损累积增益

### 运行评估
```bash
python3 test_evaluation.py --questions 10 --output ./data/evaluation_results
```

### 查看结果
评估结果会保存到 `data/evaluation_results/` 目录：
- `evaluation_results_YYYYMMDD_HHMMSS.json` - JSON格式结果
- `evaluation_report_YYYYMMDD_HHMMSS.md` - Markdown格式报告

## 配置说明

### 文件格式支持
在 `config.py` 中配置：
```python
SUPPORTED_FORMATS = ['.pdf', '.txt']
```

### 嵌入模型
支持多种嵌入模型切换：
- bge-small-zh-v1.5 (512维)
- bge-base-zh-v1.5 (768维) ← 默认
- m3e-base (768维)
- bge-m3 (1024维)

### 检索参数
```python
RETRIEVAL_CONFIG = {
    "检索数量": 8,
    "相关性阈值": 0.0,
    "查询扩展": True,
    "重排序数量": 10,
}
```

## 测试数据
- **PDF文件**: 9个银行年度报告PDF
- **TXT文件**: 9个对应的TXT文件
- **测试问题**: 10个来自sample_questions.pdf的问题

## 已知问题
1. 缓存加载逻辑需要优化
2. 大量文件加载时初始化时间较长
3. 需要更多测试用例验证

## 下一步优化
1. 优化分块策略
2. 添加更多评估指标
3. 支持更多文件格式（Word、Excel等）
4. 优化检索性能

---
**工单编号**: 人工智能NLP-RAG-功能测试及评估
**版本**: v7.0
**创建时间**: 2025年1月
