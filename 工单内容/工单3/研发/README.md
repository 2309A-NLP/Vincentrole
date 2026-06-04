# 📄 PDF智能问答系统

基于 RAG（检索增强生成）技术的 PDF 文档问答系统。

## 功能特点

- **PDF解析** - 自动提取PDF文本内容，支持超大文档（500+页）
- **语义检索** - 使用向量相似度高效检索相关文档片段
- **Query理解** - 自动识别问题意图和关键实体
- **LLM生成** - 基于检索结果生成准确、简洁的答案
- **对比分析** - RAG结果 vs 纯LLM结果对比，直观展示RAG优势
- **友好界面** - Streamlit Web界面，操作直观

## 项目结构

```
PDF-RAG-问答系统/
├── config.py                  # 系统配置
├── run.py                     # 启动脚本
├── requirements.txt           # 依赖库
├── pdf_parser/                # PDF解析模块
│   ├── __init__.py
│   ├── parser.py              # PDF文本提取
│   └── chunker.py             # 智能文本分块
├── knowledge_base/            # 知识库模块
│   ├── __init__.py
│   ├── embeddings.py          # 文本向量化
│   └── vector_store.py        # FAISS向量存储
├── qa_engine/                 # 问答引擎
│   ├── __init__.py
│   ├── query_understanding.py # Query理解
│   ├── retriever.py           # 语义检索器
│   ├── generator.py           # LLM生成器
│   └── orchestrator.py        # 系统主控制器
├── app/
│   └── ui.py                  # Streamlit界面
├── evaluation/
│   ├── test_questions.py      # 测试问题集
│   └── evaluate.py            # 系统评估器
├── docs/
│   ├── 技术文档.md             # 技术文档
│   └── 用户手册.md             # 用户手册
└── data/                      # 数据缓存目录
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
streamlit run app/ui.py

# 3. 浏览器打开 http://localhost:8501
```

## 配置说明

编辑 `config.py` 或通过界面配置：

- **LLM提供商**: OpenAI / DeepSeek / Ollama
- **Embedding模型**: 默认 BAAI/bge-base-zh-v1.5
- **检索参数**: Top-K、相关性阈值
- **分块参数**: 块大小、重叠长度

## 测试问题

系统内置了10个工单规定的测试问题，覆盖：
- 财务数据查询
- 公司基本信息
- 行业分析
- 风险因素分析
- 募资用途

## 技术栈

- **Embedding**: sentence-transformers + BGE
- **向量存储**: FAISS (IndexFlatIP)
- **PDF解析**: PyMuPDF / pypdf
- **Web界面**: Streamlit
- **LLM**: OpenAI / DeepSeek / Ollama

## 工单信息

- 工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
- 项目名称: RAG
- 创建时间: 2025年1月21日
