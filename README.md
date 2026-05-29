# AURA MODE — 多角色 AI 对话工作台

把健康梳理、情绪陪伴、学习规划、职场表达和文案策划放进同一个入口，让用户按场景切换更合适的 AI 协作角色。

## 功能特性

- **多角色切换** — 5 个精心设计的 AI 角色，覆盖日常高频场景
  - 🌿 健康顾问 — 把零散感受整理成清晰线索
  - ✨ 灵感搭子 — 轻盈、有回应感的情绪陪伴
  - 💼 职场顾问 — 沟通、简历、面试、职业决策
  - 📚 学习搭子 — 拆目标、排节奏、稳住学习状态
  - 🪄 文案策划师 — 标题优化、表达升级、创意输出
- **RAG 知识库** — 基于 Milvus 向量数据库 + BGE-m3 嵌入，支持本地文档检索增强
- **多模态知识摄入** — 支持 PDF、图片（OCR）、TXT 文件上传构建知识库
- **JWT 用户认证** — 注册 / 登录 / Token 鉴权
- **Streamlit 前端** — 清爽对话式 UI，所见即所得
- **优雅部署** — 一键部署脚本 + Docker Compose 基础设施

## 技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI + Uvicorn |
| 前端界面 | Streamlit |
| 向量数据库 | Milvus |
| 嵌入模型 | BGE-m3（本地） |
| 重排序 | BGE-reranker-v2-m3（本地） |
| 大语言模型 | DeepSeek API / Ollama 本地模型 |
| 缓存 | Redis |
| 认证 | JWT (python-jose + passlib) |
| OCR | Tesseract / PaddleOCR |
| 文档解析 | pdfplumber / PyMuPDF / PyPDF |
| 容器化 | Docker Compose（Milvus + Redis） |

## 快速开始

### 前置要求

- Python 3.10+
- Docker & Docker Compose（用于启动 Milvus + Redis）
- Tesseract OCR（可选，用于图片文字识别）

### 一键部署

```bash
chmod +x 部署/deploy.sh
./部署/deploy.sh
```

脚本会自动完成：环境检查 → 创建虚拟环境 → 安装依赖 → 启动 Docker 服务 → 启动后端 → 启动前端。

### 手动部署

1. 启动基础设施：

```bash
docker compose -f 研发/docker-compose.yml up -d
```

2. 安装依赖：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r 研发/requirements.txt
```

3. 启动后端（端口 8080）：

```bash
cd 研发
python main.py
```

4. 启动前端（端口 8501）：

```bash
cd 研发
streamlit run app.py
```

### 配置

编辑 `研发/config.py` 可配置：

- Milvus 连接地址
- Redis 连接信息
- LLM API（默认 DeepSeek，可切换至 Ollama 本地模型）
- 本地模型路径（BGE-m3 / BGE-reranker）

> 建议将 API Key 通过环境变量传入，避免硬编码。

## 项目结构

```
├── 研发/                     # 主应用
│   ├── main.py              # FastAPI 后端入口
│   ├── app.py               # Streamlit 前端
│   ├── config.py            # 全局配置
│   ├── role_catalog.py      # 角色定义与提示词
│   ├── rag_core.py          # RAG 检索核心
│   ├── embeddings.py        # 向量化引擎
│   ├── hybrid_retrieval.py  # 混合检索（向量 + BM25）
│   ├── storage.py           # Milvus 数据库接口
│   ├── persistence.py       # 持久化存储
│   ├── ingest.py            # 知识库摄入管道
│   ├── text_chunking.py     # 文本分块
│   ├── pdf_parser.py        # PDF 解析
│   ├── image_parser.py      # 图片 OCR 解析
│   ├── logging_config.py    # 日志配置
│   ├── requirements.txt     # Python 依赖
│   └── docker-compose.yml   # Milvus + Redis 容器配置
├── 测试/                     # JMeter 性能测试
│   ├── generate_jmx.py      # JMeter 测试计划生成
│   ├── roleplay-test-plan.jmx
│   ├── run-test.sh          # 测试运行脚本
│   └── analyze_results.py   # 结果分析
├── 设计/                     # 架构设计文档
└── 部署/                     # 部署脚本
    ├── deploy.sh            # 一键部署（检查→安装→启动→停止）
    └── docker-compose.yml
```

## 性能测试

`测试/` 目录包含完整的 JMeter 测试套件，支持：

- 对话接口并发压测
- 角色切换场景模拟
- 结果自动分析与报告生成

```bash
cd 测试
bash run-test.sh
```

## 许可证

MIT
