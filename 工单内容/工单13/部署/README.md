# 部署目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile` | 容器构建文件（基于 python:3.11-slim） |
| `docker-compose.yml` | Docker Compose 编排配置 |
| `config_docker.py` | 容器内使用的配置文件（API Key 通过环境变量注入） |
| `部署文档.md` | 完整部署手册（含架构图、网络配置、数据管理） |

## 构建说明

**注意：** 国内网络无法直接拉取 Docker Hub，建议在云服务器上构建。

```bash
# 构建
docker build -t w13-finance-qa:latest .

# 运行
docker run -d --name finance-qa -p 8501:8501 \
  -e DASHSCOPE_API_KEY="sk-your-api-key-here" \
  w13-finance-qa:latest

# 访问
open http://localhost:8501
```

## 本地运行（无需 Docker）

```bash
cd ../研发
streamlit run app/ui.py --server.port 8501
```

*工单编号: 人工智能NLP-RAG-金融问答系统部署*
