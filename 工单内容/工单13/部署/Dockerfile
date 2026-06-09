FROM python:3.11-slim

WORKDIR /app

# 将所有 wheel 复制到镜像中
COPY wheels2/ /tmp/wheels/

# 从本地安装所有包
RUN pip install --no-cache-dir --find-links /tmp/wheels \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    torch streamlit pymupdf sentence-transformers faiss-cpu whoosh jieba \
    numpy requests openai \
    && rm -rf /tmp/wheels

# 模型文件
COPY models/bge-small-zh-v1.5/ /app/models/bge-small-zh-v1.5/

# 应用代码（保持目录结构）
COPY main.py run.py /app/
COPY config_docker.py /app/config.py
COPY app/ /app/app/
COPY pdf_parser/ /app/pdf_parser/
COPY knowledge_base/ /app/knowledge_base/
COPY qa_engine/ /app/qa_engine/
COPY evaluation/ /app/evaluation/
COPY data/ /app/data/

EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
