FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --find-links /tmp/wheels \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    torch streamlit pymupdf sentence-transformers faiss-cpu whoosh jieba \
    numpy requests openai && rm -rf /tmp/wheels

# 模型和数据
COPY models/ /app/models/
COPY data/ /app/data/

# 应用代码
COPY config.py app.py requirements.txt /app/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
