# -*- coding: utf-8 -*-
"""
知识库构建与数据摄入模块（Ingest Pipeline）

主要功能：
- 从本地文件（TXT / PDF）加载文档
- 从 Huatuo26M 医疗数据集加载并清洗数据
- 对文档进行文本切片（chunking）
- 批量向量化并写入 Milvus 向量数据库
"""

import os
import json
import logging

# ---------- 日志配置 ----------
from logging_config import setup_logging
setup_logging()

# ---------- 项目内部模块 ----------
from embeddings import embed_engine        # 文本向量化引擎
from pdf_parser import parse_pdf           # PDF 解析器
from storage import milvus_db              # Milvus 向量数据库接口
from text_chunking import split_text       # 文本分块工具

logger = logging.getLogger(__name__)

# ---------- 路径与常量配置 ----------
# 1. 本地知识库目录
DATA_DIR = "./knowledge_base"

# 2. Huatuo26M 医疗数据集路径（JSONL 格式）
HUATUO_DATA_PATH = "/Users/suwente/Desktop/medical_data/format_data.jsonl"


def load_documents():
    """
    文档加载器（Loader）

    功能：
    1. 加载本地知识库中的 TXT / PDF 文件
    2. 加载并清洗 Huatuo26M-Lite 医疗数据集（JSONL 格式）
    """
    documents = []

    # ========== 本地文件加载逻辑 ==========
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)

            # TXT 文件处理
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "source": filename,
                        "content": content
                    })

            # PDF 文件处理
            elif filename.lower().endswith(".pdf"):
                try:
                    result = parse_pdf(file_path)
                    documents.append({
                        "source": filename,
                        "content": result.content
                    })

                    logger.info(
                        "PDF parsed: filename=%s parser=%s pages=%s",
                        filename, result.parser, result.pages
                    )

                    for warning in result.warnings:
                        logger.warning(
                            "PDF parser warning: filename=%s warning=%s",
                            filename, warning
                        )

                except Exception as exc:
                    logger.exception("PDF parsing failed: filename=%s", filename)

    else:
        # 若目录不存在则创建
        os.makedirs(DATA_DIR)
        logger.info("Created knowledge base directory: path=%s", DATA_DIR)

    # ========== Huatuo26M 数据集加载逻辑 ==========
    if os.path.exists(HUATUO_DATA_PATH):
        logger.info("Loading medical dataset: path=%s", HUATUO_DATA_PATH)

        count = 0
        empty_count = 0
        MAX_SAMPLES = 100  # 限制读取条数，防止数据过大

        with open(HUATUO_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if count >= MAX_SAMPLES:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # ---------- 数据清洗：字段提取 ----------
                    question = ""
                    answer = ""

                    # 提取问题字段（白名单）
                    for k in ["instruction", "input", "question", "query"]:
                        if k in data and isinstance(data[k], str):
                            question += data[k] + " "

                    # 提取回答字段（白名单）
                    for k in ["output", "answer", "response"]:
                        if k in data and isinstance(data[k], str):
                            answer += data[k] + " "

                    # 拼接问题与答案
                    content = f"{question.strip()} {answer.strip()}"

                    # ---------- 数据清洗：文本规范化 ----------
                    content = " ".join(content.split())

                    # ---------- 数据清洗：质量过滤 ----------
                    if len(content) < 10:
                        empty_count += 1
                        continue

                    if content:
                        documents.append({
                            "source": "Huatuo26M-Clean",
                            "content": content
                        })
                        count += 1
                    else:
                        empty_count += 1

                except json.JSONDecodeError:
                    continue

        logger.info(
            "Medical dataset loaded: count=%s discarded_or_empty=%s",
            count, empty_count
        )

    else:
        logger.warning("Medical dataset not found: path=%s", HUATUO_DATA_PATH)

    return documents


def main():
    """
    知识库构建主流程（Entry Point）

    执行顺序：
    1. 加载文档
    2. 文本分块
    3. 批量向量化
    4. 写入 Milvus
    """
    logger.info("Knowledge base build started.")

    docs = load_documents()
    if not docs:
        logger.warning("No documents found. Check knowledge_base directory or dataset path.")
        return

    logger.info("Documents loaded: count=%s. Starting chunking.", len(docs))

    all_chunks = []
    all_metas = []

    # ---------- 文本分块 ----------
    for doc in docs:
        chunks = split_text(doc["content"])
        all_chunks.extend(chunks)
        all_metas.extend([doc["source"]] * len(chunks))

    if not all_chunks:
        logger.warning("Document contents are empty after chunking.")
        return

    logger.info("Text chunking completed: total_chunks=%s", len(all_chunks))

    # ---------- 批量向量化 & 入库 ----------
    BATCH_SIZE = 16  # 防止显存 / 内存溢出
    total_count = len(all_chunks)

    logger.info(
        "Batch embedding started: batch_size=%s total_chunks=%s",
        BATCH_SIZE, total_count
    )

    for i in range(0, total_count, BATCH_SIZE):
        batch_texts = all_chunks[i:i + BATCH_SIZE]
        batch_metas = all_metas[i:i + BATCH_SIZE]

        try:
            # 1. 向量化
            vectors = embed_engine.encode(batch_texts)

            # 2. 写入 Milvus
            milvus_db.insert_data(
                texts=batch_texts,
                vectors=vectors,
                metas=batch_metas
            )

            current = min(i + BATCH_SIZE, total_count)
            logger.info(
                "Ingest progress: current=%s total=%s percent=%.1f",
                current, total_count, (current / total_count) * 100
            )

        except Exception as e:
            logger.exception(
                "Batch ingest failed: start=%s batch_size=%s",
                i, BATCH_SIZE
            )
            break

    logger.info("Knowledge base build completed.")


if __name__ == "__main__":
    main()