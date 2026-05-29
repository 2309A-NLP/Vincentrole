# -*- coding: utf-8 -*-
"""
文本分块模块（Text Chunking）

功能概述：
- 将长文本切分为适合大模型处理的片段（chunks）
- 尽量按段落和句子边界切分，保留语义完整性
- 支持重叠（overlap）以保持上下文连贯性
"""

import logging
import re

logger = logging.getLogger(__name__)


def split_text(text, chunk_size=700, overlap=100):
    """
    按段落和句子边界对文本进行切片

    设计目标：
    - 避免生硬地截断句子
    - 保留复杂文档（如 PDF 解析结果）的语义结构
    - 通过 overlap 减少上下文断裂

    :param text: 原始文本
    :param chunk_size: 每个 chunk 的最大字符长度
    :param overlap: 相邻 chunk 的重叠字符数
    :return: 分块后的文本列表
    """
    original_length = len(text or "")
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n?(【第 \d+ 页】)\n?", r"\n\n\1\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 空文本直接返回
    if not text:
        logger.debug("Text chunking skipped because input is empty.")
        return []

    chunks = []
    current = ""

    # 分割依据：中文/英文句子结束符 + 换行
    parts = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        if any(line.lstrip().startswith("|") for line in lines):
            parts.append(block)
            continue

        parts.extend(
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s+|\n+", block)
            if part.strip()
        )

    for part in parts:
        # 超长句子单独处理
        if len(part) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(part):
                chunks.append(part[start:start + chunk_size])
                start += chunk_size - overlap
            continue

        # 尝试合并当前片段
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # 保留尾部重叠内容
            tail = current[-overlap:] if overlap and len(current) > overlap else current
            current = f"{tail} {part}".strip()

    # 追加剩余内容
    if current:
        chunks.append(current)

    logger.debug(
        "Text chunking completed: original_length=%s cleaned_length=%s parts=%s chunks=%s chunk_size=%s overlap=%s",
        original_length,
        len(text),
        len(parts),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
