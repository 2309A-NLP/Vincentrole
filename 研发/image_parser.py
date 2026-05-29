# -*- coding: utf-8 -*-
"""
图片解析模块（基于 OCR）

功能概述：
- 使用 Tesseract OCR 从图片中提取文字
- 适用于截图、扫描件、表格截图、票据/报告照片等包含文字的图片
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ImageParseResult:
    """
    图片解析结果的数据结构

    :param source: 原始图片文件路径
    :param content: OCR 识别出的文本内容
    :param parser: 使用的解析器名称（此处固定为 pytesseract）
    :param warnings: 解析过程中产生的警告信息列表
    """
    source: str
    content: str
    parser: str
    warnings: List[str]


def _clean_text(text: str) -> str:
    """
    对 OCR 原始文本进行清洗和规范化处理

    处理逻辑：
    1. 将空值转为空字符串，并移除空字符 '\x00'
    2. 合并多余的空格和制表符
    3. 将连续多个换行压缩为最多两个
    4. 去除首尾空白
    """
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_image(file_path: str) -> ImageParseResult:
    """
    解析图片中的文字内容（基于 OCR）

    适用场景：
    - 截图
    - 扫描件
    - 表格截图
    - 票据 / 报告照片等包含文字的图片

    :param file_path: 图片文件路径
    :return: ImageParseResult 对象
    """
    warnings = []

    # ---------- 1. 检查 Python 依赖 ----------
    try:
        import pytesseract
    except ImportError as exc:
        # 如果未安装 pytesseract，直接抛出明确错误
        raise RuntimeError(
            "图片解析缺少 Python 依赖 pytesseract，请先执行 pip install pytesseract。"
        ) from exc

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        # 如果未安装 Pillow，直接抛出明确错误
        raise RuntimeError(
            "图片解析缺少 Python 依赖 pillow，请先执行 pip install pillow。"
        ) from exc

    # ---------- 2. 图片读取与 OCR ----------
    try:
        # 使用 PIL 打开图片
        with Image.open(file_path) as image:
            # 修正图片 EXIF 方向信息（如手机拍摄的照片）
            normalized = ImageOps.exif_transpose(image).convert("RGB")

            # 使用 Tesseract 进行 OCR，支持中英文
            text = pytesseract.image_to_string(normalized, lang="chi_sim+eng")

    except Exception as exc:
        # 针对系统未安装 Tesseract 的情况给出明确提示
        if (
            "tesseract is not installed" in str(exc).lower()
            or "no such file or directory" in str(exc).lower()
        ):
            raise RuntimeError(
                "图片解析缺少系统 tesseract。"
                "macOS 可执行：brew install tesseract tesseract-lang。"
            ) from exc

        # 其他异常统一包装后抛出
        raise RuntimeError(f"图片 OCR 失败：{exc}") from exc

    # ---------- 3. 文本清洗与结果封装 ----------
    content = _clean_text(text)

    # 如果 OCR 结果为空，给出提示性警告
    if not content:
        warnings.append(
            "OCR 未识别到有效文字。"
            "若图片主要是场景/物体而非文字，需要接入多模态视觉模型。"
        )

    # 返回结构化解析结果
    return ImageParseResult(
        source=file_path,
        content=content,
        parser="pytesseract",
        warnings=warnings,
    )