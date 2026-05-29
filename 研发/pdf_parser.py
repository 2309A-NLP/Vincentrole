# -*- coding: utf-8 -*-
"""
PDF 解析模块（多引擎解析器）

功能概述：
- 支持多种 PDF 解析引擎（pdfplumber / PyMuPDF / pypdf）
- 支持表格提取与 OCR（扫描版 PDF）
- 自动降级与兜底，保证最大兼容性
- 输出结构化解析结果（PDFParseResult）
"""

import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PDFParseResult:
    """
    PDF 解析结果的数据结构

    :param source: 原始 PDF 文件路径
    :param content: 解析出的全部文本内容
    :param parser: 实际使用的解析器名称
    :param pages: PDF 页数
    :param warnings: 解析过程中产生的警告信息
    """
    source: str
    content: str
    parser: str
    pages: int
    warnings: List[str]


def _clean_text(text: str) -> str:
    """
    对原始提取文本进行清洗和规范化

    处理逻辑：
    - 替换空字符 '\x00'
    - 合并多余空格和制表符
    - 压缩多余换行
    - 去除首尾空白
    """
    text = (text or "").replace("\x00", " ")
    text = text.replace("\u3000", " ")
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_cell(cell: object) -> str:
    """
    清洗表格单元格，保留单元格内部换行的语义。
    """
    value = _clean_text(str(cell or ""))
    return re.sub(r"\s*\n\s*", "<br>", value)


def _format_table(table: List[List[object]]) -> str:
    """
    将 pdfplumber 提取的表格转换为 Markdown，保留列边界。
    """
    rows = []
    for row in table or []:
        cells = [_clean_cell(cell) for cell in row or []]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(normalized_rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized_rows[1:])
    return "\n".join(lines)


def _looks_like_financial_table_line(line: str) -> bool:
    """
    识别研报中常见的估值/财务预测表行，便于兜底解析时保留列边界。
    """
    if not line or "|" in line:
        return False
    if re.search(r"[，。；：、]", line):
        return False

    tokens = line.split()
    if len(tokens) < 4:
        return False

    numeric_tokens = re.findall(
        r"(?:\(?-?\d+(?:,\d{3})*(?:\.\d+)?\)?%?|[-+]?\d+(?:\.\d+)?x)",
        line,
    )
    return len(numeric_tokens) >= 3 and len(numeric_tokens) / len(tokens) >= 0.45


def _pipe_financial_table_line(line: str) -> str:
    """
    给 pypdf 兜底结果补列分隔符，同时尽量保留指标名整体。
    """
    tokens = line.split()
    first_numeric_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if re.search(r"\d", token) and not re.fullmatch(r"\d+[A-Za-z]+", token)
        ),
        None,
    )
    if first_numeric_index is None or first_numeric_index == 0:
        return " | ".join(tokens)

    label = " ".join(tokens[:first_numeric_index])
    values = tokens[first_numeric_index:]
    return " | ".join([label] + values)


def _normalize_report_text(text: str) -> str:
    """
    针对中文研报做轻量规范化：去掉免责声明页眉，修复表格行边界。
    """
    if not text:
        return ""

    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = _clean_text(raw_line)
        if not line:
            cleaned_lines.append("")
            continue
        if line == "请务必阅读正文之后的免责声明及其项下所有内容":
            continue
        if _looks_like_financial_table_line(line):
            line = _pipe_financial_table_line(line)
        cleaned_lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def _content_quality_score(content: str) -> float:
    """
    粗略评估解析结果质量，用于避免第一个引擎返回低质量文本时过早退出。
    """
    if not content:
        return 0.0

    length_score = min(len(content) / 800.0, 6.0)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
    chinese_score = min(chinese_chars / 300.0, 4.0)
    table_score = min(content.count("|") / 20.0, 3.0)
    page_score = min(len(re.findall(r"【第 \d+ 页】", content)), 2.0)
    replacement_penalty = content.count("\ufffd") * 0.2
    return length_score + chinese_score + table_score + page_score - replacement_penalty


def _deduplicate_lines(text: str, max_repeats: int = 3, edge_lines: int = 1) -> str:
    """
    删除疑似 PDF 页眉/页脚的重复行

    - 只统计每页正文开头和结尾 edge_lines 行，避免误删正文中的重复标题或表格字段
    - 仅对长度 >= 4 的行进行计数
    - 在页边缘出现超过 max_repeats 次的重复行会被剔除
    """
    if not text:
        return ""

    page_blocks = re.split(r"(?=【第 \d+ 页】)", text)
    page_blocks = [block for block in page_blocks if block.strip()]
    if not page_blocks:
        return text

    line_counts = {}
    pages = []

    for block in page_blocks:
        lines = block.splitlines()
        pages.append(lines)

        content_lines = [
            line for line in lines
            if line.strip() and not re.fullmatch(r"【第 \d+ 页】", line.strip())
        ]
        edge_candidates = content_lines[:edge_lines] + content_lines[-edge_lines:]
        seen_on_page = set()
        for line in edge_candidates:
            key = line.strip()
            if len(key) < 4:
                continue
            seen_on_page.add(key)

        for key in seen_on_page:
            line_counts[key] = line_counts.get(key, 0) + 1

    repeated_edges = {
        line for line, count in line_counts.items()
        if count > max_repeats
    }

    cleaned_pages = []
    for lines in pages:
        cleaned = []
        content_indexes = [
            index for index, line in enumerate(lines)
            if line.strip() and not re.fullmatch(r"【第 \d+ 页】", line.strip())
        ]
        edge_indexes = set(content_indexes[:edge_lines] + content_indexes[-edge_lines:])

        for index, line in enumerate(lines):
            key = line.strip()
            if index in edge_indexes and key in repeated_edges:
                continue
            cleaned.append(line)

        page_text = "\n".join(cleaned).strip()
        if page_text:
            cleaned_pages.append(page_text)

    return "\n\n".join(cleaned_pages)


# ============================================================
# 解析引擎：pdfplumber
# ============================================================

def _parse_with_pdfplumber(file_path: str, ocr_if_sparse: bool) -> Optional[PDFParseResult]:
    """
    使用 pdfplumber 解析 PDF
    - 优点：保留版面、表格、分栏信息
    """
    try:
        import pdfplumber
    except ImportError:
        return None

    warnings = []
    page_blocks = []

    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            parts = []

            # 提取普通文本（保留布局）
            text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3) or ""
            text = _normalize_report_text(text)
            if text:
                parts.append(text)

            # 提取表格
            try:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "intersection_tolerance": 5,
                        "min_words_vertical": 2,
                        "min_words_horizontal": 1,
                    }
                ) or []
            except Exception as exc:
                tables = []
                warnings.append(f"第 {page_index} 页表格提取失败: {exc}")

            for table_index, table in enumerate(tables, start=1):
                table_text = _format_table(table)
                if table_text:
                    parts.append(f"[表格 {table_index}]\n{table_text}")

            page_text = "\n\n".join(parts).strip()

            # 如果文本过少，尝试 OCR
            if ocr_if_sparse and len(page_text) < 40:
                ocr_text = _ocr_page_with_pdfplumber(page, page_index, warnings)
                if ocr_text:
                    page_text = "\n\n".join([page_text, "[OCR]\n" + ocr_text]).strip()

            if page_text:
                page_blocks.append(f"【第 {page_index} 页】\n{page_text}")

        content = _normalize_report_text(_deduplicate_lines("\n\n".join(page_blocks)))
        return PDFParseResult(
            source=file_path,
            content=content,
            parser="pdfplumber",
            pages=len(pdf.pages),
            warnings=warnings,
        )


def _ocr_page_with_pdfplumber(page, page_index: int, warnings: List[str]) -> str:
    """
    使用 pytesseract 对 pdfplumber 页面进行 OCR
    """
    try:
        import pytesseract
    except ImportError:
        warnings.append("检测到疑似扫描页，但未安装 pytesseract，已跳过 OCR。")
        return ""

    try:
        image = page.to_image(resolution=220).original
        return _clean_text(pytesseract.image_to_string(image, lang="chi_sim+eng"))
    except Exception as exc:
        warnings.append(f"第 {page_index} 页 OCR 失败: {exc}")
        return ""


# ============================================================
# 解析引擎：PyMuPDF（fitz）
# ============================================================

def _parse_with_pymupdf(file_path: str, ocr_if_sparse: bool) -> Optional[PDFParseResult]:
    """
    使用 PyMuPDF 解析 PDF
    - 优点：速度快、兼容性强
    """
    try:
        import fitz
    except ImportError:
        return None

    warnings = []
    page_blocks = []
    doc = fitz.open(file_path)

    try:
        for page_index, page in enumerate(doc, start=1):
            text = _normalize_report_text(page.get_text("text", sort=True))

            if ocr_if_sparse and len(text) < 40:
                ocr_text = _ocr_page_with_pymupdf(page, page_index, warnings)
                if ocr_text:
                    text = "\n\n".join([text, "[OCR]\n" + ocr_text]).strip()

            if text:
                page_blocks.append(f"【第 {page_index} 页】\n{text}")

        content = _normalize_report_text(_deduplicate_lines("\n\n".join(page_blocks)))
        return PDFParseResult(
            source=file_path,
            content=content,
            parser="pymupdf",
            pages=doc.page_count,
            warnings=warnings,
        )
    finally:
        doc.close()


def _ocr_page_with_pymupdf(page, page_index: int, warnings: List[str]) -> str:
    """
    使用 PyMuPDF + pytesseract 进行 OCR
    """
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        warnings.append("检测到疑似扫描页，但未安装 pytesseract，已跳过 OCR。")
        return ""

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return _clean_text(pytesseract.image_to_string(image, lang="chi_sim+eng"))
    except Exception as exc:
        warnings.append(f"第 {page_index} 页 OCR 失败: {exc}")
        return ""


# ============================================================
# 解析引擎：pypdf（兜底）
# ============================================================

def _parse_with_pypdf(file_path: str) -> Optional[PDFParseResult]:
    """
    使用 pypdf 解析 PDF
    - 优点：依赖最少，作为兜底方案
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    warnings = []
    reader = PdfReader(file_path)
    page_blocks = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize_report_text(page.extract_text() or "")
        except Exception as exc:
            text = ""
            warnings.append(f"第 {page_index} 页文本提取失败: {exc}")
        if text:
            page_blocks.append(f"【第 {page_index} 页】\n{text}")

    content = _normalize_report_text(_deduplicate_lines("\n\n".join(page_blocks)))
    return PDFParseResult(
        source=file_path,
        content=content,
        parser="pypdf",
        pages=len(reader.pages),
        warnings=warnings,
    )


# ============================================================
# 解析引擎：MinerU 在线 API（最高优先级）
# ============================================================

def _parse_with_mineru_api(file_path: str) -> Optional[PDFParseResult]:
    """
    使用 MinerU 官方在线 API 解析 PDF。
    流程：申请上传链接 → PUT 上传 → 轮询任务 → 下载 zip 取 full.md。
    未配置 token 或调用失败时返回 None，走兜底引擎。
    """
    try:
        import os
        import io
        import time
        import zipfile
        import requests
        from config import settings
    except ImportError:
        return None

    token = (settings.MINERU_API_TOKEN or "").strip()
    if not token:
        return None

    api_base = settings.MINERU_API_BASE
    headers = {"Authorization": f"Bearer {token}"}
    filename = os.path.basename(file_path)
    warnings: List[str] = []

    try:
        apply_resp = requests.post(
            f"{api_base}/file-urls/batch",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "enable_formula": True,
                "enable_table": True,
                "language": "ch",
                "files": [{"name": filename, "is_ocr": True}],
            },
            timeout=30,
        )
        apply_resp.raise_for_status()
        apply_data = apply_resp.json().get("data") or {}
        batch_id = apply_data.get("batch_id")
        upload_urls = apply_data.get("file_urls") or []
        if not batch_id or not upload_urls:
            warnings.append(f"MinerU 申请上传链接失败: {apply_resp.text[:200]}")
            return None

        with open(file_path, "rb") as f:
            put_resp = requests.put(upload_urls[0], data=f, timeout=120)
        put_resp.raise_for_status()

        deadline = time.time() + settings.MINERU_TIMEOUT
        zip_url = None
        while time.time() < deadline:
            poll_resp = requests.get(
                f"{api_base}/extract-results/batch/{batch_id}",
                headers=headers,
                timeout=30,
            )
            poll_resp.raise_for_status()
            rows = (poll_resp.json().get("data") or {}).get("extract_result") or []
            if rows:
                row = rows[0]
                state = row.get("state")
                if state == "done":
                    zip_url = row.get("full_zip_url")
                    break
                if state == "failed":
                    warnings.append(f"MinerU 解析失败: {row.get('err_msg')}")
                    return None
            time.sleep(3)

        if not zip_url:
            warnings.append("MinerU 解析超时")
            return None

        zip_bytes = requests.get(zip_url, timeout=120).content
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            md_names = [n for n in zf.namelist() if n.endswith(".md")]
            if not md_names:
                warnings.append("MinerU 结果中未找到 Markdown 文件")
                return None
            content = zf.read(md_names[0]).decode("utf-8", errors="ignore")

        return PDFParseResult(
            source=file_path,
            content=content.strip(),
            parser="mineru-api",
            pages=0,
            warnings=warnings,
        )
    except Exception as exc:
        warnings.append(f"MinerU API 异常: {exc}")
        return PDFParseResult(
            source=file_path,
            content="",
            parser="mineru-api",
            pages=0,
            warnings=warnings,
        )


# ============================================================
# 对外统一接口
# ============================================================

def parse_pdf(file_path: str, ocr_if_sparse: bool = True) -> PDFParseResult:
    """
    PDF 解析的统一入口函数

    解析策略（按顺序）：
    1. MinerU 在线 API（最高优先级，配置 token 后启用）
    2. pdfplumber / PyMuPDF / pypdf 兜底，按质量分挑最优

    扫描版 PDF 会自动触发 OCR（依赖 pytesseract）
    """
    missing = []
    candidates: List[PDFParseResult] = []

    mineru_result = _parse_with_mineru_api(file_path)
    if mineru_result is None:
        missing.append("mineru-api")
    elif mineru_result.content:
        return mineru_result

    for parser_name, parser in (
        ("pdfplumber", lambda: _parse_with_pdfplumber(file_path, ocr_if_sparse)),
        ("pymupdf", lambda: _parse_with_pymupdf(file_path, ocr_if_sparse)),
        ("pypdf", lambda: _parse_with_pypdf(file_path)),
    ):
        result = parser()
        if result is None:
            missing.append(parser_name)
            continue
        if result.content:
            candidates.append(result)

    if candidates:
        return max(candidates, key=lambda item: _content_quality_score(item.content))

    raise RuntimeError(
        "PDF 解析失败或内容为空。请至少安装一个解析库："
        "pip install pdfplumber pymupdf pypdf；扫描版 PDF 还需要 pytesseract 和系统 tesseract。"
        f" 当前缺失或不可用的解析器: {', '.join(missing)}"
    )
