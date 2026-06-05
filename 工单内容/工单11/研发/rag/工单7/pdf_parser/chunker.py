"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
分块模块 - 增强版（表格感知分块 + 多文档支持）
"""

import re
from typing import List, Dict


class TextChunker:
    """
    语义分块器 - 表格感知分块 + 多文档支持。

    优化策略（工单3）:
    1. 纯文本表格 → Markdown格式表格 + 结构化JSON
    2. 表格元数据：标题、列名、列类型、行数
    3. 表格内容双重嵌入：全文描述 + 关键数值摘要
    4. 多文档标记：每个chunk带 source_file
    5. 数值表格单独摘要chunk（便于数值查询匹配）
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ============================================================
    # 语义分块（结构化输入）
    # ============================================================
    def chunk_structured(self, structured_pages: List) -> List[Dict]:
        """
        基于结构化页面进行语义分块（增强表格处理）。

        Args:
            structured_pages: PDFParser.extract_structured() 返回的页面列表

        Returns:
            List[Dict]: [{
                "text": str,
                "page": int,
                "type": str,
                "heading": str,
                "source_file": str,      # 来源PDF文件名
                "table_info": {          # 仅表格chunk有
                    "headers": [...],
                    "title": "...",
                    "has_numbers": bool,
                    "column_types": [...],
                }
            }, ...]
        """
        chunks = []

        for page in structured_pages:
            page_num = page.page_num
            headings = page.headings
            paragraphs = page.paragraphs
            tables = page.tables
            source_file = getattr(page, "source_file", "")

            # 提取标题上下文
            heading_context = ""
            if headings:
                heading_context = " | ".join(h["text"] for h in headings[:3])

            # ---- 处理表格（增强版）----
            for tab in tables:
                # 生成丰富表格文本（Markdown格式 + 标题 + 摘要）
                table_chunks = self._chunk_table_enhanced(tab, page_num, heading_context, source_file)
                chunks.extend(table_chunks)

            # ---- 处理段落 ----
            for para in paragraphs:
                para_text = para.get("text", "").strip()
                if not para_text:
                    continue

                if len(para_text) <= self.chunk_size:
                    chunks.append({
                        "text": para_text,
                        "page": page_num,
                        "type": "paragraph",
                        "heading": heading_context,
                        "source_file": source_file,
                    })
                else:
                    sub_chunks = self._split_long_text(para_text)
                    for sc in sub_chunks:
                        chunks.append({
                            "text": sc,
                            "page": page_num,
                            "type": "paragraph_chunk",
                            "heading": heading_context,
                            "source_file": source_file,
                        })

            # ---- 处理图像----
            if hasattr(page, 'images') and page.images:
                for img in page.images:
                    img_text_parts = []
                    if img.get("image_type"):
                        img_text_parts.append(f"【图像类型】{img['image_type']}")
                    if img.get("caption_text"):
                        img_text_parts.append(f"【图标题】{img['caption_text']}")
                    if img.get("description"):
                        img_text_parts.append(f"【描述】{img['description']}")
                    if img.get("surrounding_text"):
                        img_text_parts.append(f"【上下文】{img['surrounding_text']}")
                    img_text = "\n".join(img_text_parts)

                    image_info = {
                        "image_path": img.get("image_path", ""),
                        "image_type": img.get("image_type", ""),
                        "caption_text": img.get("caption_text", ""),
                        "has_caption": bool(img.get("caption_text")),
                    }

                    # 主图像chunk
                    chunks.append({
                        "text": img_text,
                        "page": page_num,
                        "type": "image",
                        "heading": heading_context,
                        "source_file": source_file,
                        "image_info": image_info,
                    })

                    # 额外描述chunk（纯文本，便于检索匹配）
                    if img.get("description"):
                        chunks.append({
                            "text": img["description"],
                            "page": page_num,
                            "type": "image_description",
                            "heading": heading_context,
                            "source_file": source_file,
                            "image_info": image_info,
                        })

        # 合并过短的相邻chunk
        chunks = self._merge_short_chunks(chunks)

        return chunks

    def _chunk_table_enhanced(self, tab: Dict, page_num: int,
                               heading_context: str, source_file: str) -> List[Dict]:
        """
        增强表格分块：
        1. Markdown格式表格（便于LLM理解）
        2. 结构化JSON描述
        3. 数值关键数据摘要chunk
        """
        result = []

        headers = tab.get("headers", [])
        rows = tab.get("rows", [])
        title = tab.get("title", "")
        has_numbers = tab.get("has_numbers", False)
        col_types = tab.get("column_types", [])
        markdown = tab.get("markdown", "")
        summary = tab.get("summary", "")
        structured = tab.get("structured", [])
        cross_page = tab.get("cross_page", False)
        page_range = tab.get("page_range", None)

        # 构建表格文本
        table_text_parts = []

        # 标题行
        if title:
            table_text_parts.append(f"【表格标题】{title}")

        # Markdown表格（主要表示）
        if markdown:
            table_text_parts.append(f"【表格数据】\n{markdown}")

        # 摘要
        if summary:
            table_text_parts.append(f"【表格信息】{summary}")

        # 结构化数据（数值表格转换为文本描述）
        if has_numbers and structured:
            desc_lines = ["【关键数值】"]
            for item in structured[:30]:  # 最多30行
                parts = []
                for key, val in item.items():
                    if isinstance(val, (int, float)):
                        parts.append(f"{key}={val}")
                    else:
                        parts.append(f"{key}: {val}")
                if parts:
                    desc_lines.append(" | ".join(parts))
            if desc_lines:
                table_text_parts.append("\n".join(desc_lines))

        table_text = "\n\n".join(table_text_parts)

        # 建立 table_info 元数据
        table_info = {
            "headers": headers,
            "title": title or "",
            "has_numbers": has_numbers,
            "column_types": col_types,
            "row_count": tab.get("row_count", 0),
        }

        # 显示页码（跨页表格显示页码范围，但page字段始终为整数）
        display_page = page_num
        page_value = page_num
        if cross_page and page_range:
            display_page = f"{page_range[0]}-{page_range[1]}"
            page_value = page_range[0]  # 排序用第一页
        else:
            display_page = page_num
            page_value = page_num

        # 主chunk（完整表格内容）
        result.append({
            "text": table_text,
            "page": page_value,
            "page_display": display_page,
            "type": "table",
            "heading": heading_context,
            "source_file": source_file,
            "table_info": table_info,
        })

        # 如果表格很大且包含数值数据，额外添加一个数值摘要chunk
        if has_numbers and len(rows) > 5:
            # 提取数值摘要用于快速匹配
            numeric_summary = self._extract_numeric_summary(
                title, headers, rows, col_types
            )
            if numeric_summary:
                result.append({
                    "text": numeric_summary,
                    "page": page_value,
                    "type": "table_numeric",
                    "heading": heading_context,
                    "source_file": source_file,
                    "table_info": table_info,
                })

        return result

    def _extract_numeric_summary(self, title: str, headers: list,
                                  rows: list, col_types: list) -> str:
        """提取表格中的数值数据摘要（便于数值查询匹配）"""
        if not rows or not headers:
            return ""

        # 找到数值列
        numeric_cols = []
        text_cols = []
        for i, ct in enumerate(col_types):
            if i < len(headers):
                if ct in ["numeric", "percentage"]:
                    numeric_cols.append((i, str(headers[i])))
                else:
                    text_cols.append((i, str(headers[i])))

        if not numeric_cols:
            return ""

        lines = []
        if title:
            lines.append(f"【{title}数值摘要】")

        # 对每一行提取 (文本标签, 数值) 对
        for row in rows:
            row_parts = []
            for col_i, col_name in text_cols:
                if col_i < len(row) and row[col_i]:
                    row_parts.append(str(row[col_i]).strip())
            for col_i, col_name in numeric_cols:
                if col_i < len(row) and row[col_i]:
                    val = str(row[col_i]).strip()
                    if row_parts:
                        lines.append(f"{' '.join(row_parts)} → {col_name}: {val}")
                    else:
                        lines.append(f"{col_name}: {val}")

        return "\n".join(lines[:50])  # 最多50行

    # ============================================================
    # 长文本切分
    # ============================================================
    def _split_long_text(self, text: str) -> List[str]:
        """长文本切分，优先按句子边界"""
        chunks = []
        sentences = self._split_sentences(text)
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current.strip())
                if len(current) > self.chunk_overlap:
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + sent
                else:
                    current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        pattern = r'([。！？.!?;；]\s*)'
        parts = re.split(pattern, text)
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                sentences.append(parts[i])
                i += 1
        return sentences if sentences else [text]

    def _merge_short_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """合并过短的相邻chunk（保留语义连续性）"""
        if not chunks:
            return chunks
        merged = []
        i = 0
        while i < len(chunks):
            current = chunks[i].copy()
            current_text = current["text"]
            while (i + 1 < len(chunks)
                   and len(current_text) < self.chunk_size // 2
                   and len(current_text) + len(chunks[i + 1]["text"]) <= self.chunk_size
                   and chunks[i + 1]["page"] == current["page"]
                   and chunks[i + 1].get("source_file") == current.get("source_file")):
                current_text += "\n" + chunks[i + 1]["text"]
                i += 1
            current["text"] = current_text
            merged.append(current)
            i += 1
        return merged

    # ============================================================
    # 兼容旧接口
    # ============================================================
    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """兼容旧接口：从简单页面列表分块"""
        chunks = []
        for page in pages:
            page_num = page["page"]
            text = page["text"]
            if len(text) <= self.chunk_size:
                chunks.append({
                    "text": text,
                    "page": page_num,
                    "type": "page",
                    "heading": "",
                    "source_file": page.get("source_file", ""),
                })
            else:
                sub_chunks = self._split_long_text(text)
                for sc in sub_chunks:
                    chunks.append({
                        "text": sc,
                        "page": page_num,
                        "type": "page_chunk",
                        "heading": "",
                        "source_file": page.get("source_file", ""),
                    })
        return chunks


if __name__ == "__main__":
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    test_text = (
        "武汉兴图新科电子股份有限公司是一家专业从事视频传输、处理与编码技术研发的企业。"
        "公司成立于2004年，注册资本为6000万元人民币。"
        "报告期内，公司军用领域主营业务收入占比达到85%以上。"
        "公司参与制定了多项国家军用视频标准。"
    )
    result = chunker._split_long_text(test_text)
    for i, c in enumerate(result):
        print(f"Chunk {i}: {len(c)}字 -> {c[:60]}...")
