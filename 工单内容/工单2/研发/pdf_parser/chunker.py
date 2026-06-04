"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
分块模块 - 语义分块（基于段落边界+滑动窗口）
"""

import re
from typing import List, Dict


class TextChunker:
    """
    语义分块器 - 基于段落边界+滑动窗口

    优化策略:
    1. 优先按段落边界分块，保持语义完整性
    2. 大段落使用滑动窗口切分
    3. 保留标题上下文
    4. 表格单独分块
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ============================================================
    # 语义分块（结构化输入）
    # ============================================================
    def chunk_structured(self, structured_pages: List) -> List[Dict]:
        """
        基于结构化页面进行语义分块

        Args:
            structured_pages: PDFParser.extract_structured() 返回的页面列表

        Returns:
            List[Dict]: [{"text": str, "page": int, "type": str, "heading": str}, ...]
        """
        chunks = []

        for page in structured_pages:
            page_num = page.page_num
            headings = page.headings
            paragraphs = page.paragraphs
            tables = page.tables

            # 提取标题上下文
            heading_context = ""
            if headings:
                heading_context = " | ".join(h["text"] for h in headings[:3])

            # 处理表格（单独成块）
            for tab in tables:
                tab_text = self._table_to_text(tab)
                if tab_text:
                    chunks.append({
                        "text": tab_text,
                        "page": page_num,
                        "type": "table",
                        "heading": heading_context,
                    })

            # 处理段落
            for para in paragraphs:
                para_text = para.get("text", "").strip()
                if not para_text:
                    continue

                # 短段落直接作为一个chunk
                if len(para_text) <= self.chunk_size:
                    chunks.append({
                        "text": para_text,
                        "page": page_num,
                        "type": "paragraph",
                        "heading": heading_context,
                    })
                else:
                    # 长段落用滑动窗口切分，保持句子边界
                    sub_chunks = self._split_long_text(para_text)
                    for sc in sub_chunks:
                        chunks.append({
                            "text": sc,
                            "page": page_num,
                            "type": "paragraph_chunk",
                            "heading": heading_context,
                        })

        # 合并过短的相邻chunk
        chunks = self._merge_short_chunks(chunks)

        return chunks

    def _table_to_text(self, table: Dict) -> str:
        """将表格转换为文本描述"""
        rows = table.get("rows", [])
        if not rows:
            return ""

        lines = ["【表格数据】"]
        for row in rows:
            if row:
                lines.append(" | ".join(str(cell or "") for cell in row))

        return "\n".join(lines)

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
                # 重叠：从上一个chunk的后半部分开始
                if len(current) > self.chunk_overlap:
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + sent
                else:
                    current = sent

        if current:
            chunks.append(current.strip())

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """按句子切分"""
        # 中文+英文句子结束符
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

            # 尝试与下一个合并，如果总长度不超过限制
            while (i + 1 < len(chunks)
                   and len(current_text) < self.chunk_size // 2
                   and len(current_text) + len(chunks[i + 1]["text"]) <= self.chunk_size
                   and chunks[i + 1]["page"] == current["page"]):
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
                })
            else:
                sub_chunks = self._split_long_text(text)
                for sc in sub_chunks:
                    chunks.append({
                        "text": sc,
                        "page": page_num,
                        "type": "page_chunk",
                        "heading": "",
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
