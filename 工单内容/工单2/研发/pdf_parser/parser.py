"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
PDF解析模块 - 结构化文本提取（标题、段落、表格）
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple


class StructuredPage:
    """结构化页面，包含段落和表格"""
    def __init__(self, page_num: int, text: str = ""):
        self.page_num = page_num
        self.text = text
        self.paragraphs: List[Dict] = []
        self.tables: List[Dict] = []
        self.headings: List[Dict] = []

    def to_dict(self) -> Dict:
        return {
            "page": self.page_num,
            "text": self.text,
            "paragraphs": self.paragraphs,
            "tables": self.tables,
            "headings": self.headings,
        }


class PDFParser:
    """PDF文档解析器 - 结构化解析，提取标题层级、段落、表格"""

    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf
        self._engine = None

    @property
    def engine(self):
        if self._engine is not None:
            return self._engine
        if self.use_pymupdf:
            try:
                import pymupdf
                self._engine = "pymupdf"
                return "pymupdf"
            except ImportError:
                pass
        try:
            from pypdf import PdfReader
            self._engine = "pypdf"
            return "pypdf"
        except ImportError:
            raise ImportError("请安装 pymupdf 或 pypdf: pip install pymupdf")

    # ============================================================
    # 结构化提取（主入口）
    # ============================================================
    def extract_structured(self, pdf_path: str) -> List[StructuredPage]:
        """结构化提取：返回带段落、表格、标题信息的页面列表"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        if self.engine == "pymupdf":
            return self._extract_structured_pymupdf(pdf_path)
        else:
            return self._extract_structured_pypdf(pdf_path)

    def _extract_structured_pymupdf(self, pdf_path: str) -> List[StructuredPage]:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        pages: List[StructuredPage] = []

        for i, page in enumerate(doc):
            sp = StructuredPage(page_num=i + 1)

            # 提取原始文本块（保留位置信息）
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))  # 按y, x排序

            raw_text = page.get_text("text")
            sp.text = self._clean_text(raw_text)

            # 解析标题（基于字体大小判断）
            sp.headings = self._detect_headings(blocks)

            # 提取段落
            sp.paragraphs = self._extract_paragraphs(blocks, sp.headings)

            # 提取表格
            sp.tables = self._extract_tables(page)

            pages.append(sp)

        doc.close()
        return pages

    def _extract_structured_pypdf(self, pdf_path: str) -> List[StructuredPage]:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages: List[StructuredPage] = []

        for i, page in enumerate(reader.pages):
            sp = StructuredPage(page_num=i + 1)
            text = page.extract_text() or ""
            sp.text = self._clean_text(text)

            # pypdf无法获取字体信息，用启发式方法检测标题和段落
            sp.headings = self._detect_headings_heuristic(text)
            sp.paragraphs = self._extract_paragraphs_heuristic(text)
            sp.tables = []  # pypdf不直接支持表格检测

            pages.append(sp)

        return pages

    # ============================================================
    # 标题检测
    # ============================================================
    def _detect_headings(self, blocks: List[Tuple]) -> List[Dict]:
        """基于字体大小检测标题"""
        headings = []
        if not blocks:
            return headings

        # 收集所有字体大小
        font_sizes = []
        for b in blocks:
            if len(b) >= 7 and b[6] > 0:  # font size
                font_sizes.append(b[6])

        if not font_sizes:
            return headings

        avg_size = sum(font_sizes) / len(font_sizes)
        threshold = avg_size * 1.15  # 标题字体一般比普通文本大15%以上

        for b in blocks:
            if len(b) < 7:
                continue
            text = b[4].strip()
            font_size = b[6]
            is_bold = b[5] if len(b) > 5 else 0

            if font_size >= threshold and len(text) < 80 and text:
                level = 1 if font_size > avg_size * 1.4 else 2
                headings.append({
                    "text": text,
                    "level": level,
                    "bbox": b[:4],
                    "font_size": font_size,
                })

        return headings

    def _detect_headings_heuristic(self, text: str) -> List[Dict]:
        """pypdf回退：基于文本特征检测标题"""
        headings = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 标题特征：短行、无标点结尾、全大写或数字开头
            if len(line) < 60 and not line[-1] in "，。、；：！？,.;:!?":
                if line.isupper() or re.match(r"^[\d一二三四五六七八九十]+", line):
                    headings.append({"text": line, "level": 1})

        return headings

    # ============================================================
    # 段落提取
    # ============================================================
    def _extract_paragraphs(self, blocks, headings: List[Dict]) -> List[Dict]:
        """从文本块中提取段落"""
        paragraphs = []
        heading_texts = {h["text"] for h in headings}

        current_para = {"text": "", "type": "paragraph"}
        for b in blocks:
            text = b[4].strip()
            if not text or text in heading_texts:
                if current_para["text"]:
                    paragraphs.append(current_para)
                    current_para = {"text": "", "type": "paragraph"}
                continue
            current_para["text"] += " " + text if current_para["text"] else text

        if current_para["text"]:
            paragraphs.append(current_para)

        # 合并过短的段落
        merged = []
        for p in paragraphs:
            if len(p["text"]) < 30 and merged:
                merged[-1]["text"] += " " + p["text"]
            else:
                merged.append(p)

        return merged

    def _extract_paragraphs_heuristic(self, text: str) -> List[Dict]:
        """pypdf回退：基于空行分割段落"""
        paragraphs = []
        raw_paras = text.split("\n\n")
        for p in raw_paras:
            p = p.strip()
            if len(p) > 10:
                paragraphs.append({"text": p, "type": "paragraph"})
        return paragraphs

    # ============================================================
    # 表格提取
    # ============================================================
    def _extract_tables(self, page) -> List[Dict]:
        """提取页面中的表格"""
        tables = []
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                headers = tab.header.names if tab.header else []
                data = tab.extract()
                tables.append({
                    "headers": headers,
                    "rows": data,
                    "bbox": tab.bbox,
                })
        except Exception:
            pass
        return tables

    # ============================================================
    # 兼容旧接口：简单按页提取
    # ============================================================
    def extract_text(self, pdf_path: str) -> List[Dict]:
        """兼容旧接口：返回[{page, text}]格式"""
        pages = self.extract_structured(pdf_path)
        return [{"page": p.page_num, "text": p.text} for p in pages]

    # ============================================================
    # 文本清洗
    # ============================================================
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\u3000', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def get_metadata(self, pdf_path: str) -> Dict:
        if self.engine == "pymupdf":
            import pymupdf
            doc = pymupdf.open(pdf_path)
            meta = {
                "页数": len(doc),
                "标题": doc.metadata.get("title", ""),
                "作者": doc.metadata.get("author", ""),
            }
            doc.close()
            return meta
        else:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            info = reader.metadata
            return {
                "页数": len(reader.pages),
                "标题": getattr(info, "title", ""),
            }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        pages = parser.extract_structured(sys.argv[1])
        print(f"共 {len(pages)} 页")
        print(f"第1页标题: {[h['text'] for h in pages[0].headings][:5]}")
        print(f"第1页段落数: {len(pages[0].paragraphs)}")
