"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
PDF解析模块 - 负责PDF文本提取与预处理
"""

import os
import re


class PDFParser:
    """PDF文档解析器，支持多种解析引擎"""

    def __init__(self, use_pymupdf: bool = True):
        """
        Args:
            use_pymupdf: 是否使用pymupdf（更精准）。False则回退到pypdf。
        """
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

    def extract_text(self, pdf_path: str) -> list[dict]:
        """
        从PDF中提取文本，按页返回。

        Returns:
            list[dict]: [{"page": 1, "text": "..."}, ...]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        if self.engine == "pymupdf":
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: str) -> list[dict]:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = self._clean_text(text)
            pages.append({"page": i + 1, "text": text})
        doc.close()
        return pages

    def _extract_with_pypdf(self, pdf_path: str) -> list[dict]:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text = self._clean_text(text)
            pages.append({"page": i + 1, "text": text or ""})
        return pages

    def _clean_text(self, text: str) -> str:
        """清洗文本：去除多余的空白、页眉页脚标记等"""
        text = re.sub(r'\u3000', ' ', text)          # 全角空格 -> 半角
        text = re.sub(r'[ \t]+', ' ', text)          # 多个空格 -> 一个
        text = re.sub(r'\n{3,}', '\n\n', text)       # 过多空行 -> 两个
        text = text.strip()
        return text

    def get_metadata(self, pdf_path: str) -> dict:
        """获取PDF元数据（页数、标题等）"""
        if self.engine == "pymupdf":
            import pymupdf
            doc = pymupdf.open(pdf_path)
            meta = {
                "页数": len(doc),
                "标题": doc.metadata.get("title", ""),
                "作者": doc.metadata.get("author", ""),
                "主题": doc.metadata.get("subject", ""),
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
                "作者": getattr(info, "author", ""),
            }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        pages = parser.extract_text(sys.argv[1])
        print(f"共 {len(pages)} 页")
        print(f"第一页前500字:\n{pages[0]['text'][:500]}")
