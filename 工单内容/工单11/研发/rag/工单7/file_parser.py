"""
工单编号: 人工智能NLP-RAG-功能测试及评估
多格式文件解析器 - 支持PDF和TXT文件解析
"""

import os
from typing import List, Dict, Optional


class TXTFileParser:
    """TXT文件解析器 - 支持多种编码"""

    def __init__(self, encodings: List[str] = None):
        self.encodings = encodings or ['utf-8', 'gbk', 'gb2312', 'latin-1']

    def parse(self, txt_path: str) -> Dict:
        """
        解析TXT文件

        Args:
            txt_path: TXT文件路径

        Returns:
            包含文本内容和元数据的字典
        """
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT文件不存在: {txt_path}")

        content = self._read_with_encoding(txt_path)
        filename = os.path.basename(txt_path)

        return {
            "filename": filename,
            "content": content,
            "page_count": 1,  # TXT文件视为单页
            "file_type": "txt",
        }

    def _read_with_encoding(self, txt_path: str) -> str:
        """尝试多种编码读取文件"""
        for encoding in self.encodings:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise UnicodeError(f"无法使用以下编码读取文件: {self.encodings}")


class MultiFormatFileParser:
    """
    多格式文件解析器
    支持PDF和TXT文件解析
    """

    def __init__(self, pdf_parser=None):
        """
        初始化解析器

        Args:
            pdf_parser: PDF解析器实例（可选）
        """
        self.txt_parser = TXTFileParser()
        self.pdf_parser = pdf_parser  # 可以注入现有的PDFParser

    def parse(self, file_path: str) -> Dict:
        """
        解析文件（自动识别格式）

        Args:
            file_path: 文件路径

        Returns:
            包含文本内容和元数据的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.txt':
            return self.txt_parser.parse(file_path)
        elif ext == '.pdf':
            return self._parse_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _parse_pdf(self, pdf_path: str) -> Dict:
        """解析PDF文件"""
        if self.pdf_parser is None:
            # 动态导入PDFParser
            try:
                from pdf_parser.parser import PDFParser
                self.pdf_parser = PDFParser()
            except ImportError:
                raise ImportError("无法导入PDFParser，请确保pdf_parser模块可用")

        # 使用现有的PDFParser
        pages = self.pdf_parser.extract_structured(pdf_path)
        filename = os.path.basename(pdf_path)

        # 合并所有页面文本
        full_text = []
        for page in pages:
            full_text.append(page.text)
            for table in page.tables:
                full_text.append(table.get("text", ""))

        return {
            "filename": filename,
            "content": "\n".join(full_text),
            "page_count": len(pages),
            "file_type": "pdf",
            "pages": [p.to_dict() for p in pages],
        }

    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        return ['.pdf', '.txt']


class DocumentChunker:
    """
    文档分块器 - 支持PDF和TXT
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Dict, source_file: str = None) -> List[Dict]:
        """
        将文档分块

        Args:
            doc: 文档解析结果
            source_file: 来源文件名

        Returns:
            分块列表
        """
        if source_file is None:
            source_file = doc.get("filename", "unknown")

        file_type = doc.get("file_type", "unknown")

        if file_type == "pdf":
            return self._chunk_pdf_document(doc, source_file)
        elif file_type == "txt":
            return self._chunk_txt_document(doc, source_file)
        else:
            return self._chunk_generic(doc, source_file)

    def _chunk_pdf_document(self, doc: Dict, source_file: str) -> List[Dict]:
        """分块PDF文档（保留表格结构）"""
        chunks = []
        chunk_id = 0

        # 如果有结构化页面信息
        if "pages" in doc:
            for page in doc["pages"]:
                page_num = page.get("page", 0)

                # 处理段落
                for para in page.get("paragraphs", []):
                    text = para.get("text", "").strip()
                    if len(text) < 20:
                        continue

                    # 按chunk_size分割长段落
                    sub_chunks = self._split_text(text)
                    for sub_text in sub_chunks:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": sub_text,
                            "page": page_num,
                            "source_file": source_file,
                            "type": "paragraph",
                        })
                        chunk_id += 1

                # 处理表格
                for table in page.get("tables", []):
                    table_text = table.get("text", "").strip()
                    if not table_text:
                        continue

                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": table_text,
                        "page": page_num,
                        "source_file": source_file,
                        "type": "table",
                        "table_title": table.get("title", ""),
                    })
                    chunk_id += 1

        # 如果没有结构化信息，按纯文本处理
        if not chunks and "content" in doc:
            return self._chunk_txt_document(doc, source_file)

        return chunks

    def _chunk_txt_document(self, doc: Dict, source_file: str) -> List[Dict]:
        """分块TXT文档"""
        content = doc.get("content", "")
        chunks = []
        chunk_id = 0

        # 按段落分割
        paragraphs = content.split('\n')
        current_chunk = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                # 空行，保存当前chunk
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= 20:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "page": 1,
                            "source_file": source_file,
                            "type": "text",
                        })
                        chunk_id += 1
                    current_chunk = []
                continue

            current_chunk.append(para)

            # 检查长度
            current_text = '\n'.join(current_chunk)
            if len(current_text) >= self.chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_text,
                    "page": 1,
                    "source_file": source_file,
                    "type": "text",
                })
                chunk_id += 1
                current_chunk = []

        # 处理剩余内容
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= 20:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": 1,
                    "source_file": source_file,
                    "type": "text",
                })

        return chunks

    def _chunk_generic(self, doc: Dict, source_file: str) -> List[Dict]:
        """通用分块"""
        content = doc.get("content", "")
        chunks = []
        chunk_id = 0

        sub_chunks = self._split_text(content)
        for sub_text in sub_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "text": sub_text,
                "page": 1,
                "source_file": source_file,
                "type": "text",
            })
            chunk_id += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """分割文本为固定大小的chunk"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            # 尝试在句子边界分割
            segment = text[start:end]
            last_period = max(segment.rfind('。'), segment.rfind('！'), segment.rfind('？'))
            if last_period > self.chunk_size * 0.5:
                end = start + last_period + 1

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return [c for c in chunks if len(c.strip()) >= 20]
