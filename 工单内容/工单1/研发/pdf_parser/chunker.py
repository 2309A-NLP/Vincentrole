"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
文本分块模块 - 将长文本切分为语义完整的chunks
"""

import re
from typing import List


class TextChunker:
    """智能文本分块器，保持段落和句子完整性"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100,
                 separators: list = None):
        """
        Args:
            chunk_size: 每个chunk的目标字符数
            chunk_overlap: chunk之间的重叠字符数
            separators: 优先使用的分隔符（按优先级排序）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "；", "，", " ", ""]

    def chunk_text(self, text: str, source_page: int = None) -> List[dict]:
        """
        将一段文本切分为多个chunk。
        
        Args:
            text: 输入文本
            source_page: 来源页码（可选）
            
        Returns:
            list[dict]: [{"text": "...", "page": N, "chunk_id": M}, ...]
        """
        chunks = []
        # 先按段落切分作为基本单元
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        for para in paragraphs:
            # 如果当前chunk + 新段落 不超过上限，直接追加
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk = self._join_chunk(current_chunk, para)
            else:
                # 当前chunk已满，保存
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 新段落如果超过chunk_size，需要进一步拆分
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_paragraph(para)
                    for sc in sub_chunks:
                        if sc.strip():
                            chunks.append(sc.strip())
                    current_chunk = ""
                else:
                    current_chunk = para

        # 最后一段
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 处理重叠：在相邻chunk之间加入重叠内容
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks_with_overlap = []
            for i, c in enumerate(chunks):
                if i > 0:
                    # 从前一个chunk尾部取重叠内容
                    prev = chunks[i - 1]
                    overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
                    c = overlap_text + "\n" + c
                chunks_with_overlap.append(c)
            chunks = chunks_with_overlap

        # 组装返回格式
        result = []
        for i, c in enumerate(chunks):
            result.append({
                "text": c,
                "page": source_page,
                "chunk_id": i,
                "char_count": len(c),
            })
        return result

    def chunk_pages(self, pages: List[dict]) -> List[dict]:
        """
        将多页文本批量分块。
        
        Args:
            pages: PDFParser返回的pages列表
            
        Returns:
            所有chunk的列表
        """
        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_text(page["text"], source_page=page.get("page"))
            all_chunks.extend(page_chunks)
        return all_chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """按双换行分割段落"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_paragraph(self, para: str) -> List[str]:
        """将一个超长段落拆分为多个chunk"""
        # 尝试按句号拆分
        sentences = re.split(r'(?<=[。！？])', para)
        chunks = []
        current = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current.strip():
                    chunks.append(current.strip())
                # 如果句子本身超长，按字符硬切
                while len(sent) > self.chunk_size:
                    chunks.append(sent[:self.chunk_size].strip())
                    sent = sent[self.chunk_size:]
                current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _join_chunk(self, current: str, new_part: str) -> str:
        """连接两个文本片段"""
        if not current:
            return new_part
        if current.endswith('\n') or new_part.startswith('\n'):
            return current + new_part
        return current + "\n" + new_part


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    test_text = ("这是第一段。这是第一段的内容，用于测试分块功能。"
                 "这是第二段。第二段包含更多信息。"
                 "这是第三段。第三段测试重叠机制。")
    chunks = chunker.chunk_text(test_text)
    for c in chunks:
        print(f"[{c['chunk_id']}] ({c['char_count']}字): {c['text'][:80]}...")
