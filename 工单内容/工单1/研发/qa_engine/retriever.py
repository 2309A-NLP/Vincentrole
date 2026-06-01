"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
检索器模块 - 从向量库检索相关文档片段
"""

from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.vector_store import VectorStore
from qa_engine.query_understanding import QueryUnderstanding


class Retriever:
    """
    RAG检索器。

    流程:
    1. Query理解与分析
    2. 文本向量化
    3. FAISS语义检索
    4. 结果排序与过滤
    """

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.query_understanding = QueryUnderstanding()

    def retrieve(self, question: str, top_k: int = 5,
                 threshold: float = 0.3) -> dict:
        """
        检索流程入口。

        Args:
            question: 用户问题
            top_k: 最大返回chunk数
            threshold: 相关性阈值

        Returns:
            dict: {
                "query_analysis": {...},   # Query理解结果
                "results": [...],           # 检索结果chunk列表
                "total_results": N,         # 有效结果数
                "context_text": "...",      # 拼接后的上下文文本
            }
        """
        # Step 1: Query理解
        analysis = self.query_understanding.analyze(question)

        # Step 2: 使用重写后的查询或原始查询
        query_text = analysis["rewritten_query"]
        if len(query_text) < 5:  # 重写后太短则用原始问题
            query_text = question

        # Step 3: 向量化查询
        query_vector = self.embedding_model.encode_single(query_text, is_query=True)

        # Step 4: 检索
        results = self.vector_store.search(query_vector, top_k=top_k, threshold=threshold)

        # Step 5: 构建上下文字符串
        context_parts = []
        for r in results:
            page_info = f"[第{r['page']}页]" if r.get("page") else ""
            context_parts.append(f"{page_info} {r['text']}")
        context_text = "\n\n".join(context_parts)

        return {
            "query_analysis": analysis,
            "results": results,
            "total_results": len(results),
            "context_text": context_text,
        }

    def format_context_for_llm(self, context_text: str, max_chars: int = 4000) -> str:
        """
        将检索结果格式化为适合LLM输入的上下文。

        Args:
            context_text: 原始拼接文本
            max_chars: 最大字符数（控制token消耗）

        Returns:
            截断后的格式化文本
        """
        if len(context_text) <= max_chars:
            return context_text
        return context_text[:max_chars] + "\n\n[内容已截断...]"
