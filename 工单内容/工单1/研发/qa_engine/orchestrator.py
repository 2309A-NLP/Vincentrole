"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
RAG系统主控制器 - 串联PDF解析 -> 索引 -> 检索 -> 生成
"""

import os
import time
from typing import Optional

from pdf_parser.parser import PDFParser
from pdf_parser.chunker import TextChunker
from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.vector_store import VectorStore
from qa_engine.retriever import Retriever
from qa_engine.generator import LLMGenerator

import config


class RAGSystem:
    """
    RAG问答系统主控制器。

    工作流程:
    1. 初始化 -> 加载/构建向量索引
    2. 问答 -> 检索 -> 生成
    3. 对比 -> RAG结果 vs 纯LLM结果
    """

    def __init__(self, llm_config: dict = None, use_cache: bool = True):
        self.use_cache = use_cache
        self.pdf_path = None
        self._is_ready = False

        try:
            # 初始化各个模块
            print("[RAGSystem] 初始化中...")

            # Embedding模型
            emb_cfg = config.EMBEDDING_CONFIG
            self.embedding_model = EmbeddingModel(
                model_name=emb_cfg["模型名称"],
                device=emb_cfg["设备"],
                normalize=emb_cfg["归一化"],
            )

            # 向量存储
            vec_cfg = config.VECTOR_STORE_CONFIG
            self.vector_store = VectorStore(
                dimension=self.embedding_model.dimension,
                index_path=vec_cfg["索引路径"],
                metadata_path=vec_cfg["元数据路径"],
            )

            # 检索器
            ret_cfg = config.RETRIEVAL_CONFIG
            self.retriever = Retriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
            )
            self.top_k = ret_cfg["检索数量"]
            self.threshold = ret_cfg["相关性阈值"]

            # LLM生成器
            if llm_config is None:
                llm_config = config.LLM_CONFIG
            self.generator = LLMGenerator.from_config({"LLM_CONFIG": llm_config})

            # PDF解析器
            self.pdf_parser = PDFParser()
            self.chunker = TextChunker(
                chunk_size=config.CHUNK_CONFIG["分块大小"],
                chunk_overlap=config.CHUNK_CONFIG["分块重叠"],
            )

            self._is_ready = False
            print("[RAGSystem] 初始化完成")

        except Exception as e:
            print(f"[RAGSystem] 初始化失败: {e}")
            raise

    def load_pdf(self, pdf_path: str) -> dict:
        """
        加载PDF文档，构建知识库。
        
        Returns:
            dict: {"status": "ok"/"cached", "pages": N, "chunks": N}
        """
        self.pdf_path = pdf_path

        # 尝试从缓存加载
        if self.use_cache and self.vector_store.load():
            try:
                meta = self.pdf_parser.get_metadata(pdf_path)
            except Exception as e:
                print(f"[RAGSystem] 获取元数据失败: {e}")
                meta = {"页数": 0}
            self._is_ready = True
            return {
                "status": "cached",
                "pages": meta["页数"],
                "chunks": self.vector_store.total_chunks,
            }

        print(f"[RAGSystem] 解析PDF: {pdf_path}")
        start = time.time()

        try:
            # Step 1: 解析PDF
            pages = self.pdf_parser.extract_text(pdf_path)
            print(f"  -> 提取 {len(pages)} 页文本")

            # Step 2: 文本分块
            chunks = self.chunker.chunk_pages(pages)
            print(f"  -> 切分为 {len(chunks)} 个chunk")

            # Step 3: 向量化
            texts = [c["text"] for c in chunks]
            embeddings = self.embedding_model.encode(texts, is_query=False)
            print(f"  -> 向量化完成，维度 {len(embeddings[0])}")

            # Step 4: 构建索引
            self.vector_store.build_index(chunks, embeddings)

            # Step 5: 持久化
            if self.use_cache:
                self.vector_store.save()

        except Exception as e:
            elapsed = time.time() - start
            print(f"[RAGSystem] PDF处理失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "elapsed": f"{elapsed:.1f}s",
            }

        elapsed = time.time() - start
        self._is_ready = True

        return {
            "status": "ok",
            "pages": len(pages),
            "chunks": len(chunks),
            "elapsed": f"{elapsed:.1f}s",
        }

    def ask(self, question: str) -> dict:
        """
        执行RAG问答流程。
        
        Args:
            question: 用户问题

        Returns:
            dict: {
                "question": "...",
                "rag_answer": "...",
                "source_chunks": [...],
                "query_analysis": {...},
                "elapsed": "...",
            }
        """
        if not self._is_ready:
            return {"error": "请先加载PDF文档", "question": question}

        start = time.time()

        try:
            # Step 1: 检索
            retrieval_result = self.retriever.retrieve(
                question,
                top_k=self.top_k,
                threshold=self.threshold,
            )

            # Step 2: 生成答案
            context = retrieval_result["context_text"]
            context_truncated = self.retriever.format_context_for_llm(context)

            gen_result = self.generator.generate(question, context_truncated)

            elapsed = time.time() - start

            # 提取来源页码
            source_pages = sorted(set(
                r["page"] for r in retrieval_result["results"] if r.get("page")
            ))

            return {
                "question": question,
                "rag_answer": gen_result.get("answer", ""),
                "source_chunks": retrieval_result["results"],
                "source_pages": source_pages,
                "query_analysis": retrieval_result["query_analysis"],
                "total_chunks_found": retrieval_result["total_results"],
                "elapsed": f"{elapsed:.2f}s",
                "error": gen_result.get("error"),
            }

        except Exception as e:
            elapsed = time.time() - start
            print(f"[RAGSystem] 问答处理失败: {e}")
            return {
                "question": question,
                "rag_answer": "",
                "source_chunks": [],
                "source_pages": [],
                "query_analysis": {},
                "total_chunks_found": 0,
                "elapsed": f"{elapsed:.2f}s",
                "error": str(e),
            }

    def ask_pure_llm(self, question: str) -> dict:
        """纯LLM回答（无RAG），用于对比"""
        start = time.time()
        try:
            result = self.generator.generate_pure_llm(question)
            elapsed = time.time() - start
            return {
                "question": question,
                "answer": result.get("answer", ""),
                "elapsed": f"{elapsed:.2f}s",
                "error": result.get("error"),
            }
        except Exception as e:
            elapsed = time.time() - start
            print(f"[RAGSystem] 纯LLM回答失败: {e}")
            return {
                "question": question,
                "answer": "",
                "elapsed": f"{elapsed:.2f}s",
                "error": str(e),
            }

    def compare(self, question: str) -> dict:
        """
        对比RAG vs 纯LLM的回答。

        Returns:
            dict: {"question": ..., "rag": {...}, "pure_llm": {...}}
        """
        rag_result = self.ask(question)
        llm_result = self.ask_pure_llm(question)

        return {
            "question": question,
            "rag": {
                "answer": rag_result.get("rag_answer", ""),
                "source_pages": rag_result.get("source_pages", []),
                "elapsed": rag_result.get("elapsed", ""),
            },
            "pure_llm": {
                "answer": llm_result.get("answer", ""),
                "elapsed": llm_result.get("elapsed", ""),
            },
        }

    @property
    def is_ready(self) -> bool:
        return self._is_ready


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    system = RAGSystem(use_cache=False)
    # 测试加载
    test_pdf = "/tmp/rag_wo2/RAG 工单/附件/招股说明书1.pdf"
    if os.path.exists(test_pdf):
        result = system.load_pdf(test_pdf)
        print(f"加载结果: {result}")
        # 测试问答
        answer = system.ask("武汉兴图新科电子股份有限公司的注册资本是多少？")
        print(f"答案: {answer['rag_answer'][:200]}")
