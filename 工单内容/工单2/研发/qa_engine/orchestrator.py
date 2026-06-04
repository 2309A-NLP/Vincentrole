"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
RAG系统主控制器 - 优化版（增强容错、计时、多语言）
"""

import os
import time
import traceback
from typing import Optional

from pdf_parser.parser import PDFParser
from pdf_parser.chunker import TextChunker
from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.vector_store import VectorStore
from qa_engine.retriever import Retriever
from qa_engine.generator import LLMGenerator, detect_language

import config


class RAGSystem:
    """
    RAG问答系统主控制器 - 优化版（支持容错机制）

    优化工作流程:
    1. 结构化PDF解析（标题+段落+表格）
    2. 语义分块（基于段落边界）
    3. 混合检索（向量+FAISS + BM25 + RRF融合）
    4. 重排序（关键词匹配度+标题奖励）
    5. 精准生成（支持中英文、容错）
    """

    def __init__(self, llm_config: dict = None, use_cache: bool = True):
        self.use_cache = use_cache
        self.pdf_path = None
        self.vector_store = None  # 防御: 确保异常时也有这个属性

        try:
            print("[RAGSystem] 初始化中...")

            emb_cfg = config.EMBEDDING_CONFIG
            self.embedding_model = EmbeddingModel(
                model_name=emb_cfg["模型名称"],
                device=emb_cfg["设备"],
                normalize=emb_cfg["归一化"],
            )

            vec_cfg = config.VECTOR_STORE_CONFIG
            self.vector_store = VectorStore(
                dimension=self.embedding_model.dimension,
                index_path=vec_cfg["索引路径"],
                metadata_path=vec_cfg["元数据路径"],
            )

            self.retriever = Retriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                expand_query=config.RETRIEVAL_CONFIG.get("查询扩展", True),
                rerank_top_k=config.RETRIEVAL_CONFIG.get("重排序数量", 5),
            )
            self.top_k = config.RETRIEVAL_CONFIG["检索数量"]
            self.threshold = config.RETRIEVAL_CONFIG["相关性阈值"]

            if llm_config is None:
                llm_config = config.LLM_CONFIG
            self.generator = LLMGenerator.from_config({"LLM_CONFIG": llm_config})

            self.pdf_parser = PDFParser()
            self.chunker = TextChunker(
                chunk_size=config.CHUNK_CONFIG["分块大小"],
                chunk_overlap=config.CHUNK_CONFIG["分块重叠"],
            )

            self._is_ready = False
            print("[RAGSystem] 初始化完成")
        except Exception as e:
            print(f"[RAGSystem] 初始化失败: {e}")
            self._is_ready = False
            self._init_error = str(e)

    def load_pdf(self, pdf_path: str) -> dict:
        """加载PDF文档（含容错机制）"""
        self.pdf_path = pdf_path

        try:
            # 容错：文件不存在
            if not os.path.exists(pdf_path):
                return {"status": "error", "error": f"PDF文件不存在: {pdf_path}"}

            # 容错：初始化未完成
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                init_err = getattr(self, '_init_error', '初始化未完成')
                return {"status": "error", "error": f"系统初始化失败: {init_err}"}

            # 从缓存加载
            if self.use_cache and self.vector_store.load():
                meta = self.pdf_parser.get_metadata(pdf_path)
                self._is_ready = True
                return {
                    "status": "cached",
                    "pages": meta["页数"],
                    "chunks": self.vector_store.total_chunks,
                }

            print(f"[RAGSystem] 解析PDF: {pdf_path}")
            start = time.time()

            # 结构化解析（容错：如果失败则降级为普通解析）
            try:
                structured_pages = self.pdf_parser.extract_structured(pdf_path)
            except Exception as e:
                print(f"  [降级] 结构化解析失败: {e}，使用普通解析")
                pages = self.pdf_parser.extract_text(pdf_path)
                structured_pages = [{"page": p["page"], "text": p["text"],
                                     "type": "text", "heading": ""} for p in pages]
            print(f"  -> 提取 {len(structured_pages)} 页")

            # 语义分块
            chunks = self.chunker.chunk_structured(structured_pages)
            print(f"  -> 语义分块完成，共 {len(chunks)} 个chunk")

            # 向量化（容错：单个文本失败不影响整体）
            texts = [c["text"] for c in chunks]
            embeddings = self.embedding_model.encode(texts, is_query=False)
            print(f"  -> 向量化完成，维度 {len(embeddings[0])}")

            # 构建索引
            self.vector_store.build_index(chunks, embeddings)

            if self.use_cache:
                self.vector_store.save()

            elapsed = time.time() - start
            self._is_ready = True

            return {
                "status": "ok",
                "pages": len(structured_pages),
                "chunks": len(chunks),
                "elapsed": f"{elapsed:.1f}s",
            }
        except Exception as e:
            print(f"[RAGSystem] 加载失败: {e}")
            traceback.print_exc()
            return {"status": "error", "error": f"加载失败: {str(e)}"}

    def ask(self, question: str) -> dict:
        """执行RAG问答流程（含容错、计时）"""
        if not self._is_ready:
            return {"error": "请先加载PDF文档", "question": question}

        start = time.time()

        try:
            retrieval_result = self.retriever.retrieve(
                question,
                top_k=self.top_k,
                threshold=self.threshold,
            )

            context = retrieval_result["context_text"]
            context_truncated = self.retriever.format_context_for_llm(context)

            gen_result = self.generator.generate(question, context_truncated)

            elapsed = time.time() - start

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
                "elapsed_seconds": round(elapsed, 2),
                "error": gen_result.get("error"),
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                "question": question,
                "rag_answer": "",
                "error": f"问答处理失败: {str(e)}",
                "elapsed": f"{elapsed:.2f}s",
                "elapsed_seconds": round(elapsed, 2),
            }

    def ask_pure_llm(self, question: str) -> dict:
        """纯LLM回答（无RAG），用于对比"""
        start = time.time()
        try:
            lang = detect_language(question)
            result = self.generator.generate_pure_llm(question, lang=lang)
            elapsed = time.time() - start
            return {
                "question": question,
                "answer": result.get("answer", ""),
                "elapsed": f"{elapsed:.2f}s",
                "elapsed_seconds": round(elapsed, 2),
                "error": result.get("error"),
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                "question": question,
                "answer": "",
                "elapsed": f"{elapsed:.2f}s",
                "elapsed_seconds": round(elapsed, 2),
                "error": str(e),
            }

    def compare(self, question: str) -> dict:
        """对比RAG vs 纯LLM的回答"""
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
            "rag_error": rag_result.get("error"),
            "llm_error": llm_result.get("error"),
        }

    @property
    def is_ready(self) -> bool:
        return self._is_ready
