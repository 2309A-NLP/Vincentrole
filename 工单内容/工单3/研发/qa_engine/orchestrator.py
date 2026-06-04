"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
RAG系统主控制器 - 增强版（多文档支持 + 表格感知检索）
"""

import os
import time
import traceback
from typing import Optional, List

from pdf_parser.parser import PDFParser
from pdf_parser.chunker import TextChunker
from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.vector_store import VectorStore
from qa_engine.retriever import Retriever
from qa_engine.generator import LLMGenerator, detect_language

import config


class RAGSystem:
    """
    RAG问答系统主控制器 - 工单3增强版。

    优化点:
    1. 多文档支持 - 同时加载两份招股说明书
    2. 表格感知解析 - 文本+表格结构化解析
    3. 表格感知检索 - 列头匹配+表格类型感知
    4. 增量加载 - 先加载doc1再加载doc2不丢失已有索引
    5. 文档来源追踪 - 每个chunk带 source_file 标记
    """

    def __init__(self, llm_config: dict = None, use_cache: bool = True):
        self.use_cache = use_cache
        self.pdf_paths = []

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
        """加载单个PDF文档（支持增量：多个PDF叠加到同一索引）"""
        if pdf_path not in self.pdf_paths:
            self.pdf_paths.append(pdf_path)
        return self._load_pdf_internal(pdf_path, incremental=len(self.pdf_paths) > 1)

    def load_pdfs(self, pdf_paths: List[str]) -> dict:
        """批量加载多个PDF文档"""
        results = []
        total_pages = 0
        total_chunks = 0

        for pdf_path in pdf_paths:
            result = self.load_pdf(pdf_path)
            results.append(result)
            if result.get("status") in ("ok", "cached"):
                total_pages += result.get("pages", 0)
                total_chunks += result.get("chunks", 0)

        return {
            "status": "ok",
            "pages": total_pages,
            "chunks": total_chunks,
            "files_loaded": len(results),
            "details": results,
        }

    def _load_pdf_internal(self, pdf_path: str, incremental: bool = False) -> dict:
        """内部PDF加载逻辑"""
        try:
            if not os.path.exists(pdf_path):
                return {"status": "error", "error": f"PDF文件不存在: {pdf_path}"}

            # 尝试从缓存加载（仅首次加载非增量模式）
            if self.use_cache and not incremental and self.vector_store.load():
                meta = self.pdf_parser.get_metadata(pdf_path)
                self._is_ready = True
                return {
                    "status": "cached",
                    "pages": meta["页数"],
                    "chunks": self.vector_store.total_chunks,
                    "file": os.path.basename(pdf_path),
                }

            print(f"[RAGSystem] 解析PDF: {pdf_path}")
            start = time.time()

            # 结构化解析（增强表格提取）
            try:
                structured_pages = self.pdf_parser.extract_structured(pdf_path)
            except Exception as e:
                print(f"  [降级] 结构化解析失败: {e}，使用普通解析")
                pages = self.pdf_parser.extract_text(pdf_path)
                structured_pages = []
                for p in pages:
                    sp = type("obj", (object,), {
                        "page_num": p["page"], "text": p["text"],
                        "tables": [], "headings": [], "paragraphs": [],
                        "source_file": os.path.basename(pdf_path)
                    })()
                    structured_pages.append(sp)

            print(f"  -> 提取 {len(structured_pages)} 页")
            table_count = sum(len(p.tables) for p in structured_pages)
            print(f"  -> 检测到 {table_count} 个表格")

            # 语义分块（含增强表格处理）
            new_chunks = self.chunker.chunk_structured(structured_pages)
            print(f"  -> 语义分块完成，新增 {len(new_chunks)} 个chunk")

            # 如果没有现有chunks（增量加载），则合并
            if incremental and self.vector_store.total_chunks > 0:
                all_chunks = self.vector_store.chunks + new_chunks
            else:
                all_chunks = new_chunks

            # 向量化
            texts = [c["text"] for c in all_chunks]
            embeddings = self.embedding_model.encode(texts, is_query=False)
            print(f"  -> 向量化完成，维度 {len(embeddings[0])}")

            # 重建索引
            self.vector_store.build_index(all_chunks, embeddings)

            if self.use_cache:
                self.vector_store.save()

            elapsed = time.time() - start
            self._is_ready = True

            return {
                "status": "ok",
                "pages": len(structured_pages),
                "chunks": len(new_chunks),
                "total_chunks": len(all_chunks),
                "elapsed": f"{elapsed:.1f}s",
                "file": os.path.basename(pdf_path),
                "tables_found": table_count,
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

            source_pages = []
            for r in retrieval_result["results"]:
                p = r.get("page")
                if p:
                    source_pages.append(str(p))
            source_pages = sorted(set(source_pages))
            source_files = list(set(
                r.get("source_file", "") for r in retrieval_result["results"] if r.get("source_file")
            ))

            return {
                "question": question,
                "rag_answer": gen_result.get("answer", ""),
                "source_chunks": retrieval_result["results"],
                "source_pages": source_pages,
                "source_files": source_files,
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
                "source_files": rag_result.get("source_files", []),
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
