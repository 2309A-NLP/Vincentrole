"""
工单编号: 人工智能NLP-RAG-混合检索任务
RAG系统主控制器 - 工单6混合检索版（向量检索+全文检索+混合检索+多重排器）

优化点:
1. 向量检索（召回+重排）- Milvus存储
2. 全文检索 - Whoosh倒排索引（布尔/短语/模糊匹配）
3. 混合检索 - 向量+全文融合（RRF/加权平均/投票）
4. 3种重排算法 - TF-IDF / LLM / CrossEncoder(bge-reranker-v2-m3)
5. 多轮对话 - 上下文维护+指代消解
6. 表格+图像感知检索
"""

import os
import time
import traceback
import base64
from typing import Optional, List

from pdf_parser.parser import PDFParser
from pdf_parser.chunker import TextChunker
from pdf_parser.image_extractor import ImageExtractor
from knowledge_base.embeddings import EmbeddingModel
from knowledge_base.image_embeddings import ImageEmbeddingEncoder
from knowledge_base.image_store import ImageStore
from knowledge_base.fulltext_store import FullTextStore
from knowledge_base.reranker import create_reranker
from qa_engine.retriever import Retriever
from qa_engine.generator import LLMGenerator, detect_language
from qa_engine.conversation import ConversationManager

import config

# 根据配置选择向量存储后端
VECTOR_STORE_BACKEND = getattr(config, "VECTOR_STORE_BACKEND", "faiss")
if VECTOR_STORE_BACKEND == "milvus":
    from knowledge_base.milvus_store import MilvusVectorStore as VectorStoreImpl
else:
    from knowledge_base.vector_store import VectorStore as VectorStoreImpl


class RAGSystem:
    """
    RAG问答系统主控制器 - 工单6混合检索版（向量/全文/混合3种模式 + 多重排器）。
    """

    def __init__(self, llm_config: dict = None, use_cache: bool = True):
        self.use_cache = use_cache
        self.pdf_paths = []
        self._retrieval_mode = "hybrid"

        try:
            print(f"[RAGSystem] 初始化中... (存储后端: {VECTOR_STORE_BACKEND}, 检索模式: hybrid)")

            emb_cfg = config.EMBEDDING_CONFIG
            self.embedding_model = EmbeddingModel(
                model_name=emb_cfg["模型名称"],
                device=emb_cfg["设备"],
                normalize=emb_cfg["归一化"],
                model_registry=getattr(config, "EMBEDDING_MODELS", {}),
            )
            print(f"[RAGSystem] 嵌入模型: {emb_cfg['模型名称']} (维度:{self.embedding_model.dimension})")

            # ---- 全文检索初始化（工单6新增） ----
            ft_cfg = getattr(config, "FULLTEXT_CONFIG", {})
            self.fulltext_enabled = ft_cfg.get("启用", True)
            if self.fulltext_enabled:
                try:
                    self._fulltext_store = FullTextStore(index_dir=ft_cfg["索引目录"])
                    print(f"[RAGSystem] 全文检索已启用: {ft_cfg['索引目录']}")
                except Exception as e:
                    print(f"[RAGSystem] 全文检索初始化失败: {e}")
                    self._fulltext_store = None
                    self.fulltext_enabled = False
            else:
                self._fulltext_store = None

            # ---- 重排器初始化（工单6新增） ----
            reranker_cfg = getattr(config, "RERANKER_CONFIG", {})
            self.reranker_enabled = reranker_cfg.get("启用重排", True)
            self._reranker = None
            if self.reranker_enabled:
                try:
                    default_type = reranker_cfg.get("默认重排器", "tfidf")
                    model_path = reranker_cfg.get("reranker_model_path", "")
                    self._reranker = create_reranker(
                        reranker_type=default_type,
                        generator=None,  # lazy init when needed
                        model_path=model_path,
                        llm_config=getattr(config, "KIMI_CONFIG", {}),
                    )
                    print(f"[RAGSystem] 重排器已加载: {default_type}")
                except Exception as e:
                    print(f"[RAGSystem] 重排器初始化失败: {e}")
                    self._reranker = None

            # 检索模式
            self._retrieval_mode = getattr(config, "HYBRID_SEARCH_CONFIG", {}).get("默认检索模式", "hybrid")

            # 根据后端创建向量存储
            if VECTOR_STORE_BACKEND == "milvus":
                mil_cfg = config.MILVUS_CONFIG
                self.vector_store = VectorStoreImpl(
                    dimension=self.embedding_model.dimension,
                    collection_name=mil_cfg["collection_name"],
                    metadata_path=mil_cfg["metadata_path"],
                    milvus_host=mil_cfg["host"],
                    milvus_port=mil_cfg["port"],
                )
            else:
                vec_cfg = config.VECTOR_STORE_CONFIG
                self.vector_store = VectorStoreImpl(
                    dimension=self.embedding_model.dimension,
                    index_path=vec_cfg["索引路径"],
                    metadata_path=vec_cfg["元数据路径"],
                )

            self.retriever = Retriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                fulltext_store=self._fulltext_store,
                expand_query=config.RETRIEVAL_CONFIG.get("查询扩展", True),
                rerank_top_k=config.RETRIEVAL_CONFIG.get("重排序数量", 5),
                hybrid_config=getattr(config, "HYBRID_SEARCH_CONFIG", {}),
                retrieval_mode=getattr(config, "HYBRID_SEARCH_CONFIG", {}).get("默认检索模式", "hybrid"),
            )
            self.top_k = config.RETRIEVAL_CONFIG["检索数量"]
            self.threshold = config.RETRIEVAL_CONFIG["相关性阈值"]

            if llm_config is None:
                llm_config = config.LLM_CONFIG
            self.generator = LLMGenerator.from_config({"LLM_CONFIG": llm_config})

            # Kimi多模态生成器（图像问题专用）
            try:
                self.kimi_generator = LLMGenerator.kimi_generator()
                print("[RAGSystem] Kimi多模态生成器已加载")
            except Exception as e:
                self.kimi_generator = None
                print(f"[RAGSystem] Kimi多模态生成器不可用: {e}")

            self.pdf_parser = PDFParser()
            self.chunker = TextChunker(
                chunk_size=config.CHUNK_CONFIG["分块大小"],
                chunk_overlap=config.CHUNK_CONFIG["分块重叠"],
            )

            # ---- 图像处理初始化（工单4新增） ----
            img_cfg = getattr(config, "IMAGE_CONFIG", {})
            self.image_enabled = img_cfg.get("启用图像提取", True)

            # 图像编码器（支持CLIP/BGE回退）
            self.image_encoder = ImageEmbeddingEncoder(
                use_clip=False,  # 国内网络暂不支持下载CLIP，使用BGE回退
                device=img_cfg.get("可用设备", "cpu"),
                bge_encoder=self.embedding_model,
            )

            # 图像索引存储
            self.image_store = ImageStore(encoder=self.image_encoder)

            # 图像提取器
            self.image_extractor = ImageExtractor(config=img_cfg)

            self._image_stats = {"total_images": 0, "by_type": {}}
            # ---- 图像处理初始化结束 ----

            # ---- 多轮对话初始化（工单5新增） ----
            conv_cfg = getattr(config, "CONVERSATION_CONFIG", {})
            self.conversation_enabled = conv_cfg.get("启用多轮对话", True)
            self.conversation = ConversationManager(
                max_history=conv_cfg.get("历史轮数", 3),
                context_window=conv_cfg.get("上下文窗口", 2000),
            )
            print(f"[RAGSystem] 多轮对话已启用（保留{conv_cfg.get('历史轮数', 3)}轮历史）")
            # ---- 多轮对话初始化结束 ----

            self._is_ready = False
            # 检测 Milvus 是否已有数据（自动初始化场景）
            if VECTOR_STORE_BACKEND == "milvus" and hasattr(self, 'vector_store'):
                try:
                    store = self.vector_store
                    if hasattr(store, 'client') and store.client:
                        stats = store.client.get_collection_stats(store.collection_name)
                        row_count = stats.get("row_count", 0)
                        if row_count > 0:
                            self._is_ready = True
                            print(f"[RAGSystem] Milvus已有数据 ({row_count} 条), 标记为就绪")
                except Exception:
                    pass
            print("[RAGSystem] 初始化完成（含图像处理模块）")
        except Exception as e:
            print(f"[RAGSystem] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._is_ready = False
            self._init_error = str(e)

    def set_retrieval_mode(self, mode: str):
        """切换检索模式: vector / fulltext / hybrid"""
        if mode in ("vector", "fulltext", "hybrid"):
            self._retrieval_mode = mode
            if hasattr(self, 'retriever') and self.retriever:
                self.retriever.set_retrieval_mode(mode)

    def set_reranker(self, reranker_type: str):
        """切换重排器: tfidf / llm / adaptive / crossencoder / none"""
        if reranker_type == "none":
            self._reranker = None
            if hasattr(self, 'retriever') and self.retriever:
                self.retriever.set_reranker(None)
            print("[RAGSystem] 重排器已关闭")
            return
        reranker_cfg = getattr(config, "RERANKER_CONFIG", {})
        model_path = reranker_cfg.get("reranker_model_path", "")
        try:
            if reranker_type == "llm":
                # 使用Kimi API（国内访问快）
                from knowledge_base.reranker import LLMReranker
                kimi_cfg = getattr(config, "KIMI_CONFIG", {})
                self._reranker = LLMReranker(generator=self.generator, llm_config=kimi_cfg)
            else:
                self._reranker = create_reranker(
                    reranker_type=reranker_type,
                    model_path=model_path,
                )
            if hasattr(self, 'retriever') and self.retriever:
                self.retriever.set_reranker(self._reranker)
            print(f"[RAGSystem] 重排器切换为: {reranker_type}")
        except Exception as e:
            print(f"[RAGSystem] 重排器切换失败: {e}")

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
                
                # 即使缓存命中，也要确保图像索引已构建
                image_count = 0
                if self.image_enabled and self.image_store.get_image_count() == 0:
                    try:
                        extracted_images = self.image_extractor.extract_from_pdf(pdf_path)
                        if extracted_images:
                            img_dicts = [img.to_dict() for img in extracted_images]
                            self.image_store.add_images(img_dicts, source_file=os.path.basename(pdf_path))
                            image_count = len(extracted_images)
                            print(f"  -> [缓存补充] 图像索引已构建: {image_count} 个图像")
                    except Exception as e:
                        print(f"  -> [缓存补充] 图像提取失败: {e}")
                
                return {
                    "status": "cached",
                    "pages": meta["页数"],
                    "chunks": self.vector_store.total_chunks,
                    "file": os.path.basename(pdf_path),
                    "images_found": image_count,
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

            # ---- 图像处理（工单4新增） ----
            image_count = 0
            image_types = {}
            image_chunks = []
            if self.image_enabled:
                try:
                    extracted_images = self.image_extractor.extract_from_pdf(pdf_path)
                    if extracted_images:
                        # 添加到图像索引
                        img_dicts = [img.to_dict() for img in extracted_images]
                        added = self.image_store.add_images(img_dicts, source_file=os.path.basename(pdf_path))

                        # 统计
                        image_count = len(extracted_images)
                        for img in extracted_images:
                            it = img.image_type
                            image_types[it] = image_types.get(it, 0) + 1

                        # 生成图像chunk（用于文本检索匹配）
                        for idx, img in enumerate(extracted_images):
                            desc_text = f"【图像】第{img.page_num}页"
                            if img.image_type:
                                desc_text += f" 类型:{img.image_type}"
                            # 使用丰富描述作为chunk文本，提高检索命中率
                            rich_desc = img.description or img.surrounding_text or img.caption_text
                            if rich_desc:
                                desc_text += f" 内容:{rich_desc[:500]}"
                            elif img.caption_text:
                                desc_text += f" 标题:{img.caption_text}"

                            image_chunks.append({
                                "text": desc_text,
                                "page": img.page_num,
                                "type": "image",
                                "heading": img.caption_text or img.image_type,
                                "source_file": os.path.basename(pdf_path),
                                "image_info": {
                                    "image_path": img.image_path,
                                    "image_type": img.image_type,
                                    "caption_text": img.caption_text,
                                    "has_caption": img.has_caption,
                                },
                            })

                        print(f"  -> 图像处理完成，提取 {image_count} 个图像")

                        # 更新全局统计
                        self._image_stats["total_images"] += image_count
                        for k, v in image_types.items():
                            self._image_stats["by_type"][k] = self._image_stats["by_type"].get(k, 0) + v
                except Exception as e:
                    print(f"  -> 图像处理跳过: {e}")

            # 合并图像chunk到文本chunks
            if image_chunks:
                new_chunks.extend(image_chunks)
                print(f"  -> 新增 {len(image_chunks)} 个图像描述chunk")
            # ---- 图像处理结束 ----

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

            # 构建全文索引（工单6新增）
            if self.fulltext_enabled and self._fulltext_store:
                try:
                    ft_chunks = []
                    for i, c in enumerate(all_chunks):
                        ft_chunks.append({
                            "chunk_id": i,
                            "text": c.get("text", ""),
                            "heading": c.get("heading", ""),
                            "source_file": c.get("source_file", ""),
                            "page": c.get("page", 0),
                            "type": c.get("type", "text"),
                        })
                    self._fulltext_store.build_index(ft_chunks)
                    print(f"  -> 全文索引构建完成: {len(ft_chunks)} 个文档块")
                except Exception as e:
                    print(f"  -> 全文索引构建失败: {e}")

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
                "images_found": image_count,
                "image_types": image_types,
            }
        except Exception as e:
            print(f"[RAGSystem] 加载失败: {e}")
            traceback.print_exc()
            return {"status": "error", "error": f"加载失败: {str(e)}"}

    def ask(self, question: str) -> dict:
        """
        执行RAG问答流程（含多轮对话支持）。
        
        流程:
        1. 指代消解（将"他/这个公司"等代词解析为具体实体）
        2. 检索（使用消解后的问题）
        3. 生成答案（注入历史上下文）
        4. 更新对话历史
        """
        if not self._is_ready:
            return {"error": "请先加载PDF文档", "question": question}

        start = time.time()

        try:
            # ---- 多轮对话：指代消解 ----
            original_question = question
            resolution_info = {}
            if self.conversation_enabled:
                question, resolution_info = self.conversation.resolve_references(question)
                if question != original_question:
                    print(f"  [指代消解] {original_question[:30]}... → {question[:30]}...")
            # ---- 指代消解结束 ----

            retrieval_result = self.retriever.retrieve(
                question,
                top_k=self.top_k,
                threshold=self.threshold,
                mode=self._retrieval_mode,
            )

            context = retrieval_result["context_text"]
            context_truncated = self.retriever.format_context_for_llm(context)

            # ---- 图像检索（工单4新增） ----
            image_results = []
            image_context = ""
            if self.image_enabled and self.image_store.get_image_count() > 0:
                try:
                    image_results = self.image_store.search(
                        query_text=question,
                        top_k=5,
                    )
                    # 将高分的图像描述文本追加到上下文中，供LLM生成答案
                    image_descriptions = []
                    image_page_texts = set()  # 记录已收集的页码
                    for ir in image_results:
                        score = ir.get("score", 0)
                        if score >= 0.3:  # 降低阈值以捕获更多相关图像
                            img_type = ir.get("image_type", "图像")
                            caption = ir.get("caption_text", "")
                            desc = ir.get("description", "")
                            surround = ir.get("surrounding_text", "")
                            page = ir.get("page_num", 0)

                            # 优先用最丰富的文本源
                            context_text = desc or surround or caption
                            if context_text:
                                image_descriptions.append(
                                    f"[{img_type}(第{page}页,相似度{score:.2f})] {context_text[:500]}"
                                )
                                image_page_texts.add(page)

                    if image_descriptions:
                        image_context = "\n\n【以上是基于图像内容提取的描述文本，回答问题时可参考】\n" + "\n---\n".join(image_descriptions[:5])
                except Exception as e:
                    print(f"  [图像检索跳过]: {e}")
            # ---- 图像检索结束 ----

            # ---- Kimi多模态生成（图像问题专用，支持多图回退） ----
            # 收集所有可用图片（分数>=0.3且文件存在）
            available_images = []
            for ir in image_results:
                s = ir.get("score", 0)
                p = ir.get("image_path", "")
                if s >= 0.3 and p and os.path.exists(p):
                    available_images.append({"path": p, "score": s})
            
            use_multimodal = (
                self.kimi_generator is not None
                and len(available_images) > 0
            )
            
            gen_result = None
            tried_images = []
            
            if use_multimodal:
                # 尝试最多3张图，直到得到有效答案
                for img_info in available_images[:3]:
                    img_path = img_info["path"]
                    try:
                        with open(img_path, "rb") as f:
                            img_base64 = base64.b64encode(f.read()).decode("utf-8")
                        
                        result = self.kimi_generator.generate_with_image(
                            question=question,
                            context=context_truncated,
                            image_base64=img_base64,
                        )
                        tried_images.append(img_path)
                        
                        answer = result.get("answer", "")
                        # 检查答案是否有效（非空且不是"未找到"类回复）
                        is_valid = (
                            answer 
                            and len(answer) > 10
                            and "未找到" not in answer
                            and "无法回答" not in answer
                            and "不包含" not in answer
                        )
                        
                        if is_valid:
                            gen_result = result
                            gen_result["used_multimodal"] = True
                            gen_result["multimodal_image"] = img_path
                            print(f"  [多模态成功] 使用图片: {img_path.split('/')[-1]}")
                            break
                        else:
                            print(f"  [多模态回退] {img_path.split('/')[-1]} 回答无效，尝试下一张...")
                    except Exception as e:
                        print(f"  [多模态异常] {img_path.split('/')[-1]}: {e}")
                        tried_images.append(img_path)
                        continue
                
                # 所有图片都失败，降级到文字模式
                if not gen_result:
                    print(f"  [多模态降级] 尝试了{len(tried_images)}张图均无效，回退文字模式")
                    # 注入历史上下文
                    history_ctx = self.conversation.get_history_context() if self.conversation_enabled else ""
                    full_context = (history_ctx + "\n" if history_ctx else "") + context_truncated + image_context
                    gen_result = self.generator.generate(question, full_context)
                    gen_result["used_multimodal"] = False
                    gen_result["tried_images"] = tried_images
            else:
                # 注入历史上下文
                history_ctx = self.conversation.get_history_context() if self.conversation_enabled else ""
                full_context = (history_ctx + "\n---\n" if history_ctx else "") + context_truncated + image_context
                gen_result = self.generator.generate(question, full_context)
                gen_result["used_multimodal"] = False
            # ---- Kimi多模态生成结束 ----

            # ---- 更新对话历史（工单5新增） ----
            answer_text = gen_result.get("answer", "")
            if self.conversation_enabled and answer_text:
                self.conversation.add_turn(original_question, answer_text)
            # ---- 更新对话历史结束 ----

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
                "question": original_question,  # 返回原始问题
                "resolved_question": question,  # 消解后的问题
                "rag_answer": gen_result.get("answer", ""),
                "source_chunks": retrieval_result["results"],
                "source_pages": source_pages,
                "source_files": source_files,
                "query_analysis": retrieval_result["query_analysis"],
                "total_chunks_found": retrieval_result["total_results"],
                "image_results": image_results,      # 图像检索结果
                "image_count": len(image_results),
                "used_multimodal": gen_result.get("used_multimodal", False),
                "multimodal_image": gen_result.get("multimodal_image", ""),
                "conversation_status": self.conversation.get_status() if self.conversation_enabled else {},
                "resolution_info": resolution_info,
                "retrieval_mode": retrieval_result.get("retrieval_mode", self._retrieval_mode),
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

    def reset_conversation(self):
        """清空多轮对话历史"""
        if self.conversation_enabled:
            self.conversation.reset()
            print("[RAGSystem] 对话历史已清空")

    # ============================================================
    # 图像查询接口（工单4新增）
    # ============================================================
    def get_images_for_page(self, page_num: int, source_file: str = "") -> list:
        """获取指定页面的图像列表（供UI使用）"""
        if not self.image_enabled:
            return []
        return self.image_store.search_by_page(page_num, source_file)

    def search_images(self, query: str, top_k: int = 3) -> list:
        """文本查询匹配图像（跨模态检索）"""
        if not self.image_enabled or self.image_store.get_image_count() == 0:
            return []
        return self.image_store.search(query_text=query, top_k=top_k)

    @property
    def image_stats(self) -> dict:
        """图像处理统计"""
        return {
            "total_images": self._image_stats["total_images"],
            "by_type": dict(self._image_stats["by_type"]),
            "indexed": self.image_store.get_image_count(),
        }
