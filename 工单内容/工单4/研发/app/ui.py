"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
Streamlit用户界面 - 支持多文档上传、表格感知检索对比
"""

import os
import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，避免 Streamlit 运行时路径问题
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import config
from qa_engine.orchestrator import RAGSystem
from qa_engine.query_understanding import detect_language


def _(text_zh: str, text_en: str) -> str:
    """多语言文本：根据用户语言设置返回对应文字"""
    lang = st.session_state.get("ui_lang", "zh")
    return text_zh if lang == "zh" else text_en


def init_session_state():
    """初始化会话状态"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = False
    if "loaded_pdfs" not in st.session_state:
        st.session_state.loaded_pdfs = []
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "提供商": config.LLM_CONFIG.get("提供商", "deepseek"),
            "模型": config.LLM_CONFIG.get("模型", "deepseek-v4-flash"),
            "API地址": config.LLM_CONFIG.get("API地址", "https://api.deepseek.com/v1"),
            "API密钥": config.LLM_CONFIG.get("API密钥", ""),
        }
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "zh"


def inject_voice_button():
    """在聊天输入框内嵌入麦克风按钮"""
    lang = st.session_state.get("ui_lang", "zh")
    speech_lang = "zh-CN" if lang == "zh" else "en-US"
    btn_title = _("语音输入", "Voice input")
    alert_msg = _("浏览器不支持语音识别", "Speech not supported")

    html = (
        '<style>'
        '.hmc-btn{position:absolute !important;right:50px !important;bottom:6px !important;'
        'z-index:999 !important;width:26px;height:26px;border-radius:50%;'
        'border:none;background:transparent;cursor:pointer;font-size:15px;'
        'display:flex;align-items:center;justify-content:center;padding:0;'
        'opacity:.45;transition:all .2s;line-height:1}'
        '.hmc-btn:hover{opacity:1;background:rgba(0,0,0,.04)}'
        '.hmc-btn.lstn{opacity:1;color:#e74c3c;background:rgba(231,76,60,.1);animation:hmp 1s infinite}'
        '@keyframes hmp{0%,100%{transform:scale(1)}50%{transform:scale(1.2)}}'
        f'</style><script>'
        f'(function(){{'
        f'var T="{btn_title}",A="{alert_msg}",L="{speech_lang}";'
        'function ti(){'
        'var ci=document.querySelector(\'[data-testid="stChatInput"]\');'
        'if(!ci||ci.querySelector(\'.hmc-btn\'))return;'
        'var btn=ci.querySelector(\'[data-testid="stChatInputSubmitButton"]\');'
        'if(!btn)return;'
        'var mic=document.createElement(\'button\');'
        'mic.className=\'hmc-btn\';mic.innerHTML=\'🎤\';mic.type=\'button\';mic.title=T;'
        'btn.parentNode.insertBefore(mic,btn);var rec=null;'
        'mic.onclick=function(e){'
        'e.preventDefault();e.stopPropagation();'
        'if(rec){rec.stop();rec=null;return}'
        'var R=window.SpeechRecognition||window.webkitSpeechRecognition;'
        'if(!R){alert(A);return}'
        'var r=new R();r.lang=L;r.interimResults=false;r.maxAlternatives=1;rec=r;'
        'mic.classList.add(\'lstn\');'
        'r.onresult=function(ev){'
        'var t=ev.results[0][0].transcript;mic.classList.remove(\'lstn\');rec=null;'
        'var ta=ci.querySelector(\'textarea\');'
        'if(ta){'
        'var s=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,\'value\').set;'
        's.call(ta,t);ta.dispatchEvent(new Event(\'input\',{bubbles:true}));'
        'ta.dispatchEvent(new Event(\'change\',{bubbles:true}));ta.focus();'
        'setTimeout(function(){btn.click()},200)}'
        '};'
        'r.onerror=function(){mic.classList.remove(\'lstn\');rec=null};'
        'r.onend=function(){mic.classList.remove(\'lstn\');rec=null};'
        'r.start()};return}'
        'if(!window.__hv){window.__hv=true;ti();'
        'new MutationObserver(function(){if(!document.querySelector(\'.hmc-btn\'))ti()})'
        '.observe(document.body,{childList:true,subtree:true})}'
        '})();</script>'
    )
    st.html(html, unsafe_allow_javascript=True)


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown(f"## {_('配置', 'Settings')}")

        # 语言切换
        lang = st.selectbox(
            _("界面语言", "Language"),
            options=["zh", "en"],
            format_func=lambda x: "中文" if x == "zh" else "English",
            index=0 if st.session_state.ui_lang == "zh" else 1,
            key="lang_selector",
        )
        if lang != st.session_state.ui_lang:
            st.session_state.ui_lang = lang
            st.rerun()

        st.divider()

        # PDF文档上传（支持多文档）
        st.markdown(f"### {_('1. 上传PDF文档', '1. Upload PDFs')}")
        uploaded_file = st.file_uploader(
            _("选择PDF文件（支持多个）", "Select PDF files (multiple supported)"),
            type=["pdf"],
            accept_multiple_files=False,
            help=_("支持上传招股说明书等PDF文档", "Supports IPO prospectus and other PDF documents"),
        )

        # 默认文档选择
        st.markdown(_("快捷加载：", "Quick load:"))
        default_pdf1 = os.path.join(config.DATA_DIR, "招股说明书1.pdf")
        default_pdf2 = os.path.join(config.DATA_DIR, "招股说明书2.pdf")

        col1, col2 = st.columns(2)
        with col1:
            use_doc1 = st.checkbox(
                _("招股说明书1（兴图新科）", "Doc1: Xingtu Xinke"),
                value=True,
                key="use_doc1",
            )
        with col2:
            use_doc2 = st.checkbox(
                _("招股说明书2（力源信息）", "Doc2: Liyuan Info"),
                value=True,
                key="use_doc2",
            )

        col_load, col_clear = st.columns(2)
        with col_load:
            load_btn = st.button(_("加载文档", "Load Docs"), type="primary", use_container_width=True)
        with col_clear:
            clear_btn = st.button(_("清空缓存", "Clear Cache"), use_container_width=True)

        if load_btn:
            pdf_paths = []

            # 处理上传的文件
            if uploaded_file:
                os.makedirs(config.DATA_DIR, exist_ok=True)
                save_path = os.path.join(config.DATA_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(save_path)

            # 默认文档
            if use_doc1 and os.path.exists(default_pdf1):
                if default_pdf1 not in pdf_paths:
                    pdf_paths.append(default_pdf1)
            if use_doc2 and os.path.exists(default_pdf2):
                if default_pdf2 not in pdf_paths:
                    pdf_paths.append(default_pdf2)

            if not pdf_paths:
                st.error(_("没有要加载的PDF文档", "No PDF to load"))
            else:
                with st.spinner(_("正在解析文档并构建知识库（含表格提取）...",
                                  "Parsing documents and building index (with table extraction)...")):
                    try:
                        system = st.session_state.rag_system
                        if system is None:
                            system = RAGSystem(llm_config=st.session_state.llm_config)

                        # 逐个或批量加载
                        all_ok = True
                        total_pages = 0
                        total_chunks = 0
                        total_tables = 0
                        loaded_files = []

                        for pdf_path in pdf_paths:
                            result = system.load_pdf(pdf_path)
                            if result.get("status") == "error":
                                st.error(_(f"加载失败: {result.get('error')}",
                                           f"Load failed: {result.get('error')}"))
                                all_ok = False
                            else:
                                total_pages += result.get("pages", 0)
                                total_chunks += result.get("total_chunks", result.get("chunks", 0))
                                total_tables += result.get("tables_found", 0)
                                loaded_files.append(result.get("file", os.path.basename(pdf_path)))

                        if all_ok and loaded_files:
                            st.session_state.rag_system = system
                            st.session_state.loaded_pdfs = loaded_files

                            # 构建成功消息
                            status_msg = _(
                                f"加载完成：{len(loaded_files)}个文档，"
                                f"共{total_pages}页，{total_chunks}个片段，{total_tables}个表格",
                                f"Loaded: {len(loaded_files)} docs, "
                                f"{total_pages} pages, {total_chunks} chunks, {total_tables} tables",
                            )
                            st.success(status_msg)

                            # 显示每个文件的详情
                            for f in loaded_files:
                                st.caption(f"✅ {f}")
                    except Exception as e:
                        st.error(_(f"加载过程出错: {str(e)}",
                                   f"Error during loading: {str(e)}"))

        if clear_btn:
            import shutil
            for p in [config.VECTOR_STORE_CONFIG["索引路径"],
                      config.VECTOR_STORE_CONFIG["元数据路径"]]:
                if os.path.exists(p):
                    os.remove(p)
            # 也清除BM25缓存
            bm25_path = config.VECTOR_STORE_CONFIG["元数据路径"].replace(".json", "_bm25.pkl")
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
            st.session_state.rag_system = None
            st.session_state.loaded_pdfs = []
            st.info(_("缓存已清空，所有索引已删除", "Cache cleared, all indexes removed"))

        st.divider()

        # LLM配置
        st.markdown(f"### {_('2. LLM配置', '2. LLM Config')}")
        config_changed = False

        llm_cfg = st.session_state.llm_config
        provider = st.selectbox(
            _("提供商", "Provider"),
            options=["deepseek", "openai", "ollama", "custom"],
            index=["deepseek", "openai", "ollama", "custom"].index(llm_cfg.get("提供商", "deepseek")),
        )
        if provider != llm_cfg.get("提供商"):
            defaults = {
                "deepseek": ("deepseek-v4-flash", "https://api.deepseek.com/v1"),
                "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
                "ollama": ("qwen2.5:7b", "http://localhost:11434/v1"),
                "custom": (llm_cfg.get("模型", ""), llm_cfg.get("API地址", "")),
            }
            model, api_base = defaults.get(provider, ("", ""))
            llm_cfg["提供商"] = provider
            llm_cfg["模型"] = model
            llm_cfg["API地址"] = api_base
            config_changed = True

        model = st.text_input(_("模型", "Model"), value=llm_cfg.get("模型", ""))
        api_base = st.text_input(_("API地址", "API Base URL"), value=llm_cfg.get("API地址", ""))
        api_key = st.text_input(
            _("API密钥", "API Key"),
            value=llm_cfg.get("API密钥", ""),
            type="password",
            help=_("已硬编码在config.py中，此处可覆盖", "Hardcoded in config.py, can override here"),
        )

        if model != llm_cfg.get("模型") or api_base != llm_cfg.get("API地址") or api_key != llm_cfg.get("API密钥"):
            llm_cfg["模型"] = model
            llm_cfg["API地址"] = api_base
            llm_cfg["API密钥"] = api_key
            config_changed = True

        if config_changed:
            st.session_state.rag_system = None
            st.success(_("✅ LLM配置已更新，请重新加载文档", "✅ Config updated, please reload documents"))

        st.divider()

        # 模式切换
        st.markdown(f"### {_('3. 问答模式', '3. QA Mode')}")
        st.session_state.comparison_mode = st.toggle(
            _("对比模式（RAG vs 纯LLM）", "Comparison Mode (RAG vs Pure LLM)"),
            value=st.session_state.comparison_mode,
            help=_("同时展示RAG检索结果和纯LLM回答的对比", "Show both RAG and pure LLM answers side by side"),
        )

        st.divider()

        # 快速测试问题
        st.markdown(f"### {_('4. 快速测试', '4. Quick Test')}")

        # 按文档来源分组
        from evaluation.test_questions import (
            TEST_QUESTIONS_XINGTU, TEST_QUESTIONS_LIYUAN, EN_TEST_QUESTIONS
        )

        if st.session_state.ui_lang == "en":
            test_qs = EN_TEST_QUESTIONS
        else:
            # 显示兴图新科问题（前5个）
            with st.expander(_("兴图新科（招股说明书1）", "Xingtu Xinke (Doc1)"), expanded=False):
                for q in TEST_QUESTIONS_XINGTU[:5]:
                    if st.button(f"[{q['id']}] {q['question'][:30]}...", key=f"qt_xt_{q['id']}"):
                        st.session_state.pending_question = q['question']
            # 显示力源信息问题（新增4个）
            with st.expander(_("力源信息（招股说明书2）", "Liyuan Info (Doc2)"), expanded=True):
                for q in TEST_QUESTIONS_LIYUAN:
                    if st.button(f"[{q['id']}] {q['question'][:30]}...", key=f"qt_ly_{q['id']}"):
                        st.session_state.pending_question = q['question']

        st.divider()

        # 显示已加载文档状态
        if st.session_state.loaded_pdfs:
            st.caption(_("已加载文档:", "Loaded documents:"))
            for f in st.session_state.loaded_pdfs:
                st.caption(f"📄 {f}")

        st.markdown("---")
        st.caption(_("PDF智能问答系统 v4.0（图像解析+表格优化版）",
                     "PDF Q&A System v4.0 (Image+Table-Optimized)"))
        st.caption(_("工单编号: 人工智能NLP-RAG-图像内容解析及检索优化",
                     "WO#: AI-NLP-RAG-Image-Content-Parsing"))

        # 图像处理统计（如有）
        rag = st.session_state.get("rag_system")
        if rag and hasattr(rag, "image_stats") and rag.image_stats["total_images"] > 0:
            stats = rag.image_stats
            st.caption(_(
                f"已提取: {stats['total_images']} 个图像 | 类型: {', '.join(f'{k}({v})' for k, v in stats['by_type'].items())}",
                f"Images: {stats['total_images']} | Types: {', '.join(f'{k}({v})' for k, v in stats['by_type'].items())}"
            ))


def render_chat_interface():
    """渲染聊天界面"""
    lang = st.session_state.ui_lang

    if lang == "zh":
        st.markdown("## 提问区")
    else:
        st.markdown("## Ask a Question")

    # 显示历史
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                expander_label = _("详细信息", "Details")
                with st.expander(expander_label):
                    st.json(msg["metadata"])

    pending_q = st.session_state.pop("pending_question", None)

    if lang == "zh":
        input_placeholder = "请输入您的问题（例如：公司注册资本是多少？）"
    else:
        input_placeholder = "Ask a question (e.g., What is the registered capital?)"

    inject_voice_button()

    question = st.chat_input(input_placeholder)

    if pending_q:
        question = pending_q

    if question and st.session_state.rag_system:
        # 显示用户问题
        with st.chat_message("user"):
            st.markdown(question)

        system = st.session_state.rag_system

        if st.session_state.comparison_mode:
            # 对比模式
            with st.chat_message("assistant"):
                if lang == "zh":
                    spinner_text = "正在检索并生成RAG回答（含表格匹配）..."
                else:
                    spinner_text = "Retrieving and generating RAG answer (with table matching)..."
                with st.spinner(spinner_text):
                    result = system.compare(question)

                # 检查错误
                rag_error = result.get("rag_error")
                llm_error = result.get("llm_error")

                tab1_label = _("RAG回答", "RAG Answer")
                tab2_label = _("纯LLM回答", "Pure LLM Answer")
                tab1, tab2 = st.tabs([tab1_label, tab2_label])

                with tab1:
                    if rag_error:
                        st.error(_(f"生成失败: {rag_error}", f"Generation failed: {rag_error}"))
                    else:
                        st.markdown(result["rag"]["answer"] or _("（无回答）", "(No answer)"))
                    src_info = []
                    if result["rag"]["source_pages"]:
                        src_info.append(_(f"来源页码: {result['rag']['source_pages']}",
                                          f"Source pages: {result['rag']['source_pages']}"))
                    if result["rag"].get("source_files"):
                        src_info.append(_(f"文档来源: {', '.join(result['rag']['source_files'])}",
                                          f"From: {', '.join(result['rag']['source_files'])}"))
                    src_info.append(_(f"耗时: {result['rag']['elapsed']}",
                                      f"Time: {result['rag']['elapsed']}"))
                    st.caption(" | ".join(src_info))

                with tab2:
                    if llm_error:
                        st.error(_(f"生成失败: {llm_error}", f"Generation failed: {llm_error}"))
                    else:
                        st.markdown(result["pure_llm"]["answer"] or _("（无回答）", "(No answer)"))
                    st.caption(_(f"耗时: {result['pure_llm']['elapsed']}",
                                 f"Time: {result['pure_llm']['elapsed']}"))

                st.session_state.history.append({
                    "role": "assistant",
                    "content": _(f"**RAG回答**: {result['rag']['answer'][:100]}...\n\n**纯LLM回答**: {result['pure_llm']['answer'][:100]}...",
                                 f"**RAG**: {result['rag']['answer'][:100]}...\n\n**Pure LLM**: {result['pure_llm']['answer'][:100]}..."),
                    "metadata": result,
                })
        else:
            # 普通模式
            with st.chat_message("assistant"):
                if lang == "zh":
                    spinner_text = "正在检索文档并生成答案（含表格数据匹配）..."
                else:
                    spinner_text = "Searching documents and generating answer (with table matching)..."
                with st.spinner(spinner_text):
                    result = system.ask(question)

                if result.get("error"):
                    st.error(_(f"生成出错: {result['error']}", f"Error: {result['error']}"))
                else:
                    st.markdown(result["rag_answer"] or _("（未能生成回答）", "(Failed to generate answer)"))

                    # 来源信息（含文档来源）
                    src_info = []
                    if result["source_pages"]:
                        src_info.append(_(f"来源: 第 {', '.join(str(p) for p in result['source_pages'])} 页",
                                          f"Source: Page {', '.join(str(p) for p in result['source_pages'])}"))
                    if result.get("source_files"):
                        src_info.append(_(f"文档: {', '.join(result['source_files'])}",
                                          f"From: {', '.join(result['source_files'])}"))
                    src_info.append(_(f"耗时: {result['elapsed']}",
                                      f"Time: {result['elapsed']}"))
                    if src_info:
                        st.caption(" | ".join(src_info))

                    # ---- 图像预览（工单4新增） ----
                    if result.get("image_results"):
                        with st.expander(_("相关图像预览", "Related Image Previews"), expanded=False):
                            for img_res in result["image_results"]:
                                img_path = img_res.get("image_path", "")
                                if img_path and os.path.exists(img_path):
                                    img_type = img_res.get("image_type", "图像")
                                    caption = img_res.get("caption_text", "")
                                    score = img_res.get("score", 0)
                                    caption_text = f"{img_type}"
                                    if caption:
                                        caption_text += f" | {caption}"
                                    caption_text += f" (相似度: {score:.3f})"
                                    st.image(img_path, caption=caption_text, use_container_width=True)
                    # ---- 图像预览结束 ----

                    # 检索详情
                    expander_label = _("查看检索详情", "View retrieval details")
                    with st.expander(expander_label):
                        qa = result.get("query_analysis", {})
                        lang_text = _("中文", "Chinese") if qa.get("lang", "zh") == "zh" else _("英文", "English")
                        table_type = qa.get("table_type", "")
                        company = qa.get("company", "")

                        detail_lines = [
                            _("**Query分析**", "**Query Analysis**"),
                            _(f"- 意图: {qa.get('intent', '?')}", f"- Intent: {qa.get('intent', '?')}"),
                            _(f"- 语言: {lang_text}", f"- Language: {lang_text}"),
                        ]
                        if company:
                            detail_lines.append(_(f"- 公司: {company}", f"- Company: {company}"))
                        if table_type:
                            detail_lines.append(_(f"- 表格类型: {table_type}", f"- Table type: {table_type}"))
                        if qa.get("entities"):
                            detail_lines.append(_(f"- 实体: {qa['entities']}", f"- Entities: {qa['entities']}"))

                        detail_lines.append(_(f"**检索结果**: 找到 {result.get('total_chunks_found', 0)} 个相关片段",
                                              f"**Results**: {result.get('total_chunks_found', 0)} relevant chunks"))

                        # 显示检索到的chunk
                        for i, chunk in enumerate(result.get("source_chunks", [])[:5]):
                            chunk_type = chunk.get("type", "unknown")
                            type_icon = "📊" if chunk_type in ("table", "table_numeric") else "📝"
                            src_file = chunk.get("source_file", "")
                            src_str = f" ({src_file})" if src_file else ""

                            detail_lines.append(
                                _(f"\n**{type_icon}片段 {i+1}** (相关性: {chunk.get('final_score', chunk.get('score', 0)):.3f}, 第{chunk.get('page','?')}页{src_str})",
                                  f"\n**{type_icon}Chunk {i+1}** (Score: {chunk.get('final_score', chunk.get('score', 0)):.3f}, Page {chunk.get('page','?')}{src_str})")
                            )
                            text = chunk.get("text", "")
                            detail_lines.append(text[:200] + ("..." if len(text) > 200 else ""))

                        st.markdown("\n".join(detail_lines))

                st.session_state.history.append({
                    "role": "assistant",
                    "content": result.get("rag_answer", ""),
                    "metadata": {
                        "source_pages": result.get("source_pages", []),
                        "source_files": result.get("source_files", []),
                        "elapsed": result.get("elapsed", ""),
                        "query_analysis": result.get("query_analysis", {}),
                        "error": result.get("error"),
                    },
                })

    elif question:
        st.warning(_("请先在左侧加载PDF文档", "Please load PDF documents first"))


def render_welcome():
    """渲染欢迎页面"""
    lang = st.session_state.ui_lang

    if lang == "zh":
        st.markdown("""
### 欢迎使用PDF智能问答系统（表格解析优化版 v3.0）

本系统基于 **RAG（检索增强生成）** 技术，针对工单3进行**表格解析及检索优化**：

**优化点：**
1. **📊 增强表格解析** — 表格标题检测、列类型识别、跨页合并
2. **🎯 表格感知检索** — 列头匹配、表格类型权重提升、数值查询优化
3. **📑 多文档支持** — 同时加载多份招股说明书，智能路由
4. **📈 查询扩展增强** — 金融术语→表格类型映射（如"发行股数"→"股本结构表"）
5. **🔬 Markdown表格格式** — 结构化表格输出，LLM更好理解
6. **📋 数值摘要chunk** — 数值表格数据独立chunk，便于快速匹配

**使用步骤：**
1. 在左侧勾选或上传PDF文档
2. 点击 **加载文档**（自动提取文本+表格）
3. 在输入框中输入或点击 🎤 语音输入
4. 可选：开启 **对比模式**（RAG vs 纯LLM）

**支持两份招股说明书：**
- 招股说明书1.pdf — 武汉兴图新科电子股份有限公司
- 招股说明书2.pdf — 武汉力源信息技术股份有限公司

**支持语言：** 中文 / English（右上角切换）
        """)
    else:
        st.markdown("""
### Welcome to PDF Q&A System (Table-Optimized v3.0)

Built on **RAG (Retrieval-Augmented Generation)** with **table parsing and retrieval optimization**:

**Optimizations:**
1. **📊 Enhanced Table Parsing** — Title detection, column type recognition, cross-page merge
2. **🎯 Table-Aware Retrieval** — Column header matching, table type weighting, numeric query boost
3. **📑 Multi-Document Support** — Load multiple IPO prospectuses simultaneously
4. **📈 Query Expansion** — Financial terms → table type mapping
5. **🔬 Markdown Table Format** — Structured table output for better LLM comprehension
6. **📋 Numeric Summary Chunks** — Independent numeric data chunks for fast matching

**How to use:**
1. Select or upload PDF documents from the left sidebar
2. Click **Load Docs** (auto-extract text + tables)
3. Type your question or click 🎤 Voice Input
4. Optional: Enable **Comparison Mode** (RAG vs Pure LLM)

**Supported documents:**
- Doc1: Wuhan Xingtu Xinke Electronics Co., Ltd.
- Doc2: Wuhan Liyuan Information Technology Co., Ltd.

**Languages:** English / 中文 (switch on top right)
        """)


def main():
    """主入口"""
    st.set_page_config(
        page_title=config.UI_CONFIG["页面标题"] + " v3.0",
        page_icon=config.UI_CONFIG["页面图标"],
        layout=config.UI_CONFIG["布局"],
    )

    init_session_state()
    render_sidebar()

    # 标题行
    lang = st.session_state.ui_lang
    if lang == "zh":
        title = f"{config.UI_CONFIG['页面图标']} {config.UI_CONFIG['页面标题']} v3.0"
    else:
        title = f"{config.UI_CONFIG['页面图标']} PDF Q&A System v3.0 (Table-Optimized)"
    st.title(title)

    if st.session_state.rag_system:
        render_chat_interface()
    else:
        render_welcome()

    # 显示系统版本信息
    st.caption(_("工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化 | v3.0",
                 "WO#: AI-NLP-RAG-PDF-Table-Optimization | v3.0"))


if __name__ == "__main__":
    main()
