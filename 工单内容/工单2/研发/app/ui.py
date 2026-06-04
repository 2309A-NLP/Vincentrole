"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
Streamlit用户界面 - 支持中英文切换、容错提示
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


TEST_QUESTIONS = [
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

EN_TEST_QUESTIONS = [
    {"id": 260, "question": "What was the military revenue of Wuhan Xingtu Xinke Electronics during the reporting period?"},
    {"id": 95, "question": "What technology standard did Wuhan Xingtu Xinke Electronics participate in formulating?"},
    {"id": 543, "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics?"},
    {"id": 531, "question": "Who is the legal representative of Wuhan Xingtu Xinke Electronics?"},
]


def _(text_zh: str, text_en: str) -> str:
    """多语言文本：根据用户语言设置返回对应文字"""
    lang = st.session_state.get("ui_lang", "zh")
    return text_zh if lang == "zh" else text_en


def init_session_state():
    """初始化会话状态（含默认配置从config.py读取）"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = False
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
    """在聊天输入框内嵌入麦克风按钮（🎤 在发送按钮左侧，紧贴输入框底部）"""
    lang = st.session_state.get("ui_lang", "zh")
    speech_lang = "zh-CN" if lang == "zh" else "en-US"
    btn_title = "语音输入" if lang == "zh" else "Voice input"
    alert_msg = "浏览器不支持语音识别" if lang == "zh" else "Speech not supported"

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
        '</style><script>'
        '(function(){'
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

        # PDF加载
        st.markdown(f"### {_('1. 上传PDF文档', '1. Upload PDF')}")
        uploaded_file = st.file_uploader(
            _("选择PDF文件", "Choose a PDF file"),
            type=["pdf"],
            help=_("支持上传PDF格式的文档（如招股说明书、年报等）", "Supports PDF documents (prospectus, annual reports, etc.)"),
        )

        default_pdf = os.path.join(config.DATA_DIR, "招股说明书1.pdf")
        use_default = st.checkbox(
            _("使用默认测试文档（招股说明书1.pdf）", "Use default test doc (招股说明书1.pdf)"),
            value=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button(_("加载文档", "Load Document"), type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button(_("清空缓存", "Clear Cache"), use_container_width=True)

        if load_btn:
            pdf_path = None
            if uploaded_file:
                os.makedirs(config.DATA_DIR, exist_ok=True)
                pdf_path = os.path.join(config.DATA_DIR, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            elif use_default and os.path.exists(default_pdf):
                pdf_path = default_pdf
            else:
                st.error(_("请上传PDF或启用默认测试文档", "Please upload a PDF or enable the default test document"))

            if pdf_path:
                with st.spinner(_("正在解析文档并构建知识库...", "Parsing document and building index...")):
                    try:
                        llm_cfg = st.session_state.llm_config
                        system = RAGSystem(llm_config=llm_cfg)
                        result = system.load_pdf(pdf_path)

                        if result.get("status") == "error":
                            st.error(_(
                                f"加载失败: {result.get('error', '未知错误')}",
                                f"Load failed: {result.get('error', 'Unknown error')}",
                            ))
                        else:
                            st.session_state.rag_system = system
                            st.success(_(
                                f"加载完成：{result.get('pages', 0)}页，{result.get('chunks', 0)}个片段"
                                + (f"（耗时{result['elapsed']}）" if result.get("status") != "cached" else "（缓存）"),
                                f"Loaded: {result.get('pages', 0)} pages, {result.get('chunks', 0)} chunks"
                                + (f" ({result['elapsed']})" if result.get("status") != "cached" else " (cached)"),
                            ))
                    except Exception as e:
                        st.error(_(
                            f"加载过程出错: {str(e)}",
                            f"Error during loading: {str(e)}",
                        ))

        if clear_btn:
            import shutil
            for p in [config.VECTOR_STORE_CONFIG["索引路径"],
                      config.VECTOR_STORE_CONFIG["元数据路径"]]:
                if os.path.exists(p):
                    os.remove(p)
            st.info(_("缓存已清空", "Cache cleared"))

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
            # 自动填入默认配置
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
            st.success(_("✅ LLM配置已更新，请重新点击「加载文档」", "✅ Config updated, please reload the document"))

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
        test_qs = EN_TEST_QUESTIONS if st.session_state.ui_lang == "en" else TEST_QUESTIONS
        for q in test_qs[:5]:
            if st.button(f"[{q['id']}] {q['question'][:35]}...", key=f"qt_{q['id']}"):
                st.session_state.pending_question = q['question']

        st.divider()

        st.markdown("---")
        st.caption(_("PDF智能问答系统 v2.0（优化版）", "PDF Q&A System v2.0 (Optimized)"))
        st.caption(_("工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化", "WO#: AI-NLP-RAG-PDF-QA-Optimization"))


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

        # 自动检测语言
        q_lang = detect_language(question)

        if st.session_state.comparison_mode:
            # 对比模式
            with st.chat_message("assistant"):
                if lang == "zh":
                    spinner_text = "正在检索并生成RAG回答..."
                else:
                    spinner_text = "Retrieving and generating RAG answer..."
                with st.spinner(spinner_text):
                    result = system.compare(question)

                # 检查是否是演示模式
                is_demo = not system.generator.api_key
                if is_demo:
                    demo_msg = _("💡 当前运行在演示模式（未配置 API Key），回答为预设模拟答案。", "💡 Demo mode (no API Key configured). Showing preset demo answers.")
                    st.info(demo_msg)

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
                    if result["rag"]["source_pages"]:
                        st.caption(_(f"来源页码: {result['rag']['source_pages']} | 耗时: {result['rag']['elapsed']}",
                                     f"Source pages: {result['rag']['source_pages']} | Time: {result['rag']['elapsed']}"))

                with tab2:
                    if llm_error:
                        st.error(_(f"生成失败: {llm_error}", f"Generation failed: {llm_error}"))
                    else:
                        st.markdown(result["pure_llm"]["answer"] or _("（无回答）", "(No answer)"))
                    st.caption(_(f"耗时: {result['pure_llm']['elapsed']}", f"Time: {result['pure_llm']['elapsed']}"))

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
                    spinner_text = "正在检索文档并生成答案..."
                else:
                    spinner_text = "Searching document and generating answer..."
                with st.spinner(spinner_text):
                    result = system.ask(question)

                if result.get("error"):
                    st.error(_(f"生成出错: {result['error']}", f"Error: {result['error']}"))
                else:
                    st.markdown(result["rag_answer"] or _("（未能生成回答）", "(Failed to generate answer)"))

                    if result["source_pages"]:
                        st.caption(_(f"来源: 第 {', '.join(str(p) for p in result['source_pages'])} 页 | 耗时: {result['elapsed']}",
                                     f"Source: Page {', '.join(str(p) for p in result['source_pages'])} | Time: {result['elapsed']}"))

                    expander_label = _("查看检索详情", "View retrieval details")
                    with st.expander(expander_label):
                        qa = result.get("query_analysis", {})
                        lang_text = _("中文", "Chinese") if qa.get("lang", "zh") == "zh" else _("英文", "English")
                        st.markdown(_(f"**Query分析**: 意图={qa.get('intent','?')}, 语言={lang_text}, 实体={qa.get('entities', [])}",
                                      f"**Query**: Intent={qa.get('intent','?')}, Language={lang_text}, Entities={qa.get('entities', [])}"))
                        st.markdown(_(f"**检索结果**: 找到 {result.get('total_chunks_found', 0)} 个相关片段",
                                      f"**Results**: {result.get('total_chunks_found', 0)} relevant chunks found"))
                        for i, chunk in enumerate(result.get("source_chunks", [])[:3]):
                            st.markdown(_(f"---\n**片段 {i+1}** (相关性: {chunk.get('final_score', chunk.get('score', 0)):.3f}, 第{chunk.get('page','?')}页)",
                                          f"---\n**Chunk {i+1}** (Score: {chunk.get('final_score', chunk.get('score', 0)):.3f}, Page {chunk.get('page','?')})"))
                            st.markdown(chunk.get("text", "")[:300] + "...")

                st.session_state.history.append({
                    "role": "assistant",
                    "content": result.get("rag_answer", ""),
                    "metadata": {
                        "source_pages": result.get("source_pages", []),
                        "elapsed": result.get("elapsed", ""),
                        "error": result.get("error"),
                    },
                })

    elif question:
        st.warning(_("请先在左侧加载PDF文档", "Please load a PDF document first"))


def render_welcome():
    """渲染欢迎页面（中英文）"""
    lang = st.session_state.ui_lang

    if lang == "zh":
        st.markdown("""
### 欢迎使用PDF智能问答系统（优化版 v2.0）

本系统基于 **RAG（检索增强生成）** 技术，经过以下优化：

**优化点：**
1. **结构化PDF解析** — 提取标题层级、表格、段落结构
2. **语义分块** — 基于段落边界保留语义完整性
3. **混合检索** — FAISS向量 + BM25关键词 + RRF融合
4. **重排序** — 关键词匹配度 + 标题奖励 + 表格加权
5. **查询扩展** — 同义词扩展提升召回率
6. **多语言支持** — 中英文自动检测
7. **语音输入** — 支持浏览器语音识别（Web Speech API）

**使用步骤：**
1. 在左侧加载PDF文档
2. 可选择开启 **对比模式**（RAG vs 纯LLM）
3. 在输入框中输入或点击 **🎤 语音输入**
4. 也可直接点击左侧 **快速测试** 问题

**支持语言：** 中文 / English（右上角切换）
        """)
    else:
        st.markdown("""
### Welcome to PDF Q&A System (Optimized v2.0)

Built on **RAG (Retrieval-Augmented Generation)** with optimizations:

**Optimizations:**
1. **Structured PDF Parsing** — Headings, tables, paragraphs extraction
2. **Semantic Chunking** — Preserve semantic boundaries
3. **Hybrid Retrieval** — FAISS vector + BM25 keyword + RRF fusion
4. **Re-ranking** — Keyword overlap + heading bonus + table weighting
5. **Query Expansion** — Synonym expansion for better recall
6. **Multi-language** — Auto-detect Chinese / English
7. **Voice Input** — Browser Web Speech API support

**How to use:**
1. Load a PDF document from the left sidebar
2. Optionally enable **Comparison Mode** (RAG vs Pure LLM)
3. Type your question or click **🎤 Voice Input**
4. Or use **Quick Test** buttons on the left

**Supported languages:** English / 中文 (switch on top right)

**Supported languages:** English / 中文 (switch on top right)
        """)


def main():
    """主入口"""
    st.set_page_config(
        page_title=config.UI_CONFIG["页面标题"] + " v2.0",
        page_icon=config.UI_CONFIG["页面图标"],
        layout=config.UI_CONFIG["布局"],
    )

    init_session_state()
    render_sidebar()

    # 标题行
    lang = st.session_state.ui_lang
    if lang == "zh":
        title = f"{config.UI_CONFIG['页面图标']} {config.UI_CONFIG['页面标题']} v2.0（优化版）"
    else:
        title = f"{config.UI_CONFIG['页面图标']} PDF Q&A System v2.0 (Optimized)"
    st.title(title)

    if not st.session_state.rag_system:
        render_welcome()

    render_chat_interface()

    # 显示系统版本信息
    st.caption(_("工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化 | v2.0",
                 "WO#: AI-NLP-RAG-PDF-QA-Optimization | v2.0"))


if __name__ == "__main__":
    main()
