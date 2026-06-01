"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
Streamlit界面 - 用户交互界面
"""

import os
import sys

# 确保能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import config
from qa_engine.orchestrator import RAGSystem
from evaluation.test_questions import TEST_QUESTIONS


def init_session_state():
    """初始化Streamlit会话状态"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = False
    # LLM 配置持久化
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "提供商": config.LLM_CONFIG.get("提供商", "deepseek"),
            "模型": config.LLM_CONFIG.get("模型", "deepseek-v4-flash"),
            "API地址": config.LLM_CONFIG.get("API地址", "https://api.deepseek.com/v1"),
            "API密钥": config.LLM_CONFIG.get("API密钥", ""),
            "最大Token数": config.LLM_CONFIG.get("最大Token数", 1024),
            "温度": config.LLM_CONFIG.get("温度", 0.3),
            "超时": config.LLM_CONFIG.get("超时", 30),
        }


def inject_voice_button():
    """在聊天输入框内嵌入麦克风按钮（🎤 位于发送按钮左侧）"""
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
        'var T="语音输入",A="浏览器不支持语音识别",L="zh-CN";'
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
        'setTimeout(function(){if(btn)btn.click()},200)}'
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
        st.markdown("## 配置面板")

        # PDF上传
        st.markdown("### 1. 上传PDF文档")
        uploaded_file = st.file_uploader(
            "选择PDF文件",
            type=["pdf"],
            help="支持上传PDF格式的文档（如招股说明书、年报等）",
        )

        # 使用默认测试文档
        default_pdf = os.path.join(config.DATA_DIR, "招股说明书1.pdf")
        use_default = st.checkbox("使用默认测试文档（招股说明书1.pdf）", value=True)

        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button("加载文档", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("清空缓存", use_container_width=True)

        if load_btn:
            pdf_path = None
            if uploaded_file:
                # 保存上传的文件
                os.makedirs(config.DATA_DIR, exist_ok=True)
                pdf_path = os.path.join(config.DATA_DIR, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            elif use_default and os.path.exists(default_pdf):
                pdf_path = default_pdf
            else:
                st.error("请上传PDF或启用默认测试文档")

            if pdf_path:
                with st.spinner("正在解析PDF并构建知识库..."):
                    system = RAGSystem(llm_config=st.session_state.llm_config)
                    result = system.load_pdf(pdf_path)
                    st.session_state.rag_system = system
                    st.session_state.current_pdf = pdf_path
                    st.success(f"加载完成: {result['pages']}页, {result['chunks']}个片段")

        if clear_btn:
            import shutil
            for p in [config.VECTOR_STORE_CONFIG["索引路径"],
                      config.VECTOR_STORE_CONFIG["元数据路径"]]:
                if os.path.exists(p):
                    os.remove(p)
            st.info("缓存已清空")

        st.divider()

        # LLM配置
        st.markdown("### 2. LLM配置")
        llm_cfg = st.session_state.llm_config

        provider = st.selectbox(
            "提供商",
            ["openai", "deepseek", "ollama"],
            index=["openai", "deepseek", "ollama"].index(llm_cfg["提供商"]),
            key="llm_provider_select",
        )

        # 根据 provider 自动推荐默认值
        if provider == "deepseek":
            default_base = "https://api.deepseek.com/v1"
            default_model = "deepseek-v4-flash"
        elif provider == "ollama":
            default_base = "http://localhost:11434/v1"
            default_model = "qwen2.5:7b"
        else:
            default_base = "https://api.openai.com/v1"
            default_model = "gpt-3.5-turbo"

        api_key = st.text_input(
            "API密钥",
            value=llm_cfg.get("API密钥", ""),
            type="password",
            key="llm_api_key_input",
        )
        api_base = st.text_input(
            "API地址",
            value=llm_cfg.get("API地址", default_base),
            key="llm_api_base_input",
        )
        model_name = st.text_input(
            "模型名称",
            value=llm_cfg.get("模型", default_model),
            key="llm_model_input",
        )

        if st.button("更新LLM配置"):
            st.session_state.llm_config = {
                "提供商": provider,
                "模型": model_name,
                "API地址": api_base,
                "API密钥": api_key,
                "最大Token数": 1024,
                "温度": 0.3,
                "超时": 30,
            }
            # 重置 RAGSystem，确保新配置生效
            st.session_state.rag_system = None
            st.success("✅ LLM配置已更新，请重新点击「加载文档」")

        st.divider()

        # 模式切换
        st.markdown("### 3. 问答模式")
        st.session_state.comparison_mode = st.toggle(
            "对比模式（RAG vs 纯LLM）",
            value=st.session_state.comparison_mode,
            help="同时展示RAG检索结果和纯LLM回答的对比",
        )

        st.divider()

        # 快速测试问题
        st.markdown("### 4. 快速测试问题")
        for q in TEST_QUESTIONS[:5]:
            if st.button(f"[{q['id']}] {q['question'][:30]}...", key=f"qt_{q['id']}"):
                st.session_state.pending_question = q['question']

        st.divider()

        st.markdown("---")
        st.caption(f"PDF智能问答系统 v1.0")
        st.caption(f"工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统")


def render_chat_interface():
    """渲染聊天界面"""
    st.markdown("## 提问区")

    # 显示历史对话
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                with st.expander("详细信息"):
                    st.json(msg["metadata"])

    # 检查是否有待处理的问题
    pending_q = st.session_state.pop("pending_question", None)

    inject_voice_button()

    # 输入框
    question = st.chat_input("请输入您的问题（例如：公司注册资本是多少？）")

    if pending_q:
        question = pending_q

    if question:
        # 自动初始化 rag_system（如果尚未就绪）
        if not st.session_state.rag_system:
            default_pdf = os.path.join(config.DATA_DIR, "招股说明书1.pdf")
            if os.path.exists(default_pdf):
                with st.spinner("正在自动加载默认文档..."):
                    system = RAGSystem(llm_config=st.session_state.llm_config)
                    result = system.load_pdf(default_pdf)
                    st.session_state.rag_system = system
                    st.session_state.current_pdf = default_pdf
                    st.toast(f"已自动加载: {result['pages']}页, {result['chunks']}个片段", icon="✅")
            else:
                st.warning("请先在左侧加载PDF文档")
                st.session_state.history.append({
                    "role": "user",
                    "content": question,
                })
                return

        # 用户问题
        with st.chat_message("user"):
            st.markdown(question)

        system = st.session_state.rag_system

        if st.session_state.comparison_mode:
            # 对比模式
            with st.chat_message("assistant"):
                with st.spinner("正在检索并生成RAG回答..."):
                    result = system.compare(question)

                # 检查是否是演示模式
                is_demo = not system.generator.api_key
                if is_demo:
                    st.info("💡 当前运行在演示模式（未配置 API Key），回答为预设模拟答案。")

                # 检查错误
                rag_error = result["rag"].get("error")
                llm_error = result["pure_llm"].get("error")

                tab1, tab2 = st.tabs(["RAG回答", "纯LLM回答"])
                with tab1:
                    if rag_error:
                        st.error(f"生成失败: {rag_error}")
                    else:
                        st.markdown(result["rag"]["answer"] or "（无回答）")
                    if result["rag"]["source_pages"]:
                        st.caption(f"来源页码: {result['rag']['source_pages']} | 耗时: {result['rag']['elapsed']}")
                with tab2:
                    if llm_error:
                        st.error(f"生成失败: {llm_error}")
                    else:
                        st.markdown(result["pure_llm"]["answer"] or "（无回答）")
                    st.caption(f"耗时: {result['pure_llm']['elapsed']}")

                st.session_state.history.append({
                    "role": "assistant",
                    "content": f"**RAG回答**: {result['rag']['answer'][:100]}...\n\n**纯LLM回答**: {result['pure_llm']['answer'][:100]}...",
                    "metadata": result,
                })
        else:
            # 普通模式
            with st.chat_message("assistant"):
                with st.spinner("正在检索文档并生成答案..."):
                    result = system.ask(question)

                if result.get("error"):
                    st.error(f"生成出错: {result['error']}")
                else:
                    st.markdown(result["rag_answer"] or "（未能生成回答）")

                    if result["source_pages"]:
                        st.caption(f"来源: 第 {', '.join(str(p) for p in result['source_pages'])} 页 | 耗时: {result['elapsed']}")

                    with st.expander("查看检索详情"):
                        st.markdown(f"**Query分析**: 意图={result['query_analysis']['intent']}, 实体={result['query_analysis']['entities']}")
                        st.markdown(f"**检索结果**: 找到 {result['total_chunks_found']} 个相关片段")
                        for i, chunk in enumerate(result["source_chunks"][:3]):
                            st.markdown(f"---\n**片段 {i+1}** (相关性: {chunk['score']:.3f}, 第{chunk['page']}页)")
                            st.markdown(chunk["text"][:300] + "...")

                st.session_state.history.append({
                    "role": "assistant",
                    "content": result["rag_answer"],
                    "metadata": {
                        "source_pages": result.get("source_pages", []),
                        "elapsed": result.get("elapsed", ""),
                    },
                })


def render_welcome():
    """渲染欢迎页面"""
    st.markdown("""
    ### 欢迎使用PDF智能问答系统
    
    本系统基于**RAG（检索增强生成）**技术，能够针对PDF文档内容进行智能问答。
    
    **使用步骤：**
    1. 在左侧上传PDF文档（默认已准备招股说明书1.pdf）
    2. 点击"加载文档"，系统会自动解析并构建知识库
    3. 在输入框中输入或点击 **🎤 语音输入** 进行提问
    4. 系统会从文档中检索相关信息并生成回答
    
    **功能特点：**
    - Query理解：自动识别问题意图和关键实体
    - 语义检索：基于向量相似度的高效检索
    - LLM生成：基于检索结果生成准确答案
    - 对比分析：RAG结果 vs 纯LLM结果对比
    - **语音输入**：支持浏览器语音识别（Web Speech API）
    
    **测试问题示例：**
    - 公司注册资本是多少？
    - 报告期内军用领域收入占比是多少？
    - 公司参与制定了哪个技术标准？
    """)


def main():
    """主入口"""
    st.set_page_config(
        page_title=config.UI_CONFIG["页面标题"],
        page_icon=config.UI_CONFIG["页面图标"],
        layout=config.UI_CONFIG["布局"],
    )

    st.title(f"{config.UI_CONFIG['页面图标']} {config.UI_CONFIG['页面标题']}")

    init_session_state()
    render_sidebar()

    render_chat_interface()


if __name__ == "__main__":
    main()
