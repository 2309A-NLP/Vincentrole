import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import UI_CONFIG

st.set_page_config(page_title=UI_CONFIG.get("页面标题", "金融问答系统"), layout="wide")
st.title("💰 金融问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 RAG 引擎
@st.cache_resource
def init_rag():
    from qa_engine import RAGSystem
    return RAGSystem(llm_config={})

rag = init_rag()
st.caption(f"状态: {'就绪 ✅' if rag and rag.is_ready() else '加载中...'} | 端口: 8501 | Docker容器")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入金融相关问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            result = rag.ask(prompt) if rag else {"rag_answer": "系统未就绪"}
        st.markdown(result.get("rag_answer", "无回答"))
        st.caption(f"来源: {', '.join(result.get('source_files', []))}" if result.get("source_files") else "")
    st.session_state.messages.append({"role": "assistant", "content": result.get("rag_answer", "")})
