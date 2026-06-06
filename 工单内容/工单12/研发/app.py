"""
工单13 - LightRAG优化任务
Streamlit交互式对比界面 - 传统RAG vs LightRAG
v2: 使用LightRAG原生查询 + Dashscope qwen-plus
"""

import os, json, time, sys, asyncio, traceback
from pathlib import Path

# 添加LightRAG路径
sys.path.insert(0, "/Users/suwente/.hermes/hermes-agent/venv/lib/python3.13/site-packages")

# 强制离线加载HuggingFace模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ssl
import certifi
import streamlit as st
import numpy as np

# SSL context for aiohttp
_ssl_ctx = ssl.create_default_context(cafile=certifi.where())

BASE_DIR = Path(__file__).resolve().parent

# ===================== 配置 =====================
DATA_DIR = "/Users/suwente/Desktop/专高六学习资料/RAG 工单/附件"
PDF_FILES = [
    os.path.join(DATA_DIR, "招股说明书1.pdf"),
    os.path.join(DATA_DIR, "招股说明书2.pdf"),
]
LIGHTRAG_DIR = os.path.join(BASE_DIR, "output")
API_KEY="sk-17342ca3d15941f7a4866c6ca4a2de15"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"

BGE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5"
    "/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
)

# 16个测试问题
CACHED_QUESTIONS = {
    5: "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
    6: "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？",
    1: "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    2: "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    3: "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
    4: "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
    260: "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    95: "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    33: "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    34: "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    957: "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    793: "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    795: "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
    543: "武汉兴图新科电子股份有限公司注册资本是多少？",
    531: "武汉兴图新科电子股份有限公司法定代表人是谁？",
    207: "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
}


# ===================== LLM函数（带429重试） =====================

def call_llm(prompt: str, system_prompt: str = None, max_retries: int = 5) -> str:
    """调用Dashscope API，429时指数退避重试"""
    import requests
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{API_URL}/chat/completions",
                headers=headers, json=data, timeout=120,
            )
            result = resp.json()
            if resp.status_code == 200 and "choices" in result:
                return result["choices"][0]["message"]["content"]
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                last_err = f"HTTP 429 限流，等待{wait}s后重试..."
                time.sleep(wait)
                continue
            last_err = f"HTTP {resp.status_code}: {result.get('error', {}).get('message', str(result)[:200])}"
        except Exception as e:
            last_err = str(e)
        if attempt < max_retries:
            time.sleep(1)
    return f"[API错误] {last_err}"


# ===================== 传统RAG =====================

def read_pdf_text():
    import pymupdf
    texts = {}
    for pdf_path in PDF_FILES:
        name = os.path.basename(pdf_path)
        doc = pymupdf.open(pdf_path)
        texts[name] = "\n".join(page.get_text() for page in doc)
        doc.close()
    return texts

@st.cache_data
def load_pdf_fulltext():
    return read_pdf_text()

def trad_rag_query(question: str) -> dict:
    start = time.time()
    pdf_texts = load_pdf_fulltext()
    full_text = "\n\n---\n\n".join(
        f"【{name}】\n{text[:300000]}"
        for name, text in pdf_texts.items()
    )
    system_prompt = (
        "你是一个专业的金融文档问答助手。基于以下招股说明书内容回答用户问题。"
        "请给出准确、具体的答案，并注明信息来源。如果文档中没有相关信息，请明确说明。"
    )
    prompt = f"请基于以下文档内容回答用户问题：\n\n{full_text[:80000]}\n\n---\n\n问题：{question}"
    answer = call_llm(prompt, system_prompt)
    elapsed = f"{time.time() - start:.1f}s"
    return {"answer": answer, "elapsed": elapsed, "source": "全文检索 + qwen-plus API"}


# ===================== LightRAG（原生查询） =====================

@st.cache_resource
def init_lightrag():
    """初始化LightRAG引擎（单例）"""
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer
    import aiohttp

    # 加载BGE模型
    encoder = SentenceTransformer(BGE_PATH, device="cpu")

    async def bge_embed(texts):
        return encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # LLM函数
    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        data = {"model": LLM_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 2048}

        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{API_URL}/chat/completions",
                        headers=headers, json=data,
                        timeout=aiohttp.ClientTimeout(total=120),
                        ssl=_ssl_ctx
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if "choices" in result:
                                return result["choices"][0]["message"]["content"]
                        if resp.status == 429:
                            await asyncio.sleep(min(2 ** attempt, 30))
                            continue
            except Exception:
                if attempt < 4:
                    await asyncio.sleep(min(2 ** attempt, 10))
        return ""

    rag = LightRAG(
        working_dir=LIGHTRAG_DIR,
        llm_model_func=llm_func,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=768, func=bge_embed),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        top_k=40,
        chunk_top_k=20,
        cosine_threshold=0.2,
        addon_params={"language": "Chinese"},
    )

    # 初始化存储（同步方式）
    loop = asyncio.new_event_loop()
    loop.run_until_complete(rag.initialize_storages())

    return rag, loop


def lightrag_query(question: str) -> dict:
    """LightRAG原生查询：mix模式（local + global）"""
    start = time.time()

    graph_file = os.path.join(LIGHTRAG_DIR, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graph_file):
        return {"answer": "LightRAG知识图谱未构建，请先运行 run_lightrag.py", "elapsed": "0s", "source": "LightRAG"}

    try:
        from lightrag import QueryParam
        rag, loop = init_lightrag()

        # 使用LightRAG原生mix模式查询
        answer = loop.run_until_complete(
            rag.aquery(question, param=QueryParam(mode="mix"))
        )

        elapsed = f"{time.time() - start:.1f}s"
        return {
            "answer": answer,
            "elapsed": elapsed,
            "source": "LightRAG (mix模式: local+global + qwen-plus)"
        }

    except Exception as e:
        elapsed = f"{time.time() - start:.1f}s"
        return {
            "answer": f"LightRAG查询失败: {type(e).__name__}: {str(e)}",
            "elapsed": elapsed,
            "source": "LightRAG"
        }


# ===================== 界面 =====================

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = True
    if "lightrag_available" not in st.session_state:
        st.session_state.lightrag_available = os.path.exists(
            os.path.join(LIGHTRAG_DIR, "graph_chunk_entity_relation.graphml")
        )

def render_sidebar():
    with st.sidebar:
        st.markdown("## 配置")
        st.markdown("### 引擎状态")
        light_ok = st.session_state.lightrag_available
        st.caption(f"传统RAG: ✅ 全文 + qwen-plus")
        st.caption(f"LightRAG: {'✅ 知识图谱已构建' if light_ok else '❌ 知识图谱未构建'}")
        st.caption(f"LLM: {LLM_MODEL}")
        st.divider()
        st.markdown("### 问答模式")
        st.session_state.comparison_mode = st.toggle(
            "对比模式（传统RAG vs LightRAG）",
            value=st.session_state.comparison_mode,
        )
        st.divider()
        st.markdown("### 快速测试")
        with st.expander("力源信息（招股说明书2）", expanded=True):
            for qid in [5, 6, 1, 2, 3, 4]:
                if st.button(f"[{qid}] {CACHED_QUESTIONS[qid][:25]}...", key=f"qt_ly_{qid}"):
                    st.session_state.pending_question = CACHED_QUESTIONS[qid]
        with st.expander("兴图新科（招股说明书1）", expanded=False):
            for qid in [260, 95, 33, 34, 957, 793, 795, 543, 531, 207]:
                if st.button(f"[{qid}] {CACHED_QUESTIONS[qid][:25]}...", key=f"qt_xt_{qid}"):
                    st.session_state.pending_question = CACHED_QUESTIONS[qid]
        st.divider()
        if st.session_state.history:
            st.caption(f"当前会话: {len(st.session_state.history)} 个问答")
        st.markdown("---")
        st.caption("工单13 v2.0 | LightRAG原生mix查询 | qwen-plus")


def render_chat_interface():
    st.markdown("## 提问区")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                with st.expander("对比详情"):
                    md = msg["metadata"]
                    if md.get("trad"):
                        st.markdown("**传统RAG**")
                        st.markdown(md["trad"]["answer"])
                        st.caption(f"耗时: {md['trad']['elapsed']} | 来源: {md['trad']['source']}")
                    if md.get("light"):
                        st.markdown("**LightRAG**")
                        st.markdown(md["light"]["answer"])
                        st.caption(f"耗时: {md['light']['elapsed']} | 来源: {md['light']['source']}")

    pending_q = st.session_state.pop("pending_question", None)
    question = st.chat_input("请输入您的问题（例如：公司注册资本是多少？）")
    if pending_q:
        question = pending_q

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if st.session_state.comparison_mode:
                with st.spinner("正在查询传统RAG..."):
                    trad_result = trad_rag_query(question)
                with st.spinner("正在查询LightRAG（mix模式）..."):
                    light_result = lightrag_query(question)

                tab1, tab2 = st.tabs(["传统RAG", "LightRAG"])
                with tab1:
                    st.markdown(trad_result["answer"])
                    st.caption(f"耗时: {trad_result['elapsed']} | {trad_result['source']}")
                with tab2:
                    st.markdown(light_result["answer"])
                    st.caption(f"耗时: {light_result['elapsed']} | {light_result['source']}")

                st.session_state.history.append({
                    "role": "assistant",
                    "content": f"**传统RAG**: {trad_result['answer'][:150]}...\n\n**LightRAG**: {light_result['answer'][:150]}...",
                    "metadata": {"trad": trad_result, "light": light_result},
                })
            else:
                with st.spinner("正在查询传统RAG..."):
                    result = trad_rag_query(question)
                st.markdown(result["answer"])
                st.caption(f"耗时: {result['elapsed']} | {result['source']}")
                st.session_state.history.append({
                    "role": "assistant", "content": result["answer"],
                    "metadata": {"trad": result},
                })

        st.rerun()


def main():
    st.set_page_config(page_title="传统RAG vs LightRAG 对比", layout="wide")
    init_session_state()
    render_sidebar()

    st.title("传统RAG vs LightRAG 实时对比")
    st.caption("基于招股说明书PDF的知识问答 - 传统全文检索 vs LightRAG知识图谱mix模式")
    render_chat_interface()
    st.markdown("---")
    st.caption("工单13 v2.0 | 传统RAG: 全文+qwen-plus | LightRAG: 知识图谱mix模式+qwen-plus")


if __name__ == "__main__":
    main()
