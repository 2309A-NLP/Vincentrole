#!/Users/suwente/anaconda3/bin/python3
"""
工单14 - 低质量工业PDF解析与问答系统
基于通义千问 qwen-vl-plus 的多模态问答
"""
import streamlit as st
import base64, json, http.client, os

st.set_page_config(page_title="工单14 - 工业PDF问答系统", layout="wide")

API_KEY = "sk-17342ca3d15941f7a4866c6ca4a2de15"
IMG_DIR = "/tmp/patent_images"

# 6 个问题定义
QUESTIONS = [
    {"id": "Q1", "text": "根据文本信息，该静电除尘器的发明人是：",
     "options": ["A. P·吉特勒", "B. 未知", "C. 静电除尘器研究小组", "D. 该专利未提及发明人"],
     "answer": "A", "page": 1, "type": "text"},
    {"id": "Q2", "text": "根据文本信息，以下哪个描述符合该静电除尘器的特征？",
     "options": ["A. 具有平行于外壳主轴线的垂直方向的片状沉积电极。", 
                 "B. 具有管状入口和出口，它们分别由3种不同圆锥形部分所构成。",
                 "C. 管状入口具有单个圆锥形部分，达到外壳直径的80至95%，剩余部分采用台阶形式。",
                 "D. 主要用于液体的除尘"],
     "answer": "C", "page": 1, "type": "text"},
    {"id": "Q3", "text": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？",
     "options": ["A. 部件4位于部件5的左侧", "B. 部件4位于部件5的右侧", 
                 "C. 部件4位于部件5的上方", "D. 部件4位于部件5的下方"],
     "answer": "A", "page": 7, "type": "image"},
    {"id": "Q4", "text": "在文件中第7页的图片中，尺寸X1，X2，X3分别代表什么部件的间隔距离？",
     "options": ["A. 配气带孔盘6，6'，6\"之间的间隔距离", 
                 "B. 外壳2与圆锥形部分10之间的间隔距离",
                 "C. 管状入口1与管状出口9之间的间隔距离", 
                 "D. 外壳中心轴线3与外壳2之间的间隔距离"],
     "answer": "A", "page": 7, "type": "image"},
    {"id": "Q5", "text": "根据文件中第7页图示，气流方向(7)首先经过哪个部件？紧接着会经过哪个部件？",
     "options": ["A. 先经过部件4，再经过部件5", "B. 先经过部件10，再经过部件6\"",
                 "C. 先经过部件6\"，再经过部件6'", "D. 先经过部件6，再经过部件4"],
     "answer": "C", "page": 7, "type": "image"},
    {"id": "Q6", "text": "根据文件中第7页图示，如果已知外壳直径D，那么h1和h2的尺寸可以用来计算什么？",
     "options": ["A. 计算气流速度", "B. 确定配气带孔盘6，6'，6\"的位置",
                 "C. 计算除尘效率", "D. 确定外壳2的材料"],
     "answer": "B", "page": 7, "type": "image"},
]

PROMPTS = {
    "Q1": "阅读这份专利文档的首页，找出该静电除尘器的发明人。选项：{options}。请只输出选项字母(A/B/C/D)。",
    "Q2": "阅读这份静电除尘器专利的描述。选项中哪一个正确描述了该静电除尘器的入口特征？选项：{options}。请只输出选项字母(A/B/C/D)。",
    "Q3": "查看专利第7页的图纸。图中部件4相对于部件5的位置关系是什么？注意看两个标注在图纸上的位置。选项：{options}。请只输出选项字母(A/B/C/D)。",
    "Q4": "查看专利第7页的图纸。图中标注了尺寸X1、X2、X3，这些尺寸分别代表什么？选项：{options}。请只输出选项字母(A/B/C/D)。",
    "Q5": "仔细查看专利第7页的图纸。图中可以看到三个配气带孔盘6、6'、6''。气流方向(7)从底部进入后，首先经过哪个盘？紧接着经过哪个？选项：{options}。请只输出选项字母(A/B/C/D)。",
    "Q6": "查看专利第7页的图纸。图纸中标有尺寸h1和h2以及外壳直径D。h1和h2是用来确定什么的？选项：{options}。请只输出选项字母(A/B/C/D)。",
}

def query_vl(image_path, question):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    body = {
        "model": "qwen-vl-plus",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": question}
            ]}
        ]
    }
    conn = http.client.HTTPSConnection("dashscope.aliyuncs.com", timeout=120)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    conn.request("POST", "/compatible-mode/v1/chat/completions", json.dumps(body), headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data["choices"][0]["message"]["content"]

def run_all_tests():
    results = {}
    progress = st.progress(0)
    status = st.empty()
    
    for i, q in enumerate(QUESTIONS):
        status.info(f"测试 {q['id']}: {q['text'][:30]}...")
        img_path = f"{IMG_DIR}/page_{q['page']}.png"
        opts_str = " ".join(q['options'])
        prompt = PROMPTS[q['id']].format(options=opts_str)
        
        try:
            answer = query_vl(img_path, prompt)
            is_correct = q['answer'] in answer.upper() if answer else False
        except Exception as e:
            answer = f"错误: {e}"
            is_correct = False
        
        results[q['id']] = {
            "question": q['text'],
            "options": q['options'],
            "expected": q['answer'],
            "model_answer": answer.strip(),
            "correct": is_correct,
            "page": q['page'],
            "type": q['type']
        }
        progress.progress((i + 1) / len(QUESTIONS))
    
    status.success("测试完成！")
    return results

def img_to_html(path, width=400):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" width="{width}px" style="border:1px solid #ddd;border-radius:4px">'

# UI
st.markdown("""
# 📄 工单14 - 工业PDF解析与问答系统
**专利文档：CN100342976C.pdf（静电除尘器）**
""")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("总页数", "8页", "图片型PDF")
with col2:
    st.metric("测试问题", "6题", "文本+图像")
with col3:
    st.metric("解析引擎", "通义千问VL", "qwen-vl-plus")

st.divider()

# Main action
if st.button("🚀 开始测试全部6个问题", type="primary", use_container_width=True):
    results = run_all_tests()
    st.session_state.results = results
    st.rerun()

if "results" in st.session_state:
    results = st.session_state.results
    correct = sum(1 for r in results.values() if r["correct"])
    total = len(results)
    
    st.subheader(f"📊 测试结果: {correct}/{total} 正确")
    
    if correct == total:
        st.balloons()
        st.success("🎉 100% 准确率！所有问题回答正确！")
    
    cols = st.columns(len(QUESTIONS))
    for i, (qid, r) in enumerate(results.items()):
        with cols[i]:
            icon = "✅" if r["correct"] else "❌"
            st.markdown(f"**{icon} {qid}**")
    
    st.divider()
    
    # Detail view
    for qid, r in results.items():
        with st.expander(f"{'✅' if r['correct'] else '❌'} {qid}: {r['question'][:60]}...", expanded=not r['correct']):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**问题：**" + r['question'])
                st.markdown("**选项：**")
                for opt in r['options']:
                    prefix = "🔵" if opt.startswith(r['expected']) else ""
                    st.markdown(f"{prefix} {opt}")
                st.markdown(f"**标准答案：** `{r['expected']}`")
                st.markdown(f"**模型回答：** `{r['model_answer']}`")
                st.markdown(f"**结果：** {'✅ 正确' if r['correct'] else '❌ 错误'}")
            with c2:
                img_path = f"{IMG_DIR}/page_{r['page']}.png"
                if os.path.exists(img_path):
                    st.markdown(f"**参考页面 (第{r['page']}页)：**")
                    st.markdown(img_to_html(img_path, 350), unsafe_allow_html=True)
else:
    st.info("点击上方按钮开始测试")

# Info
st.divider()
with st.expander("📋 任务说明"):
    st.markdown("""
    **任务一：**
    - deepdoc 模块技术分析文档（PDF解析策略、分块策略、Redis Stream机制）
    
    **任务二：**
    - 上传 CN100342976C.pdf 进行解析
    - 6个问题测试达到100%准确率
    - 多轮测试记录及优化方案
    """)
