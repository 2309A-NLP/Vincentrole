#!/usr/bin/env python3
"""
工单13 - LightRAG vs 传统RAG 对比
简化版：直接对比检索+LLM问答效果
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from config import PDF_FILES, LIGHTRAG_WORKING_DIR, KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL, TEST_QUESTIONS

# ===================== LLM函数 =====================

def kimi_llm(prompt: str, system: str = None) -> str:
    """Kimi API调用"""
    import requests
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    resp = requests.post(
        f"{KIMI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {KIMI_API_KEY}", "Content-Type": "application/json"},
        json={"model": KIMI_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 2048},
        timeout=120
    )
    return resp.json()["choices"][0]["message"]["content"]

# ===================== PDF提取 =====================

def extract_pdfs():
    """提取所有PDF文本"""
    import pymupdf
    texts = []
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            print(f"跳过: {pdf}")
            continue
        doc = pymupdf.open(pdf)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        texts.append({"file": os.path.basename(pdf), "text": text})
        print(f"已提取: {os.path.basename(pdf)} ({len(text)} 字符)")
    return texts

# ===================== LightRAG =====================

def get_lightrag_module():
    """获取LightRAG模块（从hermes虚拟环境）"""
    import sys
    venv_path = "/Users/suwente/.hermes/hermes-agent/venv/lib/python3.13/site-packages"
    if venv_path not in sys.path:
        sys.path.insert(0, venv_path)
    import lightrag
    return lightrag

async def run_lightrag(questions):
    """LightRAG构建+查询"""
    lightrag = get_lightrag_module()
    LightRAG = lightrag.LightRAG
    QueryParam = lightrag.QueryParam
    
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "=" * 50)
    print("LightRAG模式")
    print("=" * 50)
    
    # 加载embedding模型
    model_path = "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    encoder = SentenceTransformer(model_path)
    
    async def embed_func(texts):
        return encoder.encode(texts, normalize_embeddings=True)
    
    # 初始化
    rag = LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=lambda p, **kw: kimi_llm(p),
        embedding_func=EmbeddingFunc(embedding_dim=768, func=embed_func),
        addon_params={"language": "Chinese"}
    )
    await rag.initialize_storages()
    
    # 构建知识图谱（如果不存在）
    graph_file = os.path.join(LIGHTRAG_WORKING_DIR, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graph_file):
        print("构建知识图谱...")
        pdf_texts = extract_pdfs()
        for item in pdf_texts:
            print(f"  插入: {item['file']}")
            await rag.ainsert(item["text"], file_paths=[item["file"]])
        print("知识图谱构建完成!")
    else:
        print("知识图谱已存在，跳过构建")
    
    # 查询
    results = []
    for q in questions:
        print(f"\n[ID={q['id']}] {q['question'][:40]}...")
        answer = await rag.aquery(q["question"], param=QueryParam(mode="mix"))
        results.append({"id": q["id"], "question": q["question"], "answer": answer})
        print(f"  -> {answer[:80]}...")
    
    await rag.finalize_storages()
    return results

# ===================== 传统RAG（简化版） =====================

def run_traditional_rag(questions):
    """传统RAG：向量检索+LLM"""
    from sentence_transformers import SentenceTransformer
    import pymupdf
    import numpy as np
    
    print("\n" + "=" * 50)
    print("传统RAG模式")
    print("=" * 50)
    
    # 加载模型
    model_path = "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    encoder = SentenceTransformer(model_path)
    
    # 提取PDF并分块
    print("处理文档...")
    chunks = []
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            continue
        doc = pymupdf.open(pdf)
        for i, page in enumerate(doc):
            text = page.get_text()
            # 简单按段落分块
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 50:
                    chunks.append({"text": para, "source": os.path.basename(pdf), "page": i+1})
        doc.close()
    
    print(f"  共 {len(chunks)} 个文本块")
    
    # 计算embedding
    print("计算向量...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = encoder.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True)
    
    # 查询
    results = []
    for q in questions:
        print(f"\n[ID={q['id']}] {q['question'][:40]}...")
        
        # 向量检索Top5
        q_emb = encoder.encode([q["question"]], normalize_embeddings=True)[0]
        scores = np.dot(chunk_embeddings, q_emb)
        top_idx = np.argsort(scores)[-5:][::-1]
        context = "\n".join([chunks[i]["text"] for i in top_idx])
        
        # LLM生成答案
        prompt = f"""根据以下参考资料回答问题。如果资料中没有相关信息，请说明。

参考资料：
{context}

问题：{q['question']}

请简洁回答："""
        
        answer = kimi_llm(prompt)
        results.append({"id": q["id"], "question": q["question"], "answer": answer})
        print(f"  -> {answer[:80]}...")
    
    return results

# ===================== 简单评估 =====================

def simple_evaluate(trad_results, light_results):
    """简单评估：对比答案长度和关键词覆盖"""
    print("\n" + "=" * 50)
    print("对比评估")
    print("=" * 50)
    
    comparisons = []
    for trad, light in zip(trad_results, light_results):
        # 提取关键词（简单方式：取问题中的公司名和核心名词）
        q = trad["question"]
        keywords = []
        for word in ["力源信息", "兴图新科", "发行股数", "募集资金", "关联方", "注册资本", "法定代表人", "收入", "技术标准"]:
            if word in q:
                keywords.append(word)
        
        # 检查关键词覆盖
        trad_cover = sum(1 for k in keywords if k in trad["answer"])
        light_cover = sum(1 for k in keywords if k in light["answer"])
        
        comparisons.append({
            "id": trad["id"],
            "question": q[:30] + "...",
            "trad_len": len(trad["answer"]),
            "light_len": len(light["answer"]),
            "trad_kw_cover": trad_cover,
            "light_kw_cover": light_cover
        })
    
    # 统计
    trad_avg_len = sum(c["trad_len"] for c in comparisons) / len(comparisons)
    light_avg_len = sum(c["light_len"] for c in comparisons) / len(comparisons)
    trad_avg_kw = sum(c["trad_kw_cover"] for c in comparisons) / len(comparisons)
    light_avg_kw = sum(c["light_kw_cover"] for c in comparisons) / len(comparisons)
    
    print(f"\n平均回答长度: 传统RAG={trad_avg_len:.0f}, LightRAG={light_avg_len:.0f}")
    print(f"关键词覆盖: 传统RAG={trad_avg_kw:.2f}, LightRAG={light_avg_kw:.2f}")
    
    return comparisons

# ===================== 主函数 =====================

async def main():
    print("=" * 50)
    print("工单13: LightRAG vs 传统RAG 对比")
    print("=" * 50)
    
    start = time.time()
    
    # 运行两种模式
    trad_results = run_traditional_rag(TEST_QUESTIONS)
    light_results = await run_lightrag(TEST_QUESTIONS)
    
    # 评估
    comparisons = simple_evaluate(trad_results, light_results)
    
    # 保存结果
    output = {
        "traditional_rag": trad_results,
        "lightrag": light_results,
        "comparisons": comparisons,
        "elapsed": time.time() - start
    }
    
    os.makedirs(os.path.join(BASE_DIR, "comparison"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "comparison", "results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n总耗时: {output['elapsed']:.1f}秒")
    print(f"结果已保存: comparison/results.json")

if __name__ == "__main__":
    asyncio.run(main())
