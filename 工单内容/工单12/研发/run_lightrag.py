#!/usr/bin/env python3
"""
工单13 - LightRAG知识图谱构建 + 查询
v2: 使用Dashscope qwen-plus + BGE本地embedding
"""

import sys
import os
import json
import asyncio

# 添加LightRAG路径
sys.path.insert(0, "/Users/suwente/.hermes/hermes-agent/venv/lib/python3.13/site-packages")

# 强制离线加载HuggingFace模型（防止网络超时）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ssl
import certifi
import numpy as np
import pymupdf
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# SSL context for aiohttp (certifi certificates)
_ssl_ctx = ssl.create_default_context(cafile=certifi.where())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from config import (
    LIGHTRAG_WORKING_DIR, PDF_FILES, TEST_QUESTIONS,
    API_KEY, API_URL, LLM_MODEL, BGE_MODEL_PATH
)

# ===================== LLM函数 (Dashscope qwen-plus) =====================

async def dashscope_llm(prompt: str, system_prompt: str = None,
                        history_messages: list = None, **kwargs) -> str:
    """异步Dashscope API调用（带429重试，退避最长60s）"""
    import aiohttp

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.3),
        "max_tokens": kwargs.get("max_tokens", 2048),
    }

    max_retries = kwargs.get("max_retries", 10)
    last_error = ""
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_URL}/chat/completions",
                    headers=headers, json=data,
                    timeout=aiohttp.ClientTimeout(total=120),
                    ssl=_ssl_ctx
                ) as resp:
                    status = resp.status
                    result = await resp.json()

                    if status == 200 and "choices" in result:
                        return result["choices"][0]["message"]["content"]

                    if status == 429:
                        wait = min(2 ** attempt, 60)
                        print(f"  429限流, 等待{wait}s后重试({attempt+1}/{max_retries})...")
                        await asyncio.sleep(wait)
                        continue

                    err_msg = result.get("error", {}).get("message", str(result)[:200])
                    last_error = f"HTTP {status}: {err_msg}"

        except asyncio.TimeoutError:
            last_error = "请求超时"
        except aiohttp.ClientError as e:
            last_error = f"网络错误: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries - 1:
            wait = min(2 ** attempt, 30)
            print(f"  {last_error}, {wait}s后重试({attempt+1}/{max_retries})...")
            await asyncio.sleep(wait)

    print(f"  LLM API最终失败: {last_error}")
    return ""

# ===================== Embedding函数 (BGE本地) =====================

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"加载BGE模型: {BGE_MODEL_PATH}")
        _model = SentenceTransformer(BGE_MODEL_PATH, device="cpu")
        print("BGE模型加载完成")
    return _model

async def bge_embed_func(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(emb, dtype=np.float32)

# ===================== 主函数 =====================

async def main():
    print("=" * 60)
    print("LightRAG v2 - Dashscope qwen-plus + BGE embedding")
    print("=" * 60)

    # === 每次启动强制清空output目录，确保全新构建 ===
    import shutil
    if os.path.exists(LIGHTRAG_WORKING_DIR):
        for fname in os.listdir(LIGHTRAG_WORKING_DIR):
            fpath = os.path.join(LIGHTRAG_WORKING_DIR, fname)
            if fname != ".gitkeep":
                if os.path.isfile(fpath):
                    os.remove(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
        print("已清空output目录（确保全新构建）")
    else:
        os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)

    # 初始化LightRAG
    rag = LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=dashscope_llm,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=768, func=bge_embed_func),
        chunk_token_size=600,
        chunk_overlap_token_size=50,
        top_k=40,
        chunk_top_k=20,
        cosine_threshold=0.2,
        addon_params={"language": "Chinese"},
    )

    print("\n初始化存储...")
    await rag.initialize_storages()

    print("\n开始构建知识图谱...")
    for pdf_path in PDF_FILES:
        fname = os.path.basename(pdf_path)
        print(f"\n  处理: {fname}")
        doc = pymupdf.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        print(f"    文本长度: {len(text)} 字符")

        await rag.ainsert(text, file_paths=[pdf_path])
        print(f"    {fname} 插入完成")

    print("\n知识图谱构建完成!")

    # 查询测试
    print("\n" + "=" * 60)
    print("开始查询测试 (mode=mix)")
    print("=" * 60)

    results = []
    for q in TEST_QUESTIONS:
        print(f"\n[ID={q['id']}] {q['question'][:50]}...")
        try:
            answer = await rag.aquery(
                q["question"],
                param=QueryParam(mode="mix", enable_rerank=False)
            )
            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": answer,
                "status": "ok"
            })
            print(f"  -> {answer[:100]}...")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"  查询失败: {err}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": f"错误: {err}",
                "status": "error"
            })

    await rag.finalize_storages()

    # 保存结果
    os.makedirs(os.path.join(BASE_DIR, "comparison"), exist_ok=True)
    output_path = os.path.join(BASE_DIR, "comparison", "lightrag_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")

    # 统计
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n查询统计: {ok_count}/{len(results)} 成功")

if __name__ == "__main__":
    asyncio.run(main())
