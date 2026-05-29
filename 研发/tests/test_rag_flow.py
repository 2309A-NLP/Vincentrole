# tests/test_rag_flow.py
import pytest
import json
from datasets import Dataset

from config import settings
from embeddings import embed_engine
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 数据
# ============================================================
def build_dataset():
    return Dataset.from_list([
        {
            "question": "熬夜后口干心烦怎么办？",
            "contexts": [
                "熬夜会耗伤阴液，容易出现口干、心烦、易怒等症状。",
                "调理建议包括早睡、清淡饮食、避免辛辣刺激。"
            ],
            "answer": "从中医角度看，熬夜伤阴，建议早睡、多喝水、饮食清淡。",
        }
    ])


# ============================================================
# 2. LLM
# ============================================================
def get_llm():
    return ChatOpenAI(
        base_url=settings.LLM_API_BASE,
        # 原 Ollama 本地模型配置：
        # api_key="EMPTY",
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL_NAME,
        temperature=0,
    )


# ============================================================
# 3. Faithfulness（自己实现）
# ============================================================
@pytest.mark.asyncio
async def test_faithfulness_manual():
    dataset = build_dataset()
    llm = get_llm()

    row = dataset[0]

    prompt = f"""
请根据给定的上下文判断回答是否忠实（faithful）。

上下文：
{chr(10).join(row['contexts'])}

回答：
{row['answer']}

请只返回 JSON：
{{"faithful": true / false}}
"""

    response = await llm.ainvoke(prompt)
    result = json.loads(response.content)

    print("\n✅ Faithfulness:", result)
    assert result.get("faithful") is True


# ============================================================
# 4. Answer Relevancy（自己实现）
# ============================================================
@pytest.mark.asyncio
async def test_answer_relevancy_manual():
    dataset = build_dataset()
    llm = get_llm()

    row = dataset[0]

    prompt = f"""
请判断回答是否与问题相关。

问题：
{row['question']}

回答：
{row['answer']}

请只返回 JSON：
{{"relevant": true / false}}
"""

    response = await llm.ainvoke(prompt)
    result = json.loads(response.content)

    print("\n✅ Answer Relevancy:", result)
    assert result.get("relevant") is True
