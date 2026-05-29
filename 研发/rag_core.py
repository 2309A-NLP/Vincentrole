# -*- coding: utf-8 -*-
"""
RAG 核心对话服务（ChatService）

功能概述：
- 封装完整的 RAG（检索增强生成）对话流程
- 对接向量数据库（Milvus）与重排序模型
- 对接大语言模型（Ollama / OpenAI 兼容接口）
- 管理短期对话记忆（Redis）
- 提供角色化、可控、稳定的 AI 回复
"""

import logging
import re
import asyncio
import time
from functools import partial
from typing import Callable, Any

from openai import OpenAI
from config import settings
from storage import milvus_db, redis_mem
from embeddings import embed_engine
from hybrid_retrieval import build_query_variants, keyword_index, merge_retrieval_results


# ============================================================
# Python < 3.9 兼容 asyncio.to_thread
# ============================================================
if not hasattr(asyncio, "to_thread"):
    async def to_thread(func: Callable, *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    asyncio.to_thread = to_thread


logger = logging.getLogger(__name__)


class ChatService:
    """
    ChatService 是整个 AI 对话系统的“大脑”

    职责：
    1. 接收用户输入
    2. 执行 RAG 检索与重排序
    3. 构造 Prompt
    4. 调用 LLM 生成回复
    5. 保存对话历史
    """

    def __init__(self):
        """
        初始化 ChatService

        - 创建 OpenAI 兼容的客户端
        """
        self.client = OpenAI(
            base_url=settings.LLM_API_BASE,
            # 原 Ollama 本地模型配置：
            # api_key="EMPTY"
            api_key=settings.LLM_API_KEY,
        )
        logger.info(
            "Chat service initialized: llm_base=%s model=%s",
            settings.LLM_API_BASE,
            settings.LLM_MODEL_NAME
        )

    @staticmethod
    def _build_response_rules(char_id: str) -> str:
        """
        根据角色构造统一的输出规则
        """
        base_rules = [
            "请直接输出自然、完整的中文回复。",
            "不要只输出关键词、标签、标题或类似“饮食调理”“作息调整”的短语。",
            "默认写成一段连贯的话，至少使用 2 句完整句子，并把建议和原因说清楚。",
            "除非用户明确要求分点，否则不要只列提纲。",
        ]

        if char_id == "doctor_tcm":
            base_rules.append(
                "如果给出调理建议，请结合症状解释原因，并补充可执行的生活方式建议。"
            )
            base_rules.append(
                "遇到明显不适、持续加重或无法判断的情况，要提醒用户及时就医。"
            )
        else:
            base_rules.append(
                "保持轻松自然的陪伴感，但回复仍然要是完整句子，避免只说口号。"
            )

        return "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(base_rules, start=1)
        )

    @staticmethod
    def _format_knowledge(relevant_contexts):
        """
        将 RAG 检索到的知识库内容格式化为字符串
        """
        if not relevant_contexts:
            return "暂无相关知识库信息。"
        return "\n".join(
            f"{index}. {text}" for index, text in enumerate(relevant_contexts, start=1)
        )

    @staticmethod
    def _looks_like_fragment(text: str) -> bool:
        """
        判断 LLM 的回复是否像“碎片式回答”
        """
        stripped = text.strip(" \n\t-•*：:;；，,。！？!?")
        if not stripped:
            return True

        if len(stripped) <= 12 and not re.search(r"[。！？!?]", text):
            return True

        segments = [
            segment.strip()
            for segment in re.split(r"[，,、/\n]", stripped)
            if segment.strip()
        ]
        if 1 <= len(segments) <= 4 and all(len(segment) <= 6 for segment in segments):
            return True

        return False

    async def _hybrid_retrieve(self, user_input: str, user_id: str):
        """
        并行执行多路召回：
        - 多个查询变体的 Milvus 稠密召回
        - 本地 BM25 关键词召回
        - RRF 融合后交给 reranker 精排
        """
        variants = build_query_variants(user_input)

        started_at = time.perf_counter()
        # embedding 模型通常不是线程安全瓶颈友好型组件；先批量编码，再并行做外部检索。
        vectors = await asyncio.to_thread(
            embed_engine.encode, variants[:3]
        )
        logger.debug(
            "Hybrid retrieval embedding completed: variants=%s elapsed_ms=%.1f",
            len(variants[:3]),
            (time.perf_counter() - started_at) * 1000,
        )

        async def dense_search(query: str, vector):
            route_started_at = time.perf_counter()
            hits = await asyncio.to_thread(
                milvus_db.search,
                vector,
                8,
                user_id,
            )
            for hit in hits:
                # route 用于后续融合调试，不参与 prompt，避免把检索细节暴露给模型。
                hit["route"] = f"dense:{query[:24]}"
            logger.debug(
                "Dense retrieval route completed: query_length=%s hits=%s elapsed_ms=%.1f",
                len(query or ""),
                len(hits),
                (time.perf_counter() - route_started_at) * 1000,
            )
            return hits

        async def keyword_search():
            route_started_at = time.perf_counter()
            hits = await asyncio.to_thread(
                partial(
                    keyword_index.search,
                    user_input,
                    limit=12,
                    user_id=user_id
                )
            )
            logger.debug(
                "Keyword retrieval route completed: hits=%s elapsed_ms=%.1f",
                len(hits),
                (time.perf_counter() - route_started_at) * 1000,
            )
            return hits

        retrieval_tasks = [
            asyncio.create_task(dense_search(query, vector))
            for query, vector in zip(variants[:3], vectors)
        ]

        retrieval_tasks.append(asyncio.create_task(keyword_search()))

        result_groups = await asyncio.gather(*retrieval_tasks)
        # BM25 和向量检索的分数不可直接比较，先用 RRF 做候选级融合，再交给 reranker 精排。
        candidates = merge_retrieval_results(result_groups, limit=20)

        logger.debug(
            "Hybrid retrieval completed: user_id=%s variants=%s routes=%s candidates=%s elapsed_ms=%.1f",
            user_id,
            len(variants),
            len(result_groups),
            len(candidates),
            (time.perf_counter() - started_at) * 1000,
        )
        return candidates

    @staticmethod
    def _normalize_ai_reply(ai_reply: str, char_id: str) -> str:
        """
        对 LLM 的回复进行规范化处理
        """
        reply = " ".join((ai_reply or "").split())

        if not reply:
            return (
                "我先根据你提供的信息做一个初步判断，不过目前信息还不够完整，"
                "你可以再补充一下具体情况，我会尽量给出更清楚的建议。"
            )

        if ChatService._looks_like_fragment(reply):
            if char_id == "doctor_tcm":
                return (
                    f"从你目前的描述来看，可以先从{reply}这个方向入手调理。"
                    "建议你同时观察近期的作息、饮食、睡眠和情绪变化，"
                    "如果症状持续存在或明显加重，还是要尽快就医做进一步辨证。"
                )
            return (
                f"如果让我认真回应你的情况，我觉得可以先从{reply}这个方向开始。"
                "别只把它当成一个词看待，我们可以继续把你的感受、原因和下一步怎么做聊清楚。"
            )

        if reply[-1] not in "。！？!?":
            reply += "。"

        return reply

    async def chat(
        self,
        user_id: str,
        char_id: str,
        user_input: str,
        character_prompt: str,
        history_messages=None,
    ):
        """
        完整的 RAG 对话流程入口
        """
        logger.info(
            "Chat pipeline started: user_id=%s char_id=%s input_length=%s",
            user_id, char_id, len(user_input or ""),
        )

        # ---------- 1. 短期记忆 ----------
        redis_mem.add_message(user_id, char_id, "user", user_input)

        # ---------- 2. RAG 检索 ----------
        raw_contexts = await self._hybrid_retrieve(user_input, user_id=user_id)
        logger.debug(
            "RAG hybrid retrieval completed: user_id=%s char_id=%s hits=%s",
            user_id, char_id, len(raw_contexts),
        )

        rerank_started_at = time.perf_counter()
        relevant_contexts = embed_engine.rerank(user_input, raw_contexts)
        logger.debug(
            "RAG rerank completed: user_id=%s char_id=%s candidates=%s contexts=%s elapsed_ms=%.1f",
            user_id,
            char_id,
            len(raw_contexts),
            len(relevant_contexts),
            (time.perf_counter() - rerank_started_at) * 1000,
        )

        knowledge_str = self._format_knowledge(relevant_contexts)

        # ---------- 3. 历史对话 ----------
        history = (
            history_messages
            if history_messages is not None
            else redis_mem.get_history(user_id, char_id)
        )
        history_str = ""
        for msg in history[-5:]:
            history_str += f"{msg['role']}: {msg['content']}\n"

        # ---------- 4. 构造 Prompt ----------
        messages = [
            {
                "role": "system",
                "content": (
                    f"{character_prompt}\n\n"
                    "输出要求：\n"
                    f"{self._build_response_rules(char_id)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请根据以下内容回答用户问题。\n\n"
                    f"【参考信息】\n{knowledge_str}\n\n"
                    f"【历史对话】\n{history_str or '暂无历史对话。'}\n"
                    f"【用户当前输入】\n{user_input}\n\n"
                    "请直接给出完整回复，不要重复这些标题。"
                ),
            },
        ]

        # ---------- 5. 调用 LLM ----------
        try:
            logger.debug(
                "Calling LLM: model=%s user_id=%s char_id=%s",
                settings.LLM_MODEL_NAME,
                user_id,
                char_id,
            )
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                temperature=0.5,
                stream=False,
            )
            ai_reply = self._normalize_ai_reply(
                response.choices[0].message.content, char_id
            )
        except Exception as e:
            logger.exception(
                "LLM call failed: user_id=%s char_id=%s", user_id, char_id
            )
            ai_reply = f"[系统错误] 无法连接到大模型服务: {str(e)}"

        # ---------- 6. 保存 AI 回复 ----------
        redis_mem.add_message(user_id, char_id, "assistant", ai_reply)
        logger.info(
            "Chat pipeline completed: user_id=%s char_id=%s response_length=%s",
            user_id, char_id, len(ai_reply or ""),
        )

        return ai_reply


# 全局单例，便于其他模块直接引用
chat_service = ChatService()
