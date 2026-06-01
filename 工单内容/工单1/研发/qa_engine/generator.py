"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
生成器模块 - 使用LLM生成答案
"""

import json
import requests
from typing import Optional

import re


def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Chinese or English.

    Returns "zh" if Chinese characters are found, "en" otherwise.
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


EN_MOCK_ANSWERS = {
    "register": "According to the prospectus, the registered capital of Wuhan XingTuXinKe Electronics Co., Ltd. is **60 million RMB**.",
    "establish": "The company was established in **2004**, with its initial business focusing on military video processing.",
    "revenue": "During the reporting period, the company's military sector revenue accounted for over **85%** of total revenue, reflecting its core position in the national defense informatization field.",
    "industry": "The company operates in the video processing and communications sector under the electronic information industry, a seller's market in national defense informatization.",
    "risk": "Key risks disclosed in the prospectus include: high customer concentration (top 5 customers account for over 60%), product upgrade risks, and adjustments to national military industry policies.",
    "downstream": "The company's downstream clients are primarily units under the Central Military Commission, military branches, and weaponry research institutes.",
    "core": "The company's core competitiveness lies in its independently developed video encoding, transmission, and processing integrated technology platform.",
    "standard": "The company has participated in formulating multiple national military video standards; details can be found on pages 200-210 of the prospectus.",
    "prospectus": "This offering plans to raise approximately **350 million RMB**, mainly for product R&D, production base construction, and working capital supplementation.",
    "competition": "Major competitors include Zhongke Haizheng, Sangfor Technologies, and others. The company maintains a high market share through its deep presence in the military market.",
}


class LLMGenerator:
    """
    LLM答案生成器。

    支持多种后端:
    - openai: OpenAI兼容API（含DeepSeek、通义千问等）
    - ollama: 本地Ollama服务
    """

    def __init__(self, provider: str = "openai", model: str = None,
                 api_base: str = None, api_key: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3,
                 timeout: int = 30, max_retries: int = 3):
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # 默认模型配置
        if not self.model:
            if provider == "ollama":
                self.model = "qwen2.5:7b"
            elif provider == "deepseek":
                self.model = "deepseek-chat"
            else:
                self.model = "gpt-3.5-turbo"

        # 默认API地址
        if not self.api_base:
            if provider == "ollama":
                self.api_base = "http://localhost:11434/v1"
            elif provider == "deepseek":
                self.api_base = "https://api.deepseek.com/v1"
            else:
                self.api_base = "https://api.openai.com/v1"

    def generate(self, question: str, context: str,
                 system_prompt: Optional[str] = None,
                 lang: Optional[str] = None) -> dict:
        """
        基于上下文生成答案 / Generate answer based on context.

        Args:
            question: 用户问题
            context: 检索到的文档上下文
            system_prompt: 可选的系统提示词
            lang: 语言 "zh" 或 "en"，不传则自动检测

        Returns:
            dict: {"answer": "...", "source_pages": [...], "error": None}
        """
        lang = lang or detect_language(question)
        if not system_prompt:
            system_prompt = self._default_system_prompt()

        # 模拟模式：未配置 API Key 时返回模拟答案
        if not self.api_key:
            return self._mock_generate(question, context, lang)

        user_prompt = self._build_prompt(question, context)

        try:
            if self.provider == "ollama":
                answer = self._call_ollama(system_prompt, user_prompt)
            else:
                answer = self._call_openai(system_prompt, user_prompt)

            return {
                "answer": answer,
                "error": None,
            }
        except Exception as e:
            return {
                "answer": "",
                "error": f"[{self.provider}] 生成答案失败: {e}",
            }

    def _mock_generate(self, question: str, context: str, lang: str = "zh") -> dict:
        """未配置API Key时的模拟答案（用于演示）"""
        q = question.lower()

        # 根据语言选择预设答案表
        if lang == "en":
            answers = EN_MOCK_ANSWERS
        else:
            answers = {
            "注册": "根据招股说明书，武汉兴图新科电子股份有限公司的注册资本为 **6,000万元** 人民币。",
            "成立": "公司成立于 **2004年**，初期业务主要聚焦军用视频处理领域。",
            "收入": "报告期内，公司军用领域主营业务收入占比达到 **85% 以上**，反映出公司在国防信息化领域的核心地位。",
            "行业": "公司所属行业为电子信息行业下的视频处理与通信领域，属于国防信息化卖方市场。",
            "风险": "招股说明书披露的主要风险包括：客户集中度较高（前五大客户占比超过60%）、产品更新换代风险以及国家军工行业政策调整风险。",
            "下游": "公司下游客户主要为中央军委各部、军兵种、武器装备研究院所等国防领域单位。",
            "核心": "公司核心竞争力在于自主研发的视频编码、传输、处理一体化技术平台。",
            "标准": "公司参与制定了多项国家军用视频标准，包括详情可参阅招股说明书第 200-210 页。",
            "招股": "本次招股拟募集资金约 **3.5亿元**，主要用于产品研发、生产基地建设及补充流动资金。",
            "竞争": "主要竞争对手包括中科海正、深信服等同行业企业，公司依托在军用市场的深耕保持较高市占率。",
        }

        for keyword, ans in answers.items():
            if keyword in q:
                return {"answer": ans, "error": None}

        # 默认模拟答案
        if lang == "en":
            default = (
                f"[Demo Mode] No LLM API Key configured. Returning mock answer.\n\n"
                f"Your question: **{question}**\n\n"
                f"Retrieved document content:\n```\n{context[:300]}...\n```\n\n"
                f"In production, configure a valid API Key on the left panel and retry."
            )
        else:
            default = (
                f"【演示模式】由于未配置 LLM API Key，系统返回模拟答案。\n\n"
                f"您的问题是：**{question}**\n\n"
                f"检索到的相关文档内容：\n```\n{context[:300]}...\n```\n\n"
                f"在正式环境中，请在左侧配置正确的 API Key 后重试。"
            )
        return {
            "answer": default,
            "error": None,
        }

    def generate_pure_llm(self, question: str,
                          lang: Optional[str] = None) -> dict:
        """
        纯LLM生成（无RAG上下文），用于对比实验。
        Pure LLM generation (no RAG context) for comparison experiments.

        Args:
            question: 用户问题
            lang: 语言 "zh" 或 "en"，不传则自动检测

        Returns:
            dict: {"answer": "...", "error": None}
        """
        lang = lang or detect_language(question)
        system_prompt = "你是一个通用AI助手。请基于你的知识回答用户的问题。如果你不确定答案，请明确说明。"

        # 模拟模式：未配置 API Key 时返回模拟答案
        if not self.api_key:
            return self._mock_generate(question, "", lang)

        user_prompt = f"请回答以下问题：\n\n{question}\n\n请基于你的知识给出准确、简洁的回答。"

        try:
            if self.provider == "ollama":
                answer = self._call_ollama(system_prompt, user_prompt)
            else:
                answer = self._call_openai(system_prompt, user_prompt)

            return {
                "answer": answer,
                "error": None,
            }
        except Exception as e:
            return {
                "answer": "",
                "error": f"[{self.provider}] 生成答案失败(纯LLM模式): {e}",
            }

    def _build_prompt(self, question: str, context: str) -> str:
        """构建包含上下文的用户提示词"""
        return f"""请基于以下文档内容回答问题。

文档内容：
{context}

问题：{question}

要求：
1. 仅基于上述文档内容回答，不要添加文档中没有的信息
2. 如果文档内容不足以回答，请明确说明"文档中未找到相关信息"
3. 回答要简洁准确
4. 如有需要，引用具体数据"""

    def _default_system_prompt(self) -> str:
        return """你是一个专业的金融文档问答助手。你的任务是：
1. 仔细阅读提供的文档片段
2. 基于文档内容给出准确的回答
3. 如果文档信息不足，如实告知
4. 回答使用中文，保持简洁专业"""

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """调用OpenAI兼容API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        resp = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """调用本地Ollama服务"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        resp = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    @classmethod
    def from_config(cls, config: dict):
        """从配置字典创建实例"""
        llm_cfg = config.get("LLM_CONFIG", config)
        return cls(
            provider=llm_cfg.get("提供商", "openai"),
            model=llm_cfg.get("模型"),
            api_base=llm_cfg.get("API地址"),
            api_key=llm_cfg.get("API密钥", ""),
            max_tokens=llm_cfg.get("最大Token数", 1024),
            temperature=llm_cfg.get("温度", 0.3),
            timeout=llm_cfg.get("超时", 30),
        )
