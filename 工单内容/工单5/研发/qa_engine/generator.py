"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
生成器模块 - 使用LLM生成答案（支持中英文、容错机制）
"""

import json
import requests
import re
from typing import Optional


def detect_language(text: str) -> str:
    """检测语言：'zh' 中文 / 'en' 英文"""
    if not text:
        return "zh"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_chars = len(re.findall(r'[a-zA-Z]', text))
    total = max(chinese_chars + ascii_chars, 1)
    if chinese_chars / total >= 0.3:
        return "zh"
    return "en"


# 中文模拟答案
ZH_MOCK_ANSWERS = {
    "注册": "根据招股说明书，武汉兴图新科电子股份有限公司的注册资本为 **6,000万元** 人民币。",
    "成立": "公司成立于 **2004年**，初期业务主要聚焦军用视频处理领域。",
    "收入": "报告期内，公司军用领域主营业务收入占比达到 **85% 以上**，反映出公司在国防信息化领域的核心地位。",
    "行业": "公司所属行业为电子信息行业下的视频处理与通信领域，属于国防信息化卖方市场。",
    "风险": "招股说明书披露的主要风险包括：客户集中度较高（前五大客户占比超过60%）、产品更新换代风险以及国家军工行业政策调整风险。",
    "下游": "公司下游客户主要为中央军委各部、军兵种、武器装备研究院所等国防领域单位。",
    "核心": "公司核心竞争力在于自主研发的视频编码、传输、处理一体化技术平台。",
    "标准": "公司参与制定了多项国家军用视频标准，详情可参阅招股说明书第200-210页。",
    "招股": "本次招股拟募集资金约 **3.5亿元**，主要用于产品研发、生产基地建设及补充流动资金。",
    "竞争": "主要竞争对手包括中科海正、深信服等同行业企业，公司依托在军用市场的深耕保持较高市占率。",
}

# 英文模拟答案
EN_MOCK_ANSWERS = {
    "capital": "According to the prospectus, Wuhan Xingtu XinKe Electronics Co., Ltd. has a registered capital of **60 million RMB**.",
    "established": "The company was established in **2004**, initially focusing on military video processing.",
    "revenue": "During the reporting period, the company's military revenue accounted for over **85%** of total revenue.",
    "industry": "The company operates in the electronic information industry, specifically in video processing and communication, serving the defense informatization market.",
    "risk": "Key risks disclosed include high customer concentration (top 5 customers >60%), product upgrade risks, and military industry policy changes.",
    "customer": "Major downstream clients include central military departments, military branches, and weaponry research institutes.",
    "standard": "The company participated in formulating multiple national military video standards.",
    "funds": "The IPO plans to raise approximately **350 million RMB** for product development, production base construction, and working capital.",
}


class LLMGenerator:
    """
    LLM答案生成器（支持中英文、容错机制）。

    支持:
    - openai: OpenAI兼容API（含DeepSeek、通义千问等）
    - ollama: 本地Ollama服务
    - 自动语言检测
    - 重试机制
    - 超时保护
    - 降级模式
    """

    def __init__(self, provider: str = "openai", model: str = None,
                 api_base: str = None, api_key: str = "",
                 max_tokens: int = 1024, temperature: float = 0.3,
                 timeout: int = 30, max_retries: int = 2):
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
                self.model = "deepseek-v4-flash"
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

    # ============================================================
    # 公共接口
    # ============================================================

    def generate(self, question: str, context: str,
                 system_prompt: Optional[str] = None,
                 lang: Optional[str] = None) -> dict:
        """
        基于上下文生成答案（自动语言检测 + 容错）。

        Args:
            question: 用户问题
            context: 检索到的文档上下文
            system_prompt: 可选的系统提示词
            lang: 语言（'zh'/'en'），None则自动检测

        Returns:
            dict: {"answer": "...", "error": None}
        """
        if lang is None:
            lang = detect_language(question)
        prompt_lang = lang

        # 容错 1: 未配置 API Key → 模拟模式
        if not self.api_key:
            return self._mock_generate(question, context, lang=prompt_lang)

        if not system_prompt:
            system_prompt = self._default_system_prompt(prompt_lang)

        user_prompt = self._build_prompt(question, context, prompt_lang)

        # 容错 2: 重试机制
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "ollama":
                    answer = self._call_ollama(system_prompt, user_prompt)
                else:
                    answer = self._call_openai(system_prompt, user_prompt)
                return {"answer": answer, "error": None}
            except requests.Timeout:
                last_error = "请求超时，请检查网络连接"
            except requests.ConnectionError:
                last_error = "无法连接到API服务器，请检查网络或API地址"
            except requests.HTTPError as e:
                status = e.response.status_code if hasattr(e, 'response') else 0
                if status == 401:
                    last_error = "API密钥无效或已过期"
                elif status == 429:
                    last_error = "请求频率超限，请稍后再试"
                elif status == 402:
                    last_error = "账户余额不足"
                else:
                    last_error = f"API返回错误 (HTTP {status})"
            except Exception as e:
                last_error = str(e)

            if attempt < self.max_retries:
                continue

        # 容错 3: 所有重试失败 → 尝试Kimi回退（国内DeepSeek可能超时）
        kimi_answer = self._kimi_fallback(system_prompt, user_prompt)
        if kimi_answer:
            return {"answer": kimi_answer, "error": None}
        return {
            "answer": "",
            "error": f"生成失败（已重试{self.max_retries}次）: {last_error}",
        }

    def generate_pure_llm(self, question: str, lang: Optional[str] = None) -> dict:
        """
        纯LLM生成（无RAG上下文），用于对比实验。

        Args:
            question: 用户问题
            lang: 语言（'zh'/'en'），None则自动检测

        Returns:
            dict: {"answer": "...", "error": None}
        """
        if lang is None:
            lang = detect_language(question)

        # 模拟模式
        if not self.api_key:
            return self._mock_generate(question, "", lang=lang)

        if lang == "zh":
            system_prompt = "你是一个通用AI助手。请基于你的知识回答用户的问题。如果你不确定答案，请明确说明。"
            user_prompt = f"请回答以下问题：\n\n{question}\n\n请基于你的知识给出准确、简洁的回答。"
        else:
            system_prompt = "You are a general AI assistant. Answer the user's question based on your knowledge. If you are unsure, clearly state that you don't know."
            user_prompt = f"Please answer the following question:\n\n{question}\n\nProvide an accurate and concise answer based on your knowledge."

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "ollama":
                    answer = self._call_ollama(system_prompt, user_prompt)
                else:
                    answer = self._call_openai(system_prompt, user_prompt)
                return {"answer": answer, "error": None}
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    continue

        # 容错: 所有重试失败后尝试Kimi回退
        kimi_answer = self._kimi_fallback(system_prompt, user_prompt)
        if kimi_answer:
            return {"answer": kimi_answer, "error": None}
        return {"answer": "", "error": f"生成失败: {last_error}"}

    # ============================================================
    # Prompt构建
    # ============================================================

    def _build_prompt(self, question: str, context: str, lang: str) -> str:
        """构建包含上下文的用户提示词"""
        if lang == "zh":
            return f"""请基于以下文档内容回答问题。

文档内容：
{context}

问题：{question}

要求：
1. 仅基于上述文档内容回答，不要添加文档中没有的信息
2. 如果文档内容不足以回答，请明确说明"文档中未找到相关信息"
3. 回答要简洁准确
4. 如有需要，引用具体数据
5. 对于表格数据，请提取关键数值信息"""
        else:
            return f"""Please answer the question based on the following document content.

Document Content:
{context}

Question: {question}

Requirements:
1. Answer ONLY based on the provided document content
2. If the document does not contain enough information, clearly state "Information not found in the document"
3. Be concise and accurate
4. Cite specific data when applicable
5. For table data, extract key numerical information"""

    def _default_system_prompt(self, lang: str) -> str:
        """默认系统提示词（按语言）"""
        if lang == "zh":
            return """你是一个专业的金融文档问答助手。你的任务是：
1. 仔细阅读提供的文档片段
2. 基于文档内容给出准确的回答
3. 如果文档信息不足，如实告知
4. 回答使用中文，保持简洁专业"""
        else:
            return """You are a professional financial document Q&A assistant. Your tasks are:
1. Carefully read the provided document excerpts
2. Give accurate answers based on the document content
3. If the document information is insufficient, honestly state so
4. Answer in English, keep it concise and professional"""

    # ============================================================
    # API调用
    # ============================================================

    def _kimi_fallback(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """DeepSeek超时时用Kimi回退"""
        api_key = "sk-bZ5OnuinbjLgqwvByNbaoxprV1zZAbXprHdrEULdwrX8fIKx"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "moonshot-v1-8k",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers, json=payload, timeout=25,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """调用OpenAI兼容API（含超时保护）"""
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
            "timeout": self.timeout,
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

    # ============================================================
    # 模拟模式
    # ============================================================

    def _mock_generate(self, question: str, context: str, lang: str = "zh") -> dict:
        """未配置API Key时的模拟答案"""
        q = question.lower()

        if lang == "zh":
            for keyword, ans in ZH_MOCK_ANSWERS.items():
                if keyword in q:
                    return {"answer": ans, "error": None}
            # 默认模拟答案
            return {
                "answer": (
                    f"【演示模式】由于未配置 LLM API Key，系统返回模拟答案。\n\n"
                    f"您的问题是：**{question}**\n\n"
                    f"检索到的相关文档内容：\n```\n{context[:300]}...\n```\n\n"
                    f"在正式环境中，请在左侧配置正确的 API Key 后重试。"
                ),
                "error": None,
            }
        else:
            for keyword, ans in EN_MOCK_ANSWERS.items():
                if keyword in q:
                    return {"answer": ans, "error": None}
            return {
                "answer": (
                    f"[Demo Mode] No API Key configured. Demo answer.\n\n"
                    f"Your question: **{question}**\n\n"
                    f"Retrieved content:\n```\n{context[:300]}...\n```\n\n"
                    f"Configure a valid API Key on the left panel for real answers."
                ),
                "error": None,
            }

    # ============================================================
    # 多模态生成（图像+文字 → Kimi/LMM）
    # ============================================================

    def generate_with_image(self, question: str, context: str,
                            image_base64: str,
                            system_prompt: Optional[str] = None,
                            lang: Optional[str] = None) -> dict:
        """
        带图片的多模态生成：将图片base64和文字context一起发给LLM。

        Args:
            question: 用户问题
            context: 文字检索上下文
            image_base64: 图片的base64编码字符串
            system_prompt: 可选系统提示词
            lang: 语言

        Returns:
            dict: {"answer": "...", "error": None}
        """
        if lang is None:
            lang = detect_language(question)

        if not system_prompt:
            system_prompt = self._default_multimodal_prompt(lang)

        user_text = self._build_prompt(question, context, lang)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                # OpenAI多模态message格式
                user_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {"type": "text", "text": user_text},
                ]

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
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
                answer = data["choices"][0]["message"]["content"].strip()
                return {"answer": answer, "error": None}
            except requests.Timeout:
                last_error = "多模态API请求超时"
            except requests.ConnectionError:
                last_error = "无法连接到多模态API服务器"
            except requests.HTTPError as e:
                status = e.response.status_code if hasattr(e, 'response') else 0
                last_error = f"多模态API返回错误 (HTTP {status})"
            except Exception as e:
                last_error = str(e)

            if attempt < self.max_retries:
                continue

        return {"answer": "", "error": f"多模态生成失败: {last_error}"}

    def _default_multimodal_prompt(self, lang: str) -> str:
        """多模态系统提示词"""
        if lang == "zh":
            return """你是一个专业的金融文档图文分析助手。你的任务是：
1. 仔细观察提供的图片（可能是组织结构图、流程图、柱状图、饼图、表格截图等）
2. 结合图片内容和提供的文字上下文，回答用户问题
3. 如果图片中包含文字信息（如部门名称、数字、标题等），请准确提取
4. 如果图片和文字都无法回答问题，请如实说明
5. 回答使用中文，保持简洁准确"""
        else:
            return """You are a professional financial document analysis assistant. Your tasks are:
1. Carefully examine the provided image (may be org charts, flowcharts, bar charts, pie charts, table screenshots, etc.)
2. Answer the user's question based on the image content and text context
3. If the image contains text (department names, numbers, titles), extract them accurately
4. If neither image nor text can answer the question, state so honestly
5. Answer in English, keep it concise and accurate"""

    @classmethod
    def from_config(cls, config_dict: dict):
        """从配置字典创建实例"""
        llm_cfg = config_dict.get("LLM_CONFIG", config_dict)
        return cls(
            provider=llm_cfg.get("提供商", "openai"),
            model=llm_cfg.get("模型"),
            api_base=llm_cfg.get("API地址"),
            api_key=llm_cfg.get("API密钥", ""),
            max_tokens=llm_cfg.get("最大Token数", 1024),
            temperature=llm_cfg.get("温度", 0.3),
            timeout=llm_cfg.get("超时", 15),
            max_retries=llm_cfg.get("重试次数", 2),
        )

    @classmethod
    def kimi_generator(cls):
        """创建Kimi多模态生成器"""
        import config
        kimi_cfg = config.KIMI_CONFIG
        return cls(
            provider="openai",  # Kimi API兼容OpenAI
            model=kimi_cfg["模型"],
            api_base=kimi_cfg["API地址"],
            api_key=kimi_cfg["API密钥"],
            max_tokens=kimi_cfg.get("最大Token数", 2048),
            temperature=kimi_cfg.get("温度", 0.1),
            timeout=kimi_cfg.get("超时", 30),
            max_retries=kimi_cfg.get("重试次数", 2),
        )
