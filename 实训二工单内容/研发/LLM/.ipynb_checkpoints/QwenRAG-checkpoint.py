# -*- coding: utf-8 -*-
"""通义千问 qwen-plus + 本地医疗知识库 RAG，兼容 Linly-Talker 的 LLM 接口
   (generate / chat / clear_history)。"""
import os, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

BGE_DIR  = "/root/.cache/modelscope/hub/AI-ModelScope/bge-small-zh-v1___5"
EMB      = "/root/Linly-Talker/kb_emb.npy"
META     = "/root/Linly-Talker/kb_meta.json"
KEY_FILE = "/root/Linly-Talker/dashscope.key"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# bge 检索：query 端要加指令，corpus 端不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

DEFAULT_SYS = (
    "你是一位专业、严谨的医疗助理。请参考下面检索到的真实医患问答资料，"
    "用通俗、简洁的口语化中文回答用户问题，给出实用建议。"
    "资料不相关时依据常识谨慎回答并提醒及时就医。不要编造诊断，"
    "回答控制在120字以内，适合数字人语音播报。"
)


def _read_key(api_key=None):
    if api_key:
        return api_key
    if os.environ.get("DASHSCOPE_API_KEY"):
        return os.environ["DASHSCOPE_API_KEY"]
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE, encoding="utf-8").read().strip()
    raise RuntimeError("未找到 DashScope API Key（环境变量 DASHSCOPE_API_KEY 或 %s）" % KEY_FILE)


class QwenRAG:
    def __init__(self, model="qwen-plus", api_key=None, top_k=3, min_score=0.35):
        self.model = model
        self.top_k = top_k
        self.min_score = min_score      # 相似度低于此值视为知识库无相关内容
        self.history = []
        self.client = OpenAI(api_key=_read_key(api_key), base_url=BASE_URL)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(BGE_DIR)
        self.bge = AutoModel.from_pretrained(BGE_DIR).to(self.device).eval().half()
        self.emb = np.load(EMB)                                  # (N,512) 已归一化
        self.meta = json.load(open(META, encoding="utf-8"))
        print(f"[QwenRAG] 知识库载入 {len(self.meta)} 条，模型={model}")

    # ---------- 检索 ----------
    def _embed(self, text):
        enc = self.tok([QUERY_INSTRUCTION + text], padding=True, truncation=True,
                       max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.bge(**enc)
            e = out.last_hidden_state[:, 0]
            e = torch.nn.functional.normalize(e.float(), p=2, dim=1)
        return e.cpu().numpy()[0]

    def retrieve(self, query):
        q = self._embed(query)
        scores = self.emb @ q                                    # 都归一化 → 点积=余弦
        idx = np.argsort(-scores)[:self.top_k]
        return [(self.meta[i], float(scores[i])) for i in idx if scores[i] >= self.min_score]

    def _context(self, hits):
        if not hits:
            return ""
        blocks = []
        for i, (m, s) in enumerate(hits, 1):
            blocks.append(
                f"【参考{i}｜科室:{m.get('dept','')}｜相关疾病:{m.get('dis','') or '未标注'}】\n"
                f"患者问:{m['q']}\n医生答:{m['a']}"
            )
        return "\n\n".join(blocks)

    def _call(self, messages):
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=0.3, max_tokens=512)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("[QwenRAG] API错误:", e)
            return "对不起，服务暂时出错了，请稍后再试。"

    # ---------- Linly-Talker 接口 ----------
    def generate(self, question, system_prompt=""):
        hits = self.retrieve(question)
        ctx = self._context(hits)
        sys = system_prompt or DEFAULT_SYS
        user = (f"检索到的参考资料：\n{ctx}\n\n用户问题：{question}\n\n请基于参考资料用口语化中文简洁回答："
                if ctx else f"用户问题：{question}\n\n知识库无相关资料，请依据医学常识谨慎回答并提醒就医：")
        return self._call([{"role": "system", "content": sys},
                           {"role": "user", "content": user}])

    def chat(self, system_prompt, message, history):
        hits = self.retrieve(message)
        ctx = self._context(hits)
        sys = (system_prompt or DEFAULT_SYS)
        if ctx:
            sys = sys + "\n\n以下是与当前问题相关的参考资料：\n" + ctx
        messages = [{"role": "system", "content": sys}]
        for u, a in history:                                     # 携带多轮上下文
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": message})
        response = self._call(messages)
        history.append((message, response))
        self.history = history
        return response, history

    def clear_history(self):
        self.history = []
