"""
工单15 - 跨模态检索优化核心代码

优化策略:
1. 查询理解: 检测"图3""第N页"等视觉引用 → 自动加载对应图片
2. 多路召回: 文本检索 + 图片增强检索 → RRF融合
3. 重排: 优先排序含图片描述的图文块
4. 提示词工程: 检测到图片上下文时，加入"请结合技术图纸描述"指令
"""
import os
import re
import base64
from openai import OpenAI


class CrossModalRetriever:
    """跨模态检索器：检测视觉引用 → 加载图片 → 多路召回 → 融合重排"""

    def __init__(self, patent_image_dir=None):
        self.patent_image_dir = patent_image_dir or "patent_images"
        self.vl_client = OpenAI(
            api_key="sk-173...de15",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def detect_visual_ref(self, question: str) -> dict:
        """
        检测问题中的视觉引用（图号、页号）
        返回: {"has_visual": bool, "page": int, "figure": str, "image_path": str}
        """
        result = {"has_visual": False, "page": None, "figure": None, "image_path": None}

        # 匹配 "第N页" 或 "第N页图M"
        page_m = re.search(r"第\s*(\d+)\s*页", question)
        fig_m = re.search(r"图\s*(\d+|[一二三四五六七八九十]+)", question)

        if page_m:
            page = int(page_m.group(1))
            result["page"] = page
            result["has_visual"] = True
            result["image_path"] = os.path.join(
                self.patent_image_dir, f"p{page}.png"
            )
        if fig_m:
            result["figure"] = fig_m.group(0).replace(" ", "")
        return result

    def encode_image(self, image_path: str) -> str:
        """图片转 base64"""
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_enhanced_query(self, original: str, visual_ref: dict) -> str:
        """
        当检测到视觉引用时，构建增强查询
        例如: "图3 编号13 部件12 位置关系" + 原始问题
        """
        if not visual_ref["has_visual"]:
            return original

        parts = [original]
        if visual_ref["figure"]:
            parts.insert(0, visual_ref["figure"])
        return " ".join(parts)

    def multi_recall_fusion(self, text_results: list, image_results: list,
                             top_k: int = 5) -> list:
        """
        多路召回融合: RRF (Reciprocal Rank Fusion)
        将文本检索结果和图片增强检索结果融合
        """
        rrf_scores = {}
        results_map = {}

        for rank, r in enumerate(text_results):
            key = hash(str(r.get("text", ""))[:200])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank + 1)
            results_map[key] = r

        for rank, r in enumerate(image_results):
            key = hash(str(r.get("text", ""))[:200])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank + 1)
            if key not in results_map:
                results_map[key] = r

        fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [results_map[k] for k, _ in fused]

    def build_prompt(self, question: str, context: str,
                     visual_ref: dict, image_path: str = None) -> list:
        """
        构建提示词: 检测到图片时用 VL 模型，否则用纯文本模型
        返回 messages 列表
        """
        if visual_ref["has_visual"] and image_path and os.path.exists(image_path):
            img_b64 = self.encode_image(image_path)
            if img_b64:
                return [
                    {"role": "system", "content": "你是一个工业专利分析专家。请结合以下技术图纸进行分析，仔细识别图中各部件的编号和位置关系。"},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": f"请结合技术图纸回答：{question}\n\n相关文本参考：\n{context}"}
                    ]}
                ]

        # 纯文本模式
        return [
            {"role": "system", "content": "你是一个工业专利分析专家。请根据提供的文本内容回答问题，选择最正确的选项。"},
            {"role": "user", "content": f"文本参考：\n{context}\n\n问题：{question}\n请直接回答选项字母。"}
        ]

    def answer(self, question: str, text_context: str = "",
               image_results: list = None, use_vl: bool = True) -> str:
        """端到端问答"""
        visual_ref = self.detect_visual_ref(question)

        # 构建消息
        messages = self.build_prompt(
            question, text_context, visual_ref,
            visual_ref.get("image_path") if use_vl else None
        )

        # 调用 API
        is_visual = visual_ref["has_visual"] and use_vl and \
                    visual_ref.get("image_path") and \
                    os.path.exists(visual_ref["image_path"])

        model = "qwen-vl-plus" if is_visual else "qwen-turbo"
        try:
            resp = self.vl_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[错误] {e}"
