"""
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
PDF图像提取与分析模块
从PDF中提取图像，分析图像类型，提取上下文描述
"""

import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 数据结构
# ============================================================
@dataclass
class ExtractedImage:
    """PDF中提取的图像信息"""
    image_path: str               # 保存的文件路径
    page_num: int                 # 所在页码
    source_file: str              # 来源PDF文件名
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # 页内位置
    width: int = 0                # 图像宽度
    height: int = 0               # 图像高度
    image_type: str = "未知"       # 类型: 组织结构图/柱状图/折线图/饼图/表格截图/示意图/装饰
    caption_text: str = ""        # 图标题/题注文字
    before_text: str = ""         # 图像前的文本（上下文）
    after_text: str = ""          # 图像后的文本（上下文）
    surrounding_text: str = ""    # 完整上下文描述
    has_caption: bool = False     # 是否检测到题注
    image_hash: str = ""          # 内容哈希（去重用）
    description: str = ""         # 生成的文本描述

    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "page_num": self.page_num,
            "source_file": self.source_file,
            "width": self.width,
            "height": self.height,
            "image_type": self.image_type,
            "caption_text": self.caption_text,
            "description": self.description,
            "surrounding_text": self.surrounding_text[:500],  # 截断
            "has_caption": self.has_caption,
        }


# ============================================================
# 图像类型检测规则
# ============================================================
IMAGE_TYPE_PATTERNS = {
    "组织结构图": [
        r"组织(结构|架构)?(图|示意)?",
        r"组织架构",
        r"(销售|管理|生产|研发)部(门)?(结构|组织)?",
        r"部门(构成|设置|结构)",
        r"公司(治理|组织|管理)(结构|架构)",
    ],
    "流程图": [
        r"流程(图|示意)?",
        r"业务(流程|流)?",
        r"工作(流程|流)?",
        r"生产(流程|工艺)?",
        r"申报(流程|程序)?",
    ],
    "柱状图": [
        r"(柱状|条形)(图|示意)?",
        r"(增长|变化|对比|统计)(图|表)?",
        r"(数据|分析|统计)(图|图表)?",
        r"(年度|各年|历年|各期)(.*)(情况|对比|分布)",
    ],
    "折线图": [
        r"(折线|曲线|趋势)(图|表)?",
        r"(变化|走势|趋势|增长)(曲线|图|趋势)?",
        r"(同比|环比)(增长|变化)(图|趋势)?",
    ],
    "饼图": [
        r"(饼|扇形|占比)(图|表)?",
        r"(构成|组成|占比|比例|份额)(图|表)?",
        r"(收入|成本|费用|市场)(结构|构成)(图|表)?",
    ],
    "表格截图": [
        r"(表格|表\\d+|汇总表|统计表)",
        r"(以下|如上|下图|上图)(示|所)",
    ],
    "示意图": [
        r"(示意|原理|结构|框架)(图|模型)?",
        r"(系统|技术|产品)(架构|结构)(图)?",
        r"(关系|关联|网络)(图|示意)?",
    ],
}

# 图标题/题注检测模式
CAPTION_PATTERNS = [
    # 常见图标题格式: "图 X.X 标题" 或 "图X: 标题" 或 "Figure X. Title"
    r"(?:图|Figure|FIG)[\s.]?\d+(?:[\.\-]\d+)?[\s:：]*(.+?)(?:\n|$)",
    # "图表 X 标题"
    r"(?:图表|图示|附图)[\s.]?\d*[\s:：]*(.+?)(?:\n|$)",
    # "如上图所示" / "如下图所示"
    r"(?:如上|如下|下列|以下|由上|由下)(?:图|表|图示)(?:所示|示|可见|看出|知)",
    # 数字标号: "1. 标题" 在图像之前
    r"(?:\d+)[.、](.+?)(?:\n|$)",
]


class ImageExtractor:
    """
    PDF图像提取器
    从PDF页中提取图像，分析图像类型，提取上下文文本描述。
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.output_dir = self.config.get(
            "输出目录",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "extracted_images")
        )
        self.min_size = self.config.get("提取最小尺寸", (100, 100))
        self.max_images = self.config.get("最大图像数", 200)
        self.context_window = self.config.get("上下文窗口", 500)
        self.image_quality = self.config.get("图像质量", 95)

    # ============================================================
    # 主入口
    # ============================================================
    def extract_from_pdf(self, pdf_path: str) -> List[ExtractedImage]:
        """
        从PDF中提取所有图像。

        Args:
            pdf_path: PDF文件路径

        Returns:
            List[ExtractedImage]: 提取的图像列表
        """
        if not os.path.exists(pdf_path):
            print(f"PDF不存在: {pdf_path}")
            return []

        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        doc_output_dir = os.path.join(self.output_dir, source_name)
        os.makedirs(doc_output_dir, exist_ok=True)

        try:
            import pymupdf
        except ImportError:
            print("请安装 pymupdf: pip install pymupdf")
            return []

        doc = pymupdf.open(pdf_path)
        extracted_images = []
        seen_hashes = set()

        for page_num in range(len(doc)):
            if len(extracted_images) >= self.max_images:
                break

            page = doc[page_num]
            page_text = page.get_text()

            # --- Step 1: 提取页面中的图像 ---
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                if len(extracted_images) >= self.max_images:
                    break

                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                if not image_bytes:
                    continue

                # 计算哈希去重
                img_hash = hashlib.md5(image_bytes).hexdigest()
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                # 获取图像尺寸
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # 过滤装饰性小图像
                if width < self.min_size[0] or height < self.min_size[1]:
                    continue

                # 保存图像
                ext = base_image.get("ext", "png")
                img_filename = f"p{page_num+1:03d}_img{img_idx:02d}.{ext}"
                img_path = os.path.join(doc_output_dir, img_filename)

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # --- Step 2: 提取上下文文本 ---
                caption_text, before_text, after_text = self._extract_context(
                    page_text, img_info, page
                )

                # --- Step 3: 检测图像类型 ---
                image_type = self._detect_image_type(caption_text, before_text, page_text)

                # --- Step 4: 构建描述 ---
                surrounding_text = self._build_description(
                    caption_text, before_text, after_text, image_type
                )

                extracted_image = ExtractedImage(
                    image_path=img_path,
                    page_num=page_num + 1,
                    source_file=os.path.basename(pdf_path),
                    bbox=(img_info[1], img_info[2], img_info[3], img_info[4]) if len(img_info) > 4 else (0, 0, 0, 0),
                    width=width,
                    height=height,
                    image_type=image_type,
                    caption_text=caption_text,
                    before_text=before_text[:self.context_window],
                    after_text=after_text[:self.context_window],
                    surrounding_text=surrounding_text[:self.context_window],
                    has_caption=bool(caption_text),
                    image_hash=img_hash,
                    description=surrounding_text[:self.context_window],
                )

                extracted_images.append(extracted_image)

            # --- Step 5: 如果没有嵌入图像，检查页面是否包含图表文本（有图无嵌入式图像时仍可检索）---
            # 对于纯矢量图表，pymupdf可能没有提取到嵌入式图像
            # 此时查文本中的图表引用信息

        doc.close()
        print(f"从 {os.path.basename(pdf_path)} 提取了 {len(extracted_images)} 个图像")
        return extracted_images

    # ============================================================
    # 上下文提取
    # ============================================================
    def _extract_context(self, page_text: str, img_info: tuple,
                         page) -> Tuple[str, str, str]:
        """
        提取图像周围的上下文文本。
        策略: 
        1. 先找图标题/题注
        2. 提取图像前200字符和后400字符的文本
        3. 如果页面文本较少（图像占主导），使用全页文本
        """
        # 尝试从文本中找图标题
        caption_text = self._find_caption(page_text)

        # 分段落
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        full_text = "\n".join(paragraphs)

        # 策略：提取整个页面文本中可能和该图像相关的前后段落
        before_text = ""
        after_text = ""

        # 找图像在页面的位置信息
        # 如果有位置信息(bbox)，用位置附近文本
        # 否则用全页文本
        if len(img_info) > 4:
            y0 = img_info[2]  # 图像顶部y坐标
            y1 = img_info[4]  # 图像底部y坐标
        else:
            y0, y1 = 0, 0

        # 在所有包含图表/组织相关关键词的段落前后找
        # 先尝试找图表关键词段落
        chart_para_indices = []
        for i, para in enumerate(paragraphs):
            # 扩展关键词列表
            keywords = ["图", "表", "示", "图例", "曲线", "柱状", "饼",
                       "组织", "架构", "部门", "销售", "流程",
                       "2008年", "增长率", "市场", "应用结构"]
            if any(kw in para for kw in keywords):
                chart_para_indices.append(i)

        if chart_para_indices:
            # 取第一个匹配段落的前后文
            first = chart_para_indices[0]
            before_start = max(0, first - 2)
            after_end = min(len(paragraphs), first + 3)
            before_text = "\n".join(paragraphs[before_start:first])
            after_text = "\n".join(paragraphs[first+1:after_end])

            if not caption_text:
                caption_text = paragraphs[first][:300]
        else:
            # 没有图表关键词，取前几段
            before_text = "\n".join(paragraphs[:2])
            after_text = "\n".join(paragraphs[-2:])

        # 如果页面文本总量不大（图像占主导），使用全页上下文
        if len(full_text) < 3000:
            before_text = full_text[:1500]
            after_text = full_text[-1500:] if len(full_text) > 1500 else ""
            if not caption_text:
                # 找页面标题
                for para in paragraphs[:3]:
                    if len(para) > 10 and len(para) < 100:
                        caption_text = para
                        break

        return caption_text, before_text, after_text

    def _find_caption(self, text: str) -> str:
        """从文本中检测图标题/题注"""
        for pattern in CAPTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                full_match = match.group(0)
                # 提取标题文字部分
                caption = match.group(1) if match.lastindex and match.lastindex >= 1 else full_match
                if caption and len(caption) < 100:
                    return caption.strip()
        return ""

    # ============================================================
    # 图像类型检测
    # ============================================================
    def _detect_image_type(self, caption: str, before_text: str,
                           page_text: str) -> str:
        """根据上下文推断图像类型"""
        combined_text = f"{caption} {before_text} {page_text[:500]}"

        for img_type, patterns in IMAGE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    return img_type

        # 检查是否包含图表通用关键词
        chart_keywords = ["增长率", "增长", "趋势", "对比", "分布"]
        if any(kw in combined_text for kw in chart_keywords):
            return "数据图表"

        return "示意图"

    # ============================================================
    # 图像描述构建
    # ============================================================
    def _build_description(self, caption: str, before: str,
                           after: str, image_type: str) -> str:
        """构建图像的文字描述"""
        parts = []

        if image_type:
            parts.append(f"【图像类型】{image_type}")

        if caption:
            parts.append(f"【图标题】{caption}")

        if before:
            parts.append(f"【上文】{before[:200]}")

        if after:
            parts.append(f"【下文】{after[:200]}")

        return "\n".join(parts)

    # ============================================================
    # 提取PDF页面的图表引用信息（即使没有嵌入式图像）
    # ============================================================
    def extract_chart_references(self, pdf_path: str) -> List[Dict]:
        """
        提取PDF中的图表引用信息。
        有些PDF的图表是矢量图，没有嵌入式图像，但文本中会引用。
        """
        try:
            import pymupdf
        except ImportError:
            return []

        doc = pymupdf.open(pdf_path)
        references = []

        for page_num in range(len(doc)):
            text = doc[page_num].get_text()

            # 检测图标题引用
            caption_matches = re.finditer(
                r"(?:如图|如上图|如下图所示|见下图|图\d+[\s.])",
                text
            )
            for match in caption_matches:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                if context:
                    references.append({
                        "page": page_num + 1,
                        "context": context,
                        "reference": match.group(),
                    })

        doc.close()
        return references


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extractor = ImageExtractor()
        images = extractor.extract_from_pdf(sys.argv[1])
        print(f"\n提取到 {len(images)} 个图像:")
        for img in images[:10]:
            print(f"  第{img.page_num}页 | {img.image_type} | {img.caption_text[:50] if img.caption_text else '无题注'}")
    else:
        print("用法: python image_extractor.py <pdf_path>")
