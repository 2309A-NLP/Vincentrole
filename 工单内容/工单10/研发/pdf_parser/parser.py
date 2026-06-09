"""
工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化
PDF解析模块 - 增强版表格解析（标题检测+列类型+跨页合并）
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from .image_extractor import ImageExtractor


class StructuredPage:
    """结构化页面，包含段落和表格"""
    def __init__(self, page_num: int, text: str = ""):
        self.page_num = page_num
        self.text = text
        self.paragraphs: List[Dict] = []
        self.tables: List[Dict] = []
        self.headings: List[Dict] = []
        self.source_file: str = ""  # 多文档来源
        self.images: List[Dict] = []  # 工单4新增：页面中的图像列表

    def to_dict(self) -> Dict:
        return {
            "page": self.page_num,
            "text": self.text,
            "paragraphs": self.paragraphs,
            "tables": self.tables,
            "headings": self.headings,
            "images": self.images,
            "source_file": self.source_file,
        }


class PDFParser:
    """PDF文档解析器 - 增强版表格解析（表格标题检测+列类型识别+跨页合并）"""

    # 招股说明书常见表格标题模式
    TABLE_TITLE_PATTERNS = [
        r'(?:本次)?发行(?:情况|概况)(?:表)?',
        r'股本(?:结构)?(?:变化)?(?:表)?',
        r'募集资金(?:投资)?(?:项目|用途)(?:一览)?(?:表)?',
        r'资金(?:运用|使用)(?:计划|方案)?(?:表)?',
        r'(?:存在|不存在)?控制关系的关联方',
        r'关联方(?:关系|情况|名称|企业)?(?:表)?',
        r'(?:主要)?(?:财务)?(?:数据|指标)(?:汇总)?(?:表)?',
        r'(?:合并)?(?:资产|负债|利润|现金流).+表',
        r'(?:主营)?(?:业务)?收入(?:构成|结构|情况)(?:表)?',
        r'主营业务(?:毛利|成本)(?:表)?',
        r'期间(?:费用|损益)(?:表)?',
        r'(?:研发|管理|销售)费用.+表',
        r'前十大?客户(?:情况)?(?:表)?',
        r'前十大?供应商(?:情况)?(?:表)?',
        r'董事[、, ]监事[、, ]高级管理人员',
        r'本次发行(?:前后)?(?:股权)?结构',
    ]

    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf
        self._engine = None

    @property
    def engine(self):
        if self._engine is not None:
            return self._engine
        if self.use_pymupdf:
            try:
                import pymupdf
                self._engine = "pymupdf"
                return "pymupdf"
            except ImportError:
                pass
        try:
            from pypdf import PdfReader
            self._engine = "pypdf"
            return "pypdf"
        except ImportError:
            raise ImportError("请安装 pymupdf 或 pypdf: pip install pymupdf")

    # ============================================================
    # 结构化提取（主入口）
    # ============================================================
    def extract_structured(self, pdf_path: str) -> List[StructuredPage]:
        """结构化提取：返回带段落、表格、标题信息的页面列表"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        if self.engine == "pymupdf":
            return self._extract_structured_pymupdf(pdf_path)
        else:
            return self._extract_structured_pypdf(pdf_path)

    def _extract_structured_pymupdf(self, pdf_path: str) -> List[StructuredPage]:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        pages: List[StructuredPage] = []
        pdf_name = os.path.basename(pdf_path)

        # 先收集所有页的表格用于跨页合并
        all_page_tables = []

        for i, page in enumerate(doc):
            sp = StructuredPage(page_num=i + 1)
            sp.source_file = pdf_name

            # 提取原始文本块（保留位置信息）
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))  # 按y, x排序

            raw_text = page.get_text("text")
            sp.text = self._clean_text(raw_text)

            # 解析标题（基于字体大小判断）
            sp.headings = self._detect_headings(blocks)

            # 提取段落
            sp.paragraphs = self._extract_paragraphs(blocks, sp.headings)

            # 增强表格提取：检测标题+列类型
            sp.tables = self._extract_tables_enhanced(page, blocks)
            all_page_tables.append(sp.tables)

            pages.append(sp)

        doc.close()

        # 图像提取（工单4新增）：从PDF提取图像并关联到对应页面
        try:
            image_extractor = ImageExtractor()
            extracted_images = image_extractor.extract_from_pdf(pdf_path)
            for img in extracted_images:
                for page in pages:
                    if page.page_num == img.page_num:
                        page.images.append({
                            "image_path": img.image_path,
                            "image_type": img.image_type,
                            "description": img.description,
                            "caption_text": img.caption_text,
                            "surrounding_text": img.surrounding_text,
                            "width": img.width,
                            "height": img.height,
                        })
                        break
        except Exception as e:
            print(f"图像提取失败（不影响解析）: {e}")

        # 跨页表格合并（相同标题的表格）
        self._merge_cross_page_tables(pages, all_page_tables)

        return pages

    def _extract_structured_pypdf(self, pdf_path: str) -> List[StructuredPage]:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages: List[StructuredPage] = []
        pdf_name = os.path.basename(pdf_path)

        for i, page in enumerate(reader.pages):
            sp = StructuredPage(page_num=i + 1)
            sp.source_file = pdf_name
            text = page.extract_text() or ""
            sp.text = self._clean_text(text)

            # pypdf无法获取字体信息，用启发式方法检测标题和段落
            sp.headings = self._detect_headings_heuristic(text)
            sp.paragraphs = self._extract_paragraphs_heuristic(text)
            sp.tables = []  # pypdf不直接支持表格检测

            pages.append(sp)

        return pages

    # ============================================================
    # 标题检测
    # ============================================================
    def _detect_headings(self, blocks: List[Tuple]) -> List[Dict]:
        """基于字体大小检测标题"""
        headings = []
        if not blocks:
            return headings

        # 收集所有字体大小
        font_sizes = []
        for b in blocks:
            if len(b) >= 7 and b[6] > 0:
                font_sizes.append(b[6])

        if not font_sizes:
            return headings

        avg_size = sum(font_sizes) / len(font_sizes)
        threshold = avg_size * 1.15

        for b in blocks:
            if len(b) < 7:
                continue
            text = b[4].strip()
            font_size = b[6]

            if font_size >= threshold and len(text) < 80 and text:
                level = 1 if font_size > avg_size * 1.4 else 2
                headings.append({
                    "text": text,
                    "level": level,
                    "bbox": b[:4],
                    "font_size": font_size,
                })

        return headings

    def _detect_headings_heuristic(self, text: str) -> List[Dict]:
        """pypdf回退：基于文本特征检测标题"""
        headings = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < 60 and not line[-1] in "，。、；：！？,.;:!?":
                if line.isupper() or re.match(r"^[\d一二三四五六七八九十]+", line):
                    headings.append({"text": line, "level": 1})
        return headings

    # ============================================================
    # 段落提取
    # ============================================================
    def _extract_paragraphs(self, blocks, headings: List[Dict]) -> List[Dict]:
        paragraphs = []
        heading_texts = {h["text"] for h in headings}
        current_para = {"text": "", "type": "paragraph"}
        for b in blocks:
            text = b[4].strip()
            if not text or text in heading_texts:
                if current_para["text"]:
                    paragraphs.append(current_para)
                    current_para = {"text": "", "type": "paragraph"}
                continue
            current_para["text"] += " " + text if current_para["text"] else text
        if current_para["text"]:
            paragraphs.append(current_para)
        merged = []
        for p in paragraphs:
            if len(p["text"]) < 30 and merged:
                merged[-1]["text"] += " " + p["text"]
            else:
                merged.append(p)
        return merged

    def _extract_paragraphs_heuristic(self, text: str) -> List[Dict]:
        paragraphs = []
        raw_paras = text.split("\n\n")
        for p in raw_paras:
            p = p.strip()
            if len(p) > 10:
                paragraphs.append({"text": p, "type": "paragraph"})
        return paragraphs

    # ============================================================
    # 增强表格提取（核心优化）
    # ============================================================
    def _extract_tables_enhanced(self, page, blocks) -> List[Dict]:
        """
        增强表格提取：
        1. 使用 pymupdf.find_tables() 检测表格
        2. 检测表格标题（从附近文本块提取）
        3. 识别列类型（数值/文本/百分比/日期）
        4. 生成 Markdown 格式+结构化JSON
        5. 添加表格内容摘要
        """
        tables = []
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                headers = tab.header.names if tab.header else []
                data = tab.extract()
                if not data:
                    continue

                # 检测表格标题（扫描表格上方的文本块）
                title = self._detect_table_title(tab.bbox, blocks)

                # 识别列类型
                col_types = self._detect_column_types(headers, data)

                # 生成 Markdown 格式表格
                md_table = self._table_to_markdown(headers, data)
                # 生成简短文本描述
                summary = self._table_summary(headers, data)

                # 解析表格内容的结构化JSON
                structured_data = self._table_to_structured(headers, data)

                tables.append({
                    "headers": headers,
                    "rows": data,
                    "bbox": tab.bbox,
                    "title": title,
                    "column_types": col_types,
                    "markdown": md_table,
                    "summary": summary,
                    "structured": structured_data,
                    "has_numbers": any(t in ["numeric", "percentage"] for t in col_types),
                    "row_count": len(data),
                    "col_count": len(headers) if headers else (len(data[0]) if data else 0),
                })
        except Exception:
            pass
        return tables

    def _detect_table_title(self, table_bbox, blocks) -> str:
        """
        检测表格标题：扫描表格上方紧邻的文本块。
        策略：
        1. 取表格bbox正上方最近的非空文本块
        2. 匹配表格标题关键词模式
        3. 返回最可能的标题
        """
        if not blocks or not table_bbox:
            return ""

        # 表格的y坐标范围
        table_top = table_bbox[1]
        table_x0 = table_bbox[0]
        table_x1 = table_bbox[2]

        candidates = []
        for b in blocks:
            text = b[4].strip() if len(b) > 4 else ""
            if not text or len(text) < 2:
                continue
            # 文本块在表格上方
            block_bottom = b[3]  # y1
            block_top = b[1]     # y0
            # 在表格上方80pt范围内
            if table_top - 80 <= block_bottom <= table_top:
                # x坐标与表格重叠
                block_x0, block_x1 = b[0], b[2]
                if block_x0 < table_x1 and block_x1 > table_x0:
                    # 距离表格越近权重越高
                    distance = table_top - block_bottom
                    score = 1.0 / (distance + 1)
                    candidates.append((text, score))

        if candidates:
            # 取评分最高的
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]
            # 清理：去除短的无意义文本
            if len(best) > 2 and not best.startswith(("第", "表", "图")):
                return best

        # 如果没有检测到，检查blocks中最接近表格上方的文本
        # 利用表格标题模式匹配
        for b in blocks:
            text = b[4].strip() if len(b) > 4 else ""
            if not text:
                continue
            for pattern in self.TABLE_TITLE_PATTERNS:
                if re.search(pattern, text):
                    return text

        return ""

    def _detect_column_types(self, headers: list, rows: list) -> list:
        """
        识别列类型：遍历每列数据判断类型
        - numeric: 大部分是数字
        - percentage: 大部分是百分比
        - text: 大部分是文本
        - date: 大部分是日期格式
        """
        if not rows:
            return ["text"] * len(headers) if headers else []

        n_cols = len(rows[0]) if rows else 0
        if n_cols == 0:
            return []

        col_types = []
        for col_idx in range(n_cols):
            values = []
            for row in rows:
                if col_idx < len(row) and row[col_idx]:
                    values.append(str(row[col_idx]).strip())

            if not values:
                col_types.append("text")
                continue

            numeric_count = 0
            percentage_count = 0
            date_count = 0
            for v in values:
                v = v.replace(",", "").replace(" ", "")
                if re.match(r'^\d+(\.\d+)?$', v):
                    numeric_count += 1
                elif re.match(r'^\d+(\.\d+)?%$', v):
                    percentage_count += 1
                elif re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', v):
                    date_count += 1

            total = len(values)
            if percentage_count / total > 0.5:
                col_types.append("percentage")
            elif numeric_count / total > 0.5:
                col_types.append("numeric")
            elif date_count / total > 0.5:
                col_types.append("date")
            else:
                col_types.append("text")

        return col_types

    def _table_to_markdown(self, headers: list, rows: list) -> str:
        """将表格转换为Markdown表格格式（带对齐）"""
        if not rows:
            return ""

        lines = []
        if headers:
            lines.append("| " + " | ".join(str(h or "") for h in headers) + " |")
            # 对齐行：数值列右对齐，其他左对齐
            aligns = []
            for row in rows[:3]:
                for i, cell in enumerate(row):
                    cell_str = str(cell or "").strip()
                    if re.match(r'^[\d,\.%\-]+$', cell_str.replace(",", "")):
                        aligns.append("---:")
                    else:
                        aligns.append(":---")
                    break
            if not aligns:
                aligns = [":---"] * len(headers)
            lines.append("|" + "|".join(aligns) + "|")

        for row in rows:
            cells = [str(c or "").strip() for c in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _table_summary(self, headers: list, rows: list) -> str:
        """生成表格的简短文字摘要"""
        if not rows:
            return ""

        n_rows = len(rows)
        header_text = "、".join(str(h or "") for h in headers[:5]) if headers else ""
        if len(headers) > 5:
            header_text += "等"

        # 检查是否有数值列
        numeric_info = ""
        for col_idx, row in enumerate(rows[:1]):
            for i, cell in enumerate(row):
                cell_str = str(cell or "").strip()
                if re.match(r'^\d+[\.,]?\d*$', cell_str.replace(",", "")):
                    numeric_info = f"，含数值数据"
                    break

        return f"表格({n_rows}行{', 列: ' + header_text if header_text else ''}{numeric_info})"

    def _table_to_structured(self, headers: list, rows: list) -> list:
        """将表格转换为结构化JSON列表"""
        if not rows or not headers:
            return []

        result = []
        for row in rows:
            item = {}
            for i, h in enumerate(headers):
                if i < len(row):
                    cell = str(row[i] or "").strip()
                    # 尝试转换为数字
                    try:
                        clean = cell.replace(",", "").replace("%", "").strip()
                        if clean and re.match(r'^-?\d+(\.\d+)?$', clean):
                            item[str(h)] = float(clean)
                        else:
                            item[str(h)] = cell
                    except (ValueError, TypeError):
                        item[str(h)] = cell
                else:
                    item[str(h)] = ""
            result.append(item)
        return result

    # ============================================================
    # 跨页表格合并
    # ============================================================
    def _merge_cross_page_tables(self, pages, all_page_tables):
        """
        跨页合并：当同一个表格跨页（相同标题、相同表头），合并为一个。
        策略：如果 pageN 的最后一个表格的标题与 pageN+1 的第一个表格的
        表头相同，则合并。
        """
        if len(pages) < 2:
            return

        for page_idx in range(len(pages) - 1):
            current_tables = all_page_tables[page_idx]
            next_tables = all_page_tables[page_idx + 1]

            if not current_tables or not next_tables:
                continue

            last_tab = current_tables[-1]
            first_next = next_tables[0]

            # 检查标题或表头是否匹配
            title_match = (
                last_tab.get("title", "") and
                last_tab["title"] == first_next.get("title", "")
            )
            header_match = (
                last_tab.get("headers", []) and
                first_next.get("headers", []) and
                last_tab["headers"] == first_next["headers"]
            )

            if title_match or header_match:
                # 合并数据行
                last_tab["rows"].extend(first_next["rows"])
                # 更新 Markdown
                last_tab["markdown"] = self._table_to_markdown(
                    last_tab["headers"], last_tab["rows"]
                )
                # 更新结构化数据
                last_tab["structured"] = self._table_to_structured(
                    last_tab["headers"], last_tab["rows"]
                )
                # 更新摘要
                last_tab["row_count"] = len(last_tab["rows"])
                last_tab["summary"] = self._table_summary(
                    last_tab["headers"], last_tab["rows"]
                )
                # 标记跨页
                last_tab["cross_page"] = True
                last_tab["page_range"] = (
                    pages[page_idx].page_num,
                    pages[page_idx + 1].page_num
                )

                # 从下一页删除该表格（已合并到本页）
                del next_tables[0]
                # 从下一页的 StructuredPage 中删除
                pages[page_idx + 1].tables = pages[page_idx + 1].tables[1:]

    # ============================================================
    # 兼容旧接口
    # ============================================================
    def extract_text(self, pdf_path: str) -> List[Dict]:
        """兼容旧接口：返回[{page, text}]格式"""
        pages = self.extract_structured(pdf_path)
        return [{"page": p.page_num, "text": p.text} for p in pages]

    # ============================================================
    # 文本清洗
    # ============================================================
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\u3000', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def get_metadata(self, pdf_path: str) -> Dict:
        if self.engine == "pymupdf":
            import pymupdf
            doc = pymupdf.open(pdf_path)
            meta = {
                "页数": len(doc),
                "标题": doc.metadata.get("title", ""),
                "作者": doc.metadata.get("author", ""),
            }
            doc.close()
            return meta
        else:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            info = reader.metadata
            return {
                "页数": len(reader.pages),
                "标题": getattr(info, "title", ""),
            }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        pages = parser.extract_structured(sys.argv[1])
        print(f"共 {len(pages)} 页")
        # 统计表格信息
        total_tables = sum(len(p.tables) for p in pages)
        titled_tables = sum(1 for p in pages for t in p.tables if t.get("title"))
        print(f"表格总数: {total_tables}")
        print(f"有标题的表格: {titled_tables}")
        # 显示第一个有标题的表格
        for p in pages:
            for t in p.tables:
                if t.get("title"):
                    print(f"\n--- 第{p.page_num}页 表格: {t['title']} ---")
                    print(f"列: {t['headers']}")
                    print(f"列类型: {t['column_types']}")
                    print(f"行数: {t['row_count']}")
                    print(f"Markdown:\n{t['markdown'][:500]}")
                    break
            else:
                continue
            break
