"""
工单16 - 数据格式转换脚本
将 IMDR QA 对 + PDF 专利图片 → LLaMA-Factory 多模态微调格式 (JSONL)

输出格式 (LLaMA-Factory Multi-modal):
  {"messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "patent_images/CN100342976C_p7.png"},
      {"type": "text", "text": "在文件中第7页的图片中，部件4相对于部件5在图片中的位置关系是？"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "A"}
    ]}
  ]}
"""
import json
import os
import re
import sys
from pathlib import Path

# ===== 路径配置 =====
QUESTIONS_FILE = "/Users/suwente/Desktop/专高六学习资料/RAG 新工单/14-17附件/original_problems/questions.jsonl"
DOCUMENTS_DIR = "/Users/suwente/Desktop/专高六学习资料/RAG 新工单/14-17附件/original_problems/documents"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "训练数据")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "patent_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ===== 解析 PDF 页面 → 图片 =====
def extract_page_images(questions):
    """
    从问题中提取需要哪些PDF的哪些页面，然后提取为图片
    返回 { (doc_name, page_num): image_path } 映射
    """
    import fitz

    needed = set()
    for q in questions:
        doc_name = q["document"]
        # 从问题文本中提取页号: "第N页的图片"
        m = re.search(r"第\s*(\d+)\s*页", q["question"])
        if m:
            needed.add((doc_name, int(m.group(1))))

    image_map = {}
    for doc_name, page_num in sorted(needed):
        pdf_path = os.path.join(DOCUMENTS_DIR, doc_name)
        if not os.path.exists(pdf_path):
            print(f"  ⚠ PDF不存在: {pdf_path}")
            continue
        try:
            doc = fitz.open(pdf_path)
            if page_num <= len(doc):
                page = doc[page_num - 1]  # 0-based
                # 提取为高分辨率图片
                mat = fitz.Matrix(2, 2)  # 2x zoom (大约200 DPI)
                pix = page.get_pixmap(matrix=mat)
                img_name = f"{Path(doc_name).stem}_p{page_num}.png"
                img_path = os.path.join(IMAGES_DIR, img_name)
                pix.save(img_path)
                image_map[(doc_name, page_num)] = img_path
                print(f"  ✅ {doc_name} 第{page_num}页 → {img_name}")
            else:
                print(f"  ⚠ {doc_name} 只有{len(doc)}页，请求第{page_num}页")
            doc.close()
        except Exception as e:
            print(f"  ❌ {doc_name} p{page_num}: {e}")

    return image_map


# ===== 转换 QA 对 → LLaMA-Factory 格式 =====
def convert_to_vlm_format(questions, image_map, groups=None):
    """
    将 QA 对转为 VLM 微调 JSONL
    groups: 筛选的问题组（如 [2] 只取图片题）
    """
    if groups is None:
        groups = [1, 2, 3]

    records = []
    skipped_no_image = 0
    skipped_no_group = 0

    for q in questions:
        if q.get("group") not in groups:
            skipped_no_group += 1
            continue

        doc_name = q["document"]
        m = re.search(r"第\s*(\d+)\s*页", q["question"])
        page_num = int(m.group(1)) if m else None

        # 构造消息
        user_content = []

        # 如果有图片，加上
        if page_num and (doc_name, page_num) in image_map:
            img_rel_path = f"patent_images/{Path(doc_name).stem}_p{page_num}.png"
            user_content.append({"type": "image", "image": img_rel_path})

        # 问题文本
        user_content.append({"type": "text", "text": q["question"]})

        # 答案（标准答案字母，如有选项加上选项文本）
        answer_text = q["answer"]
        if q.get("options"):
            option_map = {chr(65+i): opt for i, opt in enumerate(q["options"])}
            answer_text = f"{q['answer']}. {option_map.get(q['answer'], '')}"

        record = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer_text}
                ]}
            ]
        }
        records.append(record)

    return records, skipped_no_group, skipped_no_image


# ===== 统计评估集类别 =====
def categorize_questions(questions):
    """将问题分类为「专业术语」和「图纸推理」"""
    terminology_keywords = [
        "术语", "含义", "定义", "表示什么", "是什么", "属于",
        "淬火", "公差", "配合", "材料", "工艺", "参数",
    ]
    drawing_keywords = [
        "图片中", "图中", "部件", "位置关系", "位于", "所示",
        "标注", "箭头", "结构", "视图",
    ]

    terms, drawings = [], []
    for q in questions:
        qtext = q["question"]
        is_term = any(kw in qtext for kw in terminology_keywords)
        is_draw = any(kw in qtext for kw in drawing_keywords)
        if is_term:
            terms.append(q)
        elif is_draw:
            drawings.append(q)

    return terms, drawings


def main():
    print("=" * 60)
    print("工单16 - IMDR → VLM 微调数据转换")
    print("=" * 60)

    # 1. 加载全部 QA 对
    print("\n📖 加载问题数据...")
    questions = []
    with open(QUESTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"   总计 {len(questions)} 条 QA 对")

    # 2. 提取图片
    print("\n🖼 提取专利 PDF 图片...")
    image_map = extract_page_images(questions)
    print(f"   共提取 {len(image_map)} 张图片")

    # 3. 转换为 VLM 格式（仅 Group 2 图片题）
    print("\n🔄 转换 Group 2（图片推理题）...")
    records, skipped_group, skipped_img = convert_to_vlm_format(
        questions, image_map, groups=[2]
    )
    print(f"   转换 {len(records)} 条")
    print(f"   跳过(非目标组): {skipped_group}")

    # 4. 保存
    output_file = os.path.join(OUTPUT_DIR, "vlm_train_data.jsonl")
    with open(output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n💾 训练数据已保存: {output_file}")
    print(f"   总条数: {len(records)}")

    # 5. 分类统计
    print("\n📊 问题分类统计...")
    terms, drawings = categorize_questions(questions)
    print(f"   专业术语类: {len(terms)} 条")
    print(f"   图纸推理类: {len(drawings)} 条")

    # 6. 划分训练/评估集（8:2）
    split = int(len(records) * 0.8)
    train_file = os.path.join(OUTPUT_DIR, "vlm_train.jsonl")
    eval_file = os.path.join(OUTPUT_DIR, "vlm_eval.jsonl")
    with open(train_file, "w") as f:
        for r in records[:split]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_file, "w") as f:
        for r in records[split:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n📂 训练集: {train_file} ({split} 条)")
    print(f"📂 评估集: {eval_file} ({len(records) - split} 条)")
    print("\n✅ 转换完成!")


if __name__ == "__main__":
    main()
