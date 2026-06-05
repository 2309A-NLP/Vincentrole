"""
工单编号: 人工智能NLP-RAG-图像内容解析及检索优化
全面验证 — 15个测试问题（含2个图像问题）检索质量 + 响应时间 + 图像解析统计

使用方法:
    python evaluation/validate_wo4.py

依赖:
    需先安装 pymupdf (pip install pymupdf)
"""

import os
import sys
import time
import json
from typing import List, Dict

# ============================================================
# 路径设置
# ============================================================
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

import config
from qa_engine.orchestrator import RAGSystem
from pdf_parser.image_extractor import ImageExtractor
from knowledge_base.image_embeddings import ImageEmbeddingEncoder
from knowledge_base.image_store import ImageStore
from evaluation.test_questions_wo4 import (
    TEST_QUESTIONS,
    get_questions_by_source,
    get_image_questions,
)

# ============================================================
# 打印标题
# ============================================================
print("=" * 75)
print("   工单4 图像解析优化版 — 15个测试问题全面验证")
print("   工单编号: 人工智能NLP-RAG-图像内容解析及检索优化")
print("=" * 75)


# ============================================================
# 辅助函数
# ============================================================
def count_image_types(images: List) -> Dict[str, int]:
    """统计各类图像数量"""
    type_counts = {}
    for img in images:
        if hasattr(img, "get"):
            t = img.get("image_type", "未知")
        else:
            t = getattr(img, "image_type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1
    return type_counts


def has_meaningful_answer(answer: str) -> bool:
    """判断答案是否有效（非空、非"未找到"）"""
    if not answer:
        return False
    no_info_markers = ["未找到", "无法回答", "没有找到", "找不到"]
    for marker in no_info_markers:
        if marker in answer[:60] and len(answer) < 60:
            return False
    if "未提及" in answer[:30] and len(answer) < 40:
        return False
    return True


# ============================================================
# 阶段1: 初始化RAG系统（文本管道）
# ============================================================
print("\n" + "-" * 75)
print("  [阶段1/4] 初始化RAG系统（文本解析 + 向量检索 + LLM生成）")
print("-" * 75)

system = None
try:
    system = RAGSystem(use_cache=True)
    if system._is_ready:
        print("  [OK] RAG系统初始化成功")
    else:
        print(f"  [WARN] RAG系统初始化状态异常: {getattr(system, '_init_error', 'unknown')}")
except Exception as e:
    print(f"  [ERROR] RAG系统初始化失败: {e}")
    print("  后续将使用纯图像检索模式（降级）")

# ============================================================
# 阶段2: 加载PDF文档
# ============================================================
print("\n" + "-" * 75)
print("  [阶段2/4] 加载PDF文档")
print("-" * 75)

pdf1 = os.path.join(config.DATA_DIR, "招股说明书1.pdf")
pdf2 = os.path.join(config.DATA_DIR, "招股说明书2.pdf")

pdf_files = [(pdf1, "兴图新科"), (pdf2, "力源信息")]
load_results = []
all_pages = 0
all_chunks = 0

for pdf_path, label in pdf_files:
    if not os.path.exists(pdf_path):
        print(f"  [ERROR] PDF不存在: {pdf_path}")
        continue
    if system:
        try:
            result = system.load_pdf(pdf_path)
            status = result.get("status", "error")
            pages = result.get("pages", 0)
            chunks = result.get("chunks", 0)
            total = result.get("total_chunks", chunks)
            tables = result.get("tables_found", 0)
            load_results.append(result)
            all_pages += pages
            all_chunks += total

            print(f"  [{status.upper()}] {label} ({os.path.basename(pdf_path)}): "
                  f"{pages}页, {chunks}个chunk, {tables}个表格")
        except Exception as e:
            print(f"  [ERROR] 加载{label}失败: {e}")
    else:
        print(f"  [SKIP] RAG系统不可用，跳过加载 {label}")

if system:
    print(f"\n  总chunk数: {all_chunks}")

# ============================================================
# 阶段3: 图像提取与索引
# ============================================================
print("\n" + "-" * 75)
print("  [阶段3/4] 图像提取与索引")
print("-" * 75)

all_images = []
image_extractor = None
image_store = None

try:
    # 初始化图像提取器
    image_extractor = ImageExtractor(config.IMAGE_CONFIG if hasattr(config, "IMAGE_CONFIG") else {})

    # 为每个PDF提取图像
    for pdf_path, label in pdf_files:
        if not os.path.exists(pdf_path):
            continue
        try:
            images = image_extractor.extract_from_pdf(pdf_path)
            print(f"  {label}: 提取 {len(images)} 个图像")
            all_images.extend(images)

            # 统计图像类型
            type_counts = count_image_types(images)
            for img_type, count in sorted(type_counts.items()):
                print(f"    - {img_type}: {count}个")

            # 打印每页图像数
            page_counts = {}
            for img in images:
                p = getattr(img, "page_num", 0)
                page_counts[p] = page_counts.get(p, 0) + 1
            if page_counts:
                pages_str = ", ".join(f"第{k}页({v}个)" for k, v in sorted(page_counts.items()))
                print(f"    分布: {pages_str}")

        except Exception as e:
            print(f"  [ERROR] {label}图像提取失败: {e}")

    print(f"\n  总计提取: {len(all_images)} 个图像")

    # 初始化图像编码器和图像存储
    if all_images:
        print("\n  初始化图像编码器...")
        # 使用BGE编码器回退（CLIP可能不可用）
        bge_encoder = system.embedding_model if system else None
        image_encoder = ImageEmbeddingEncoder(
            use_clip=False,
            device=config.IMAGE_CONFIG.get("可用设备", "cpu"),
            bge_encoder=bge_encoder,
        )
        image_store = ImageStore(encoder=image_encoder)

        # 索引图像
        image_dicts = [img.to_dict() if hasattr(img, "to_dict") else img for img in all_images]
        added_count = image_store.add_images(
            image_dicts,
            source_file="招股说明书",
        )
        print(f"  图像索引完成: {added_count} 个条目")
        print(f"  各类图像: {json.dumps(count_image_types(all_images), ensure_ascii=False)}")

except ImportError as e:
    print(f"  [WARN] 图像提取库不可用: {e}")
    print("  图像问题(Q5, Q6)将使用纯文本检索")
except Exception as e:
    print(f"  [ERROR] 图像处理阶段失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 阶段4: 运行全部15个测试问题
# ============================================================
print("\n" + "-" * 75)
print("  [阶段4/4] 运行15个测试问题")
print("-" * 75)

if not system:
    print("  [ERROR] RAG系统不可用，无法执行问答测试")
    sys.exit(1)

all_results = []
retrieval_times = []
image_question_times = []

image_q_ids = {q["id"] for q in get_image_questions()}
print(f"  图像相关问题ID: {sorted(image_q_ids)}")

print()
for i, q in enumerate(TEST_QUESTIONS):
    qid = q["id"]
    question = q["question"]
    source = q.get("source_pdf", "")
    category = q.get("category", "")
    is_image_q = qid in image_q_ids
    image_type = q.get("image_type", "")

    # 检索 + 生成计时
    try:
        start = time.time()
        result = system.ask(question)
        elapsed = time.time() - start
    except Exception as e:
        elapsed = 0
        result = {"error": str(e), "rag_answer": "", "source_chunks": []}

    chunks = result.get("source_chunks", [])
    num_chunks = len(chunks)
    query_analysis = result.get("query_analysis", {})
    error = result.get("error")
    answer = result.get("rag_answer", "")
    answer_ok = has_meaningful_answer(answer) and (error is None)
    has_info = "未找到" not in answer[:50] if answer else False

    source_files = result.get("source_files", [])
    top_scores = [c.get("final_score", c.get("score", 0)) for c in chunks[:3]]
    top_score = max(top_scores) if top_scores else 0

    # 如果是图像问题，尝试图像检索
    image_search_results = []
    if is_image_q and image_store:
        try:
            image_search_results = image_store.search(
                question,
                top_k=config.IMAGE_CONFIG.get("检索图像上限", 3),
            )
        except Exception as e:
            print(f"    [图像检索] 错误: {e}")

    all_results.append({
        "id": qid,
        "question": question,
        "source": source,
        "category": category,
        "is_image_question": is_image_q,
        "image_type": image_type,
        "num_chunks": num_chunks,
        "top_score": round(top_score, 4),
        "elapsed_s": round(elapsed, 3),
        "has_answer": answer_ok and has_info,
        "source_files": source_files,
        "error": error,
        "image_results": len(image_search_results),
        "answer_preview": answer[:120] if answer else "",
    })
    retrieval_times.append(elapsed)
    if is_image_q:
        image_question_times.append(elapsed)

    # 打印结果
    status = "PASS" if (answer_ok and has_info) else "FAIL"
    marker = ""
    if is_image_q:
        marker = " [图像]"
    elif qid in [1, 2, 3, 4]:
        marker = " [新增]"

    print(f"  [{status:4s}] q[{qid:3d}]{marker} {source}")
    print(f"         检索{num_chunks}chunk | 最高分{top_score:.4f} | {elapsed:.2f}s")
    if source_files:
        print(f"         来源: {', '.join(source_files)}")
    if error:
        print(f"         错误: {error}")
    if answer:
        print(f"         答案: {answer[:120]}...")
    if image_search_results:
        for ir in image_search_results[:2]:
            print(f"         [图像] p{ir['page_num']} {ir.get('image_type','')} "
                  f"score={ir.get('score',0):.4f}")

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 75)
print("  验证汇总报告")
print("=" * 75)

total = len(all_results)
success = sum(1 for r in all_results if r["has_answer"])
fail = total - success
avg_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
avg_top_score = sum(r["top_score"] for r in all_results) / total if total else 0

# 分文档统计
xingtu_results = [r for r in all_results if "招股说明书1" in r.get("source", "")]
liyuan_results = [r for r in all_results if "招股说明书2" in r.get("source", "")]
xingtu_success = sum(1 for r in xingtu_results if r["has_answer"])
liyuan_success = sum(1 for r in liyuan_results if r["has_answer"])

# 图像问题统计
image_results = [r for r in all_results if r.get("is_image_question")]
image_success = sum(1 for r in image_results if r["has_answer"])

print(f"\n  [总体统计]")
print(f"  总问题数:                {total}")
print(f"  通过 (PASS):             {success}/{total} ({success/total*100:.1f}%)")
print(f"  未通过 (FAIL):           {fail}/{total} ({fail/total*100:.1f}%)")
print(f"  平均响应时间:            {avg_time:.2f}s")
print(f"  平均最高检索分:          {avg_top_score:.4f}")

print(f"\n  [分文档统计]")
if xingtu_results:
    print(f"  兴图新科(10问):          {xingtu_success}/{len(xingtu_results)} "
          f"({xingtu_success/len(xingtu_results)*100:.1f}%)")
if liyuan_results:
    print(f"  力源信息(5问):           {liyuan_success}/{len(liyuan_results)} "
          f"({liyuan_success/len(liyuan_results)*100:.1f}%)")

print(f"\n  [图像问题统计]")
if image_results:
    print(f"  图像相关问题(2问):       {image_success}/{len(image_results)} "
          f"({image_success/len(image_results)*100:.1f}%)")
    if image_question_times:
        avg_img_time = sum(image_question_times) / len(image_question_times)
        print(f"  图像问题平均响应时间:    {avg_img_time:.2f}s")
else:
    print(f"  图像相关问题: 无")

print(f"\n  [图像提取统计]")
if all_images:
    type_counts = count_image_types(all_images)
    print(f"  总计提取: {len(all_images)} 个图像")
    for img_type, count in sorted(type_counts.items()):
        print(f"    {img_type}: {count}个")
    if image_store:
        print(f"  已索引: {image_store.get_image_count()} 个")
else:
    print(f"  未提取到图像")

within_3s = sum(1 for t in retrieval_times if t <= 3)
within_5s = sum(1 for t in retrieval_times if t <= 5)
print(f"\n  [响应时间分布]")
print(f"  <= 3秒:  {within_3s}/{total} ({within_3s/total*100:.1f}%)")
print(f"  <= 5秒:  {within_5s}/{total} ({within_5s/total*100:.1f}%)")
if retrieval_times:
    print(f"  最快:    {min(retrieval_times):.2f}s")
    print(f"  最慢:    {max(retrieval_times):.2f}s")

# ============================================================
# 详细结果表格
# ============================================================
print(f"\n  [详细结果]")
print(f"  {'ID':<5} {'结果':<6} {'图像':<5} {'类型':<12} {'chunk':<6} {'最高分':<10} {'耗时(s)':<10}")
print(f"  {'-'*54}")
for r in sorted(all_results, key=lambda x: x["id"]):
    status = "PASS" if r["has_answer"] else "FAIL"
    img_mark = "Y" if r.get("is_image_question") else "N"
    cat = r.get("category", "")[:10]
    print(f"  {r['id']:<5} {status:<6} {img_mark:<5} {cat:<12} "
          f"{r['num_chunks']:<6} {r['top_score']:<10.4f} {r['elapsed_s']:<10.2f}")

# ============================================================
# Pass/Fail 逐条明细
# ============================================================
print(f"\n  [Pass/Fail 逐条明细]")
for r in sorted(all_results, key=lambda x: x["id"]):
    status = "PASS" if r["has_answer"] else "FAIL"
    marker = ""
    if r.get("is_image_question"):
        marker = " [图像]"
    elif r["id"] in [1, 2, 3, 4]:
        marker = " [新增]"
    print(f"    {status:4s} | q[{r['id']:3d}]{marker} | {r['question'][:60]}...")

# ============================================================
# 保存结果
# ============================================================
output_path = os.path.join(os.path.dirname(__file__), "validation_results_wo4.json")
try:
    # 清理不可序列化的字段
    serializable_results = []
    for r in all_results:
        sr = {
            "id": r.get("id"),
            "question": r.get("question"),
            "source": r.get("source"),
            "category": r.get("category"),
            "is_image_question": r.get("is_image_question", False),
            "image_type": r.get("image_type", ""),
            "num_chunks": r.get("num_chunks", 0),
            "top_score": r.get("top_score", 0),
            "elapsed_s": r.get("elapsed_s", 0),
            "has_answer": r.get("has_answer", False),
            "source_files": r.get("source_files", []),
            "error": r.get("error"),
            "image_results": r.get("image_results", 0),
            "answer_preview": r.get("answer_preview", ""),
        }
        serializable_results.append(sr)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "pass": success,
                "fail": fail,
                "pass_rate": round(success / total * 100, 1) if total else 0,
                "avg_response_time": round(avg_time, 2),
                "avg_top_score": round(avg_top_score, 4),
                "xingtu_pass": xingtu_success,
                "xingtu_total": len(xingtu_results),
                "liyuan_pass": liyuan_success,
                "liyuan_total": len(liyuan_results),
                "image_question_pass": image_success,
                "image_question_total": len(image_results),
                "image_count": len(all_images),
            },
            "image_stats": count_image_types(all_images) if all_images else {},
            "results": serializable_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {output_path}")
except Exception as e:
    print(f"\n  [ERROR] 保存结果失败: {e}")

# ============================================================
# 完成
# ============================================================
print(f"\n{'='*75}")
print(f"  验证完成 | 通过: {success}/{total} ({success/total*100:.1f}%) "
      f"| 平均耗时: {avg_time:.2f}s")
print(f"{'='*75}")
