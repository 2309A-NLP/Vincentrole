"""
工单15 - 测试脚本: 6题跨模态检索测试
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from config import TEST_QUESTIONS, PATENT_IMAGE_DIR
from cross_modal_retriever import CrossModalRetriever


def test_text_question(q, retriever):
    """测试纯文本问题: 直接调用 LLM"""
    print(f"\n▶ [{q['id']}] {q['question'][:50]}...")
    visual_ref = retriever.detect_visual_ref(q["question"])
    enhanced = retriever.build_enhanced_query(q["question"], visual_ref)
    print(f"   增强查询: {enhanced[:60]}...")
    answer = retriever.answer(q["question"], text_context="", use_vl=False)
    return check_answer(q, answer)


def test_vision_question(q, retriever):
    """测试图文问题: 使用 VL 模型 + 图片"""
    print(f"\n▶ [{q['id']}] {q['question'][:50]}...")
    visual_ref = retriever.detect_visual_ref(q["question"])
    print(f"   视觉引用: 第{visual_ref['page']}页 {visual_ref.get('figure','')}")
    answer = retriever.answer(q["question"], use_vl=True)
    return check_answer(q, answer)


def check_answer(q, answer):
    """检查答案是否正确"""
    expected = q["answer"]
    # 提取选项字母
    ans_letter = answer.strip()[0] if answer.strip() else ""
    correct = ans_letter == expected
    status = "✅" if correct else "❌"
    print(f"   {status} 预期={expected}, 实际={ans_letter}")
    print(f"   回答: {answer[:80]}")
    return correct


def main():
    print("=" * 60)
    print("工单15 - 跨模态检索测试")
    print("=" * 60)

    retriever = CrossModalRetriever(patent_image_dir=PATENT_IMAGE_DIR)
    results = []

    for q in TEST_QUESTIONS:
        if q["type"] == "text":
            correct = test_text_question(q, retriever)
        else:
            correct = test_vision_question(q, retriever)
        results.append({"id": q["id"], "correct": correct})

    # 汇总
    score = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"📊 测试结果: {score}/{total} = {score/total*100:.0f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
