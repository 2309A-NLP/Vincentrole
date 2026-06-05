"""
工单5测试脚本 - 验证多轮对话 + Milvus存储
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_engine.orchestrator import RAGSystem

def test_conversation():
    """测试多轮对话功能"""
    print("=" * 60)
    print("工单5测试 - 多轮对话 + Milvus存储")
    print("=" * 60)
    
    # 初始化系统
    print("\n1. 初始化RAG系统...")
    rag = RAGSystem()
    
    # 加载PDF
    print("\n2. 加载PDF文档...")
    pdf_paths = [
        "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书1.pdf",
        "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf",
    ]
    
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            print(f"  加载: {os.path.basename(pdf_path)}")
            result = rag.load_pdf(pdf_path)
            print(f"  结果: {result.get('status')}, chunks: {result.get('total_chunks')}")
        else:
            print(f"  文件不存在: {pdf_path}")
    
    if not rag.is_ready:
        print("❌ 系统未就绪")
        return
    
    print("\n3. 测试多轮对话场景...")
    
    # 测试问题序列（工单要求的测试场景）
    test_questions = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "他参与的哪个工程荣获了国家科技进步一等奖？",  # "他"应消解为上一轮实体
        "这个公司的法定代表人是谁？",  # "这个公司"应消解为兴图新科
        "那武汉力源信息技术股份有限公司呢？",  # 切换公司
        "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 第{i}轮 ---")
        print(f"用户: {question}")
        
        start = time.time()
        result = rag.ask(question)
        elapsed = time.time() - start
        
        # 显示消解信息
        if result.get("resolved_question") != question:
            print(f"  [消解] {result['resolved_question']}")
        
        answer = result.get("rag_answer", "")
        print(f"助手: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  对话状态: {result.get('conversation_status', {})}")
    
    print("\n4. 测试完毕")

if __name__ == "__main__":
    test_conversation()
