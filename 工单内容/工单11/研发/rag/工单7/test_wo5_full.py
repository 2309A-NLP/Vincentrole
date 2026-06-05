"""测试工单5核心功能 - 多轮对话 + Milvus"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_engine.orchestrator import RAGSystem

def main():
    print("=" * 60)
    print("工单5测试 - 多轮对话 + Milvus存储")
    print("=" * 60)
    
    # 初始化
    print("\n1. 初始化RAG系统...")
    rag = RAGSystem(use_cache=True)
    
    # 加载PDF
    print("\n2. 加载PDF文档...")
    pdf_paths = [
        "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书1.pdf",
        "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单4/data/招股说明书2.pdf",
    ]
    
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            print(f"  加载: {os.path.basename(pdf_path)}")
            start = time.time()
            result = rag.load_pdf(pdf_path)
            elapsed = time.time() - start
            print(f"  结果: {result.get('status')}, chunks: {result.get('total_chunks')}, 耗时: {elapsed:.1f}s")
    
    if not rag.is_ready:
        print("❌ 系统未就绪")
        return
    
    # 测试多轮对话
    print("\n3. 测试多轮对话...")
    questions = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "这个公司的法定代表人是谁？",  # 测试指代消解
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n--- 第{i}轮 ---")
        print(f"用户: {q}")
        start = time.time()
        result = rag.ask(q)
        elapsed = time.time() - start
        
        if result.get("resolved_question") != q:
            print(f"消解: {result['resolved_question']}")
        
        answer = result.get("rag_answer", "")
        print(f"回答: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print(f"耗时: {elapsed:.2f}s")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()
