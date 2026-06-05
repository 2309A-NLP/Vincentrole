"""
工单7 - 快速功能测试
测试多格式文件加载功能
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_engine.orchestrator import RAGSystem
import config


def test_multi_format_loading():
    """测试多格式文件加载"""
    print("=" * 60)
    print("工单7 - 多格式文件加载测试")
    print("=" * 60)

    # 1. 初始化RAG系统
    print("\n1. 初始化RAG系统...")
    rag = RAGSystem(use_cache=True)

    # 2. 测试加载data目录下的文件
    print("\n2. 测试加载data目录...")
    data_dir = getattr(config, "DATA_DIR", "./data")
    ccf_dir = os.path.join(data_dir, "ccf_competition")

    if os.path.exists(ccf_dir):
        print(f"   找到ccf_competition目录: {ccf_dir}")

        # 统计文件
        pdf_count = 0
        txt_count = 0
        for root, dirs, files in os.walk(ccf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_count += 1
                elif file.lower().endswith('.txt'):
                    txt_count += 1

        print(f"   PDF文件: {pdf_count}")
        print(f"   TXT文件: {txt_count}")

        # 测试加载一个PDF和一个TXT
        test_files = []
        for root, dirs, files in os.walk(ccf_dir):
            for file in files:
                if file.lower().endswith('.pdf') and len(test_files) < 1:
                    test_files.append(os.path.join(root, file))
                elif file.lower().endswith('.txt') and len(test_files) < 2:
                    test_files.append(os.path.join(root, file))

        if test_files:
            print(f"\n3. 测试加载 {len(test_files)} 个文件...")
            for file_path in test_files:
                filename = os.path.basename(file_path)
                print(f"   加载: {filename}")
                result = rag.load_file(file_path)
                if result.get("status") == "ok":
                    print(f"     ✅ 成功: {result.get('chunks', 0)} 个chunk")
                else:
                    print(f"     ❌ 失败: {result.get('error', '未知错误')}")

            # 测试问答
            if rag.is_ready:
                print("\n4. 测试问答...")
                test_question = "这家公司的主要业务是什么？"
                result = rag.ask(test_question)
                print(f"   问题: {test_question}")
                print(f"   回答: {result.get('rag_answer', '')[:100]}...")
                print(f"   检索到 {len(result.get('source_chunks', []))} 个片段")
                print(f"   耗时: {result.get('elapsed', 'N/A')}")
            else:
                print("\n4. RAG系统未就绪，跳过问答测试")
        else:
            print("\n3. 没有找到测试文件")
    else:
        print(f"   警告: 找不到ccf_competition目录: {ccf_dir}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_multi_format_loading()
