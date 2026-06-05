"""快速测试RAG系统初始化（不加载PDF）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_engine.orchestrator import RAGSystem

print("测试RAG系统初始化...")
rag = RAGSystem()
print(f"✅ 初始化成功")
print(f"  存储后端: {rag.vector_store.__class__.__name__}")
print(f"  多轮对话: {rag.conversation_enabled}")
