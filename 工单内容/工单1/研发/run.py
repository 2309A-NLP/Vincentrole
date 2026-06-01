"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统

PDF智能问答系统 - 启动脚本

使用方法:
    streamlit run app/ui.py
    
或直接运行本脚本:
    python run.py
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(project_root, "app", "ui.py")

    print("=" * 60)
    print("  PDF智能问答系统")
    print("  工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统")
    print("=" * 60)
    print()
    print("正在启动Streamlit...")
    print("如果浏览器未自动打开，请访问: http://localhost:8501")
    print()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", "8501",
    ])
