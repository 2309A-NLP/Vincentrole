#!/usr/bin/env python3
"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化
启动脚本
"""

import subprocess
import sys

cmd = [sys.executable, "-m", "streamlit", "run", "app/ui.py", "--server.headless", "true"]
subprocess.run(cmd)
