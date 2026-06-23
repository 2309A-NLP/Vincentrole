# -*- coding: utf-8 -*-
"""把 webui.py 的默认 LLM(直接回复) 替换成 QwenRAG。幂等：已打过补丁则跳过。"""
import shutil, sys

WEBUI = "/root/Linly-Talker/webui.py"
OLD = """    llm_class = LLM(mode='offline')
    llm = llm_class.init_model('直接回复 Direct Reply')
    success_print("默认不使用LLM模型，直接回复问题，同时减少显存占用！")"""
NEW = """    from LLM.QwenRAG import QwenRAG
    llm = QwenRAG(model='qwen-plus', top_k=3)
    success_print("已启用 QwenRAG：通义千问 qwen-plus + 医疗知识库(score=5) RAG检索")"""

src = open(WEBUI, encoding="utf-8").read()
if "QwenRAG" in src:
    print("已打过补丁，跳过")
    sys.exit(0)
if OLD not in src:
    print("!! 未找到目标代码块，未修改。请手动检查 webui.py 第921行附近")
    sys.exit(1)
shutil.copy(WEBUI, WEBUI + ".bak")
open(WEBUI, "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("补丁完成，已备份 webui.py.bak")
