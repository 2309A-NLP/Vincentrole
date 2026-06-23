# -*- coding: utf-8 -*-
"""修复 llm_class 未定义 + 让界面"Qwen"选项使用 QwenRAG。幂等。"""
import shutil, sys

WEBUI = "/root/Linly-Talker/webui.py"
src = open(WEBUI, encoding="utf-8").read()

# --- 编辑1：在 QwenRAG 初始化后补回 llm_class（供界面切换其它LLM） ---
ANCHOR = '''    llm = QwenRAG(model='qwen-plus', top_k=3)
    success_print("已启用 QwenRAG：通义千问 qwen-plus + 医疗知识库(score=5) RAG检索")'''
ADD = ANCHOR + '''
    llm_class = LLM(mode='offline')  # 界面切换其它LLM时用'''

# --- 编辑2：界面"Qwen"选项 → QwenRAG（千问API+知识库） ---
OLD_QWEN = '''        elif model_name == 'Qwen':
            llm = llm_class.init_model('Qwen', 'Qwen/Qwen-1_8B-Chat', prefix_prompt=PREFIX_PROMPT)
            gr.Info("Qwen模型导入成功")'''
NEW_QWEN = '''        elif model_name == 'Qwen':
            from LLM.QwenRAG import QwenRAG
            llm = QwenRAG(model='qwen-plus', top_k=3)
            gr.Info("通义千问 qwen-plus + 医疗知识库RAG 已启用")'''

changed = False
if "llm_class = LLM(mode='offline')  # 界面切换其它LLM时用" not in src:
    if ANCHOR not in src:
        print("!! 未找到 QwenRAG 初始化锚点，请检查 webui.py")
        sys.exit(1)
    src = src.replace(ANCHOR, ADD)
    changed = True

if NEW_QWEN not in src:
    if OLD_QWEN not in src:
        print("!! 未找到 Qwen 分支，请检查 llm_model_change")
        sys.exit(1)
    src = src.replace(OLD_QWEN, NEW_QWEN)
    changed = True

if not changed:
    print("已是最新，无需修改")
    sys.exit(0)

shutil.copy(WEBUI, WEBUI + ".bak2")
open(WEBUI, "w", encoding="utf-8").write(src)
print("patch2 完成，已备份 webui.py.bak2")
