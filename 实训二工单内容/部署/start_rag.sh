#!/bin/bash
# Linly-Talker + 医疗RAG 一键启动
cd /root/Linly-Talker
pkill -f "miniconda3/bin/python webui.py" 2>/dev/null   # 只杀真正的服务进程
sleep 2
export TOKENIZERS_PARALLELISM=false
setsid /root/miniconda3/bin/python webui.py > /root/Linly-Talker/webui.log 2>&1 < /dev/null &
echo "已后台启动，约90秒后访问 AutoDL自定义服务(6006)。看日志: tail -f /root/Linly-Talker/webui.log"
