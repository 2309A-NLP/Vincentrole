#!/bin/bash
# 启动角色扮演系统后端服务

cd "/Users/suwente/Desktop/角色扮演系统-"

# 检查Python版本
python3 --version

# 检查依赖是否已安装
echo "检查依赖..."
pip3 list | grep -E "fastapi|uvicorn|streamlit|requests" || echo "依赖可能未安装"

# 启动后端服务
echo "启动后端服务在端口8080..."
python3 main.py