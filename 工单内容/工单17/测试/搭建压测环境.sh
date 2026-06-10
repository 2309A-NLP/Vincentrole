#!/bin/bash
# 工单17：搭建压测环境
# 安装必要的压测工具

set -e

echo "=== 搭建RAGFlow压测环境 ==="

# 检查JMeter是否安装
if command -v jmeter &> /dev/null; then
    echo "✓ JMeter已安装: $(jmeter --version 2>&1 | head -1)"
else
    echo "安装JMeter..."
    if command -v brew &> /dev/null; then
        brew install jmeter
    else
        echo "请手动安装JMeter: https://jmeter.apache.org/download_jmeter.cgi"
        exit 1
    fi
fi

# 检查Python依赖
echo "安装Python压测依赖..."
pip3 install memory-profiler py-spy psutil requests 2>/dev/null || \
pip install memory-profiler py-spy psutil requests 2>/dev/null || \
echo "请手动安装: pip install memory-profiler py-spy psutil requests"

# 创建压测结果目录
mkdir -p /Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/压测结果
mkdir -p /Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/性能分析

echo "=== 压测环境搭建完成 ==="
