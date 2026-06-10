#!/bin/bash
# 工单17：一键运行压测
# 使用方法: ./run_load_test.sh <API_KEY> <CHAT_ID>

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <API_KEY> <CHAT_ID>"
    echo "示例: $0 ragflow-xxxxxxxxxxxx xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    exit 1
fi

API_KEY=$1
CHAT_ID=$2
OUTPUT_DIR="/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/压测结果"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== RAGFlow API 压测 ==="
echo "API Key: ${API_KEY:0:20}..."
echo "Chat ID: $CHAT_ID"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查Python依赖
echo "检查Python依赖..."
python3 -c "import requests, aiohttp" 2>/dev/null || {
    echo "安装Python依赖..."
    pip3 install requests aiohttp
}

# 检查RAGFlow服务状态
echo "检查RAGFlow服务状态..."
if ! curl -s http://localhost:8080 > /dev/null; then
    echo "错误: RAGFlow Web界面无法访问"
    echo "请先启动RAGFlow服务: cd /Users/suwente/Desktop/ragflow/docker && docker compose --profile cpu up -d"
    exit 1
fi
echo "✓ RAGFlow服务正常"

# 运行快速验证
echo ""
echo "=== 步骤1: 快速验证 ==="
cd /Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试
python3 quick_test.py "$API_KEY" "$CHAT_ID"

# 询问是否继续
echo ""
read -p "是否继续运行完整压测？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 启动资源监控（后台）
echo ""
echo "=== 步骤2: 启动资源监控 ==="
python3 resource_monitor.py --duration 900 --interval 5 &
MONITOR_PID=$!
echo "资源监控已启动 (PID: $MONITOR_PID)"

# 等待监控启动
sleep 3

# 运行场景A压测
echo ""
echo "=== 步骤3: 运行场景A压测 (20并发，10分钟) ==="
cd /Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/研发
python3 ragflow_load_test.py \
    --api-key "$API_KEY" \
    --chat-id "$CHAT_ID" \
    --scenario A \
    --concurrent 20 \
    --duration 600 \
    --output "$OUTPUT_DIR"

# 运行场景B压测
echo ""
echo "=== 步骤4: 运行场景B压测 (10并发，10分钟) ==="
python3 ragflow_load_test.py \
    --api-key "$API_KEY" \
    --chat-id "$CHAT_ID" \
    --scenario B \
    --concurrent 10 \
    --duration 600 \
    --output "$OUTPUT_DIR"

# 停止资源监控
echo ""
echo "=== 步骤5: 停止资源监控 ==="
kill $MONITOR_PID 2>/dev/null || true
echo "资源监控已停止"

# 生成报告
echo ""
echo "=== 步骤6: 生成测试报告 ==="
echo "测试完成！"
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "请查看以下文件："
echo "  - 压测结果: $OUTPUT_DIR/report_*.json"
echo "  - 资源监控: /Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/性能分析/"
echo ""
echo "=== 压测完成 ==="
