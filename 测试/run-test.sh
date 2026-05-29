#!/bin/bash
# 运行JMeter完整测试脚本
JMETER_HOME="/Users/suwente/apache-jmeter-5.6.3"
TEST_PLAN="/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/roleplay-test-plan.jmx"
RESULTS_FILE="/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/results.jtl"
REPORT_DIR="/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/report"
JTL_DIR="/Users/suwente/Desktop/角色扮演系统-/jmeter-tests"

# 清理上次结果
rm -f "$RESULTS_FILE"
rm -rf "$REPORT_DIR"
mkdir -p "$(dirname "$RESULTS_FILE")"
mkdir -p "$REPORT_DIR"

echo "=== 角色扮演系统 JMeter 测试 ==="
echo "测试计划: $TEST_PLAN"
echo "结果文件: $RESULTS_FILE"
echo "报告目录: $REPORT_DIR"
echo ""

# 检查后端
echo "检查后端状态..."
curl -sf http://localhost:8080/health > /dev/null
if [ $? -ne 0 ]; then
    echo "错误：后端服务未运行，请先启动后端！"
    exit 1
fi
echo "后端状态：正常"
echo ""

# 运行测试（带进度输出）
"$JMETER_HOME/bin/jmeter" -n -t "$TEST_PLAN" -l "$RESULTS_FILE" -e -o "$REPORT_DIR" \
    -Jjmeter.reportgenerator.overall_granularity=1000

echo ""
echo "=== 测试完成 ==="
echo "结果文件: $RESULTS_FILE"
echo "HTML报告: $REPORT_DIR/index.html"
echo ""

# 显示汇总
echo "=== 结果摘要 ==="
if [ -f "$RESULTS_FILE" ]; then
    TOTAL=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    ERRORS=$(tail -n +2 "$RESULTS_FILE" | grep ",false," | wc -l)
    echo "总请求数: $TOTAL"
    echo "错误数: $ERRORS"
    if [ "$TOTAL" -gt 0 ]; then
        echo "成功率: $(awk "BEGIN {printf \"%.1f%%\", (${TOTAL}-${ERRORS})/${TOTAL}*100}")"
    fi
fi
echo ""
echo "各接口详情："
tail -n +2 "$RESULTS_FILE" 2>/dev/null | awk -F',' '
{
    labels[$2]++
    errors[$2,$8]++
    total_time += $2
}
END {
    for (l in labels) {
        err_count = 0
        for (k in errors) {
            split(k, arr, SUBSEP)
            if (arr[1] == l && arr[2] == "false") err_count = errors[k]
        }
        printf "  %-30s %3d次  %s失败=%d\n", l, labels[l], (err_count>0 ? "❌" : "✅"), err_count
    }
}'
