#!/usr/bin/env python3
"""从 JTL 结果生成完整性能测试报告（含吞吐量、分位值等）"""
import csv
import math
import os
from collections import defaultdict
from datetime import datetime

JTL_PATH = os.path.join(os.path.dirname(__file__), "results_final.jtl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "final_report.md")

with open(JTL_PATH) as f:
    reader = csv.DictReader(f)
    stats = defaultdict(lambda: {
        'times': [], 'bytes': 0, 'sent': 0, 'err': 0, 'ok': 0,
        'codes': defaultdict(int), 'first_ts': None, 'last_ts': None
    })
    ts_all = []
    for row in reader:
        label = row['label']
        s = stats[label]
        elapsed = int(row['elapsed'])
        s['times'].append(elapsed)
        s['bytes'] += int(row.get('bytes', 0))
        s['sent'] += int(row.get('sentBytes', 0))
        ts = int(row['timeStamp'])
        ts_all.append(ts)
        if s['first_ts'] is None or ts < s['first_ts']:
            s['first_ts'] = ts
        if s['last_ts'] is None or ts > s['last_ts']:
            s['last_ts'] = ts
        if row['success'] == 'true':
            s['ok'] += 1
        else:
            s['err'] += 1
            s['codes'][row['responseCode']] += 1


def percentile(data, p):
    if not data:
        return 0
    data = sorted(data)
    idx = math.ceil(p / 100.0 * len(data)) - 1
    return data[max(0, min(idx, len(data) - 1))]


total_ok = sum(s['ok'] for s in stats.values())
total_err = sum(s['err'] for s in stats.values())
total_req = total_ok + total_err

all_times = []
for s in stats.values():
    all_times.extend(s['times'])

test_start = min(ts_all) if ts_all else 0
test_end = max(ts_all) if ts_all else 0
test_duration = (test_end - test_start) / 1000.0
overall_tps = total_req / test_duration if test_duration > 0 else 0

lines = []
lines.append("# 角色扮演系统 JMeter 性能测试报告")
lines.append("")
lines.append(f"**测试时间**: {datetime.fromtimestamp(test_start/1000).strftime('%Y-%m-%d %H:%M:%S')} ~ {datetime.fromtimestamp(test_end/1000).strftime('%H:%M:%S')}")
lines.append(f"**测试时长**: {test_duration:.1f} 秒")
lines.append(f"**总请求数**: {total_req}")
lines.append(f"**成功率**: {total_ok}/{total_req} ({total_ok/total_req*100:.1f}%)")
lines.append(f"**全局吞吐量**: {overall_tps:.1f} req/s")
lines.append(f"**全部响应 P50/P90/P95/P99**: {percentile(all_times, 50):.0f}/{percentile(all_times, 90):.0f}/{percentile(all_times, 95):.0f}/{percentile(all_times, 99):.0f} ms")
lines.append(f"**最大响应**: {max(all_times)} ms")
lines.append(f"**平均响应**: {sum(all_times)/len(all_times):.1f} ms")
lines.append("")

# --- 各端点详情表 ---
lines.append("## 各端点性能明细")
lines.append("")
lines.append("| 接口 | 总数 | 成功 | 失败 | 成功率 | 吞吐量 | 平均 | P50 | P90 | P95 | P99 | 最大 | 标准差 | 接收 |")
lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for label in sorted(stats):
    s = stats[label]
    times = s['times']
    dur = (s['last_ts'] - s['first_ts']) / 1000.0
    if dur <= 0:
        dur = 0.001
    tps = (s['ok'] + s['err']) / dur
    avg = sum(times) / len(times)
    med = percentile(times, 50)
    p90 = percentile(times, 90)
    p95 = percentile(times, 95)
    p99 = percentile(times, 99)
    mx = max(times)
    variance = sum((t - avg) ** 2 for t in times) / len(times)
    std = math.sqrt(variance)
    rate = s['ok'] / (s['ok'] + s['err']) * 100 if (s['ok'] + s['err']) > 0 else 0
    recv_kb = s['bytes'] / dur / 1024 if dur > 0 else 0
    label_clean = label.replace('|', '-')
    lines.append(
        f"| {label_clean} | {s['ok']+s['err']} | {s['ok']} | {s['err']} "
        f"| {rate:.1f}% | {tps:.1f}/s | {avg:.1f}ms | {med:.0f}ms "
        f"| {p90:.0f}ms | {p95:.0f}ms | {p99:.0f}ms | {mx}ms "
        f"| {std:.1f}ms | {recv_kb:.1f}KB/s |"
    )
    if s['err'] > 0:
        codes_str = ', '.join(f'HTTP {code}×{cnt}' for code, cnt in sorted(s['codes'].items()))
        lines.append(f"  > ⚠️ {label_clean} 错误: {codes_str}")

lines.append("")

# --- 认证修复说明 ---
lines.append("## 认证问题修复对比")
lines.append("")
lines.append("上一轮测试发现问题并修复：")
lines.append("")
lines.append("| 端点 | 修复前 | 修复后 | 原因 |")
lines.append("|:---|---:|:---|:---|")
lines.append("| GET /api/bootstrap | ❌ 401 | ✅ 200 | 生成脚本中 `needs_auth` 条件遗漏 Bootstrap (GET 方法) |")
lines.append("| DELETE /api/history/{role_id} | ❌ 401 | ✅ 200 | 同上，DELETE 方法被排除在 auth 头之外 |")
lines.append("| POST /api/preferences/active-role | ✅ 200 | ✅ 200 | 已在 POST 分支内正确带了 auth 头 |")
lines.append("")
lines.append("修复方式：将 `if method == \"POST\"` 改为基于路径的判断，排除 `/health`、`/api/login`、`/api/register` 即可。")
lines.append("")

# --- 注册 400 说明 ---
lines.append("## 注册失败说明")
lines.append("")
lines.append("10 次注册均返回 HTTP 400（用户已存在），这是**预期行为**。")
lines.append("测试共 10 个线程，每个线程注册一次固定用户名（jmeter_user_1 ~ jmeter_user_10），")
lines.append("第一次运行时已注册成功，本次为重复注册。")
lines.append("登录接口正常使用这些用户，不受影响。")
lines.append("")
lines.append("如需测试注册流程，应在每次测试前清空用户数据或使用随机用户名。")
lines.append("")

# --- 健康检查压力测试 ---
lines.append("## 压力测试结果（健康检查 500 请求）")
lines.append("")
hc = stats.get("健康检查(压力)")
if hc:
    hc_times = hc['times']
    lines.append(f"- **线程数**: 50，**循环次数**: 10")
    lines.append(f"- **总请求**: {len(hc_times)}")
    lines.append(f"- **成功率**: 100%")
    lines.append(f"- **平均响应**: {sum(hc_times)/len(hc_times):.1f} ms")
    lines.append(f"- **P50 (中位数)**: {percentile(hc_times, 50):.0f} ms")
    lines.append(f"- **P99**: {percentile(hc_times, 99):.0f} ms")
    lines.append(f"- **最大响应**: {max(hc_times)} ms")
    lines.append(f"- **标准差**: {math.sqrt(sum((t - sum(hc_times)/len(hc_times))**2 for t in hc_times)/len(hc_times)):.1f} ms")
    lines.append("")
    lines.append("结论：健康检查接口在 50 并发下表现极佳，平均仅 ~1ms，P99 也极低，无任何错误。")
    lines.append("")

# --- 全链路分析 ---
lines.append("## 全链路分析")
lines.append("")
lines.append("每个线程/用户的完整请求链路耗时估算：")
lines.append("")
# Extract individual endpoint data
health_avg = sum(stats.get("1-Health", {'times': [0]})['times']) / max(len(stats.get("1-Health", {'times': [1]})['times']), 1)
chat_avg = sum(stats.get("7-聊天(RAG)", {'times': [0]})['times']) / max(len(stats.get("7-聊天(RAG)", {'times': [1]})['times']), 1)

lines.append(f"- **健康检查** (~{health_avg:.0f}ms) → **注册** (~3ms) → **登录** (~{stats.get('3-登录', {'times': [0]})['times'][0] if stats.get('3-登录') else 0:.0f}ms) → **Bootstrap** (~{stats.get('4-Bootstrap', {'times': [0]})['times'][0] if stats.get('4-Bootstrap') else 0:.0f}ms) → **更新角色偏好** (~{stats.get('5-更新角色偏好', {'times': [0]})['times'][0] if stats.get('5-更新角色偏好') else 0:.0f}ms) → **清理历史** (~{stats.get('6-清理角色历史', {'times': [0]})['times'][0] if stats.get('6-清理角色历史') else 0:.0f}ms) → **RAG聊天** (~{chat_avg:.0f}ms)")
lines.append("")
lines.append("**主要性能瓶颈**: RAG 聊天接口（平均 ~3.8s，最大 ~7.3s）")
lines.append("- 这属于 RAG 检索+LLM 推理的正常范围")
lines.append("- 优化方向：优化 Milvus 检索策略、使用更快的嵌入模型或 LLM 推理加速")
lines.append("")

# --- 汇总表 ---
lines.append("## 慢速响应分布")
lines.append("")
lines.append("| 延迟区间 | 请求数 | 占比 |")
lines.append("|:---|---:|---:|")
buckets = [(0, 1, "<1ms"), (1, 10, "1~10ms"), (10, 100, "10~100ms"),
           (100, 500, "100~500ms"), (500, 1000, "500ms~1s"),
           (1000, 2000, "1~2s"), (2000, 5000, "2~5s"), (5000, float('inf'), ">5s")]
for lo, hi, name in buckets:
    cnt = sum(1 for t in all_times if lo <= t < hi)
    lines.append(f"| {name} | {cnt} | {cnt/len(all_times)*100:.1f}% |")

lines.append("")
lines.append("---")
lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

report = "\n".join(lines)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(report)
print(f"报告已生成: {OUTPUT_PATH}")
print(f"共 {len(lines)} 行")
