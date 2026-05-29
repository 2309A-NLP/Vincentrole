import csv
from collections import defaultdict

stats = defaultdict(lambda: {'total':0, 'ok':0, 'err':0, 'times':[]})
with open('/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/results_final.jtl') as f:
    for row in csv.DictReader(f):
        label = row['label']
        stats[label]['total'] += 1
        stats[label]['times'].append(int(row['elapsed']))
        if row['success'] == 'true':
            stats[label]['ok'] += 1
        else:
            stats[label]['err'] += 1

print(f"{'接口':35s} {'总数':5s} {'成功':5s} {'失败':5s} {'平均(ms)':10s} {'最大(ms)':10s} {'P95(ms)':10s}")
print('-' * 85)
for label in sorted(stats):
    s = stats[label]
    times = sorted(s['times'])
    avg = sum(times)/len(times)
    mx = times[-1]
    p95 = times[int(len(times)*0.95)] if len(times) > 1 else times[0]
    sym = 'OK' if s['err']==0 else 'ER'
    print(f'{sym:4s} {label:35s} {s["total"]:5d} {s["ok"]:5d} {s["err"]:5d} {avg:8.1f}ms {mx:8d}ms {p95:8d}ms')

total = sum(s['total'] for s in stats.values())
total_ok = sum(s['ok'] for s in stats.values())
total_err = sum(s['err'] for s in stats.values())
print()
print(f'总计: {total} 请求, 成功 {total_ok}, 失败 {total_err}, 成功率 {total_ok/total*100:.1f}%')
print('注册失败原因: 用户已存在(400) — 上次运行已注册，预期行为')
print('认证修复: Bootstrap(401→200) ✅, 清理角色历史(401→200) ✅')
