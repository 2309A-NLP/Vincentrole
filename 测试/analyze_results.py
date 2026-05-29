import csv
from collections import defaultdict

with open('results_final.jtl') as f:
    reader = csv.DictReader(f)
    stats = {}
    for row in reader:
        label = row['label']
        success = row['success'] == 'true'
        code = row['responseCode']
        msg = row.get('responseMessage', '')
        elapsed = int(row['elapsed'])
        if label not in stats:
            stats[label] = {'total': 0, 'ok': 0, 'err': 0, 'codes': set(), 'msgs': set(), 'times': []}
        stats[label]['total'] += 1
        stats[label]['times'].append(elapsed)
        if success:
            stats[label]['ok'] += 1
        else:
            stats[label]['err'] += 1
            stats[label]['codes'].add(code)
            stats[label]['msgs'].add(msg)

    print(f"{'状态':4s} {'API名称':40s} {'总数':6s} {'成功':6s} {'失败':6s} {'平均(ms)':10s} {'最大(ms)':10s} {'错误码':15s}")
    print('-' * 100)
    for label in sorted(stats):
        s = stats[label]
        status = '✅' if s['err'] == 0 else '❌'
        avg_t = sum(s['times']) / len(s['times'])
        max_t = max(s['times'])
        codes = ','.join(s['codes']) if s['codes'] else '-'
        print(f'{status:4s} {label:40s} {s["total"]:6d} {s["ok"]:6d} {s["err"]:6d} {avg_t:8.1f}ms {max_t:8d}ms {codes:15s}')

    total_ok = sum(s['ok'] for s in stats.values())
    total_err = sum(s['err'] for s in stats.values())
    total_all = total_ok + total_err
    print()
    print(f'总计: {total_all} 请求, 成功 {total_ok}, 失败 {total_err}, 成功率 {total_ok/total_all*100:.1f}%')
