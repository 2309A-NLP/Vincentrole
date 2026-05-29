import csv

with open('/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/results_final.jtl') as f:
    reader = csv.DictReader(f)
    bootstrap_rows = [r for r in reader if 'Bootstrap' in r['label']]
    print(f"Total Bootstrap entries: {len(bootstrap_rows)}")
    for r in bootstrap_rows[:5]:
        print(f"  code={r['responseCode']} success={r['success']} msg={r.get('responseMessage','')}")
    errs = [r for r in bootstrap_rows if r.get('success') != 'true']
    print(f"Bootstrap failures: {len(errs)}")
    for e in errs[:3]:
        print(f"  code={e['responseCode']} msg={e.get('responseMessage','')}")

with open('/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/results_final.jtl') as f:
    reader = csv.DictReader(f)
    del_rows = [r for r in reader if '清理角色历史' in r['label']]
    print(f"\nTotal 清理角色历史 entries: {len(del_rows)}")
    for r in del_rows[:5]:
        print(f"  code={r['responseCode']} success={r['success']} msg={r.get('responseMessage','')}")
    errs = [r for r in del_rows if r.get('success') != 'true']
    print(f"清理角色历史 failures: {len(errs)}")

# Also check auth-relevant ones
with open('/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/results_final.jtl') as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get('responseCode') != '200' and r.get('responseCode') != '201':
            print(f"\nNon-200: label={r['label']} code={r['responseCode']} success={r['success']}")
