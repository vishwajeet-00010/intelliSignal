"""Verify every protected preexisting artifact without rewriting any of it."""
import argparse,json
from pathlib import Path
import run_controlled as base
from artifact_io import write_json

def verify(output='integrity.json'):
    snapshots=['protected_before.json','protected_source_before.json']
    expected={};conflicts=[]
    for name in snapshots:
        data=json.loads((base.HERE/name).read_text())
        for relative,sha in data.items():
            if relative in expected and expected[relative]!=sha:conflicts.append(relative)
            expected[relative]=sha
    changed=[];missing=[]
    for relative,sha in expected.items():
        p=base.ROOT/relative
        if not p.is_file():missing.append(relative)
        elif base.digest(p)!=sha:changed.append(relative)
    outcome={'protected_files':len(expected),'changed':changed,'missing':missing,'snapshot_conflicts':conflicts,'passed':not(changed or missing or conflicts),'snapshots':snapshots,'policy':'No protected file repaired/replaced; changes are reported as found.'}
    write_json(base.HERE/output,outcome,True)
    print(json.dumps(outcome,indent=2))
    if not outcome['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='integrity.json');args=p.parse_args();verify(args.output)
