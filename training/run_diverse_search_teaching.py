#!/usr/bin/env python3
"""Run the diverse-50k search-teaching pipeline end to end."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def call(args):
    print("+"," ".join(map(str,args)),flush=True)
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zq-bridge",type=Path,required=True)
    p.add_argument("--claustro-bridge",type=Path,required=True)
    p.add_argument("--claustro-checkpoint",type=Path,required=True)
    p.add_argument("--zq-nnue",type=Path,default=ROOT/"results/experiments/race512-search10-ft-s20260917/student_int8.bin")
    p.add_argument("--max-positions",type=int,default=50000)
    p.add_argument("--teaching-fraction",type=float,default=0.10)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()

    positions=ROOT/"data/teaching/search-priority-diverse-50k/positions.jsonl"
    zq_targets=ROOT/"data/teaching/search-priority-diverse-50k/zq-search.npz"
    cl_targets=ROOT/"data/teaching/search-priority-diverse-50k/claustro-search.npz"
    search_dataset=ROOT/"data/teaching/search-priority-diverse-50k/dataset.npz"
    final_dataset=ROOT/"data/teaching/historical2m-search10-diverse50k/dataset.npz"
    replay=ROOT/"data/teaching/replay-historical-2m-cuda/dataset.npz"

    cmds=[
      [sys.executable,ROOT/"training/select_replay_diverse_disagreement.py",
       "--max-positions",a.max_positions,"--out",positions],
      [sys.executable,ROOT/"training/teachers/zq_deep_relabel.py",
       "--positions",positions,"--bridge",a.zq_bridge,"--nnue",a.zq_nnue,
       "--node-budgets","512,2048","--out",zq_targets],
      [sys.executable,ROOT/"training/teachers/claustrophobia_search_relabel.py",
       "--positions",positions,"--bridge",a.claustro_bridge,
       "--checkpoint",a.claustro_checkpoint,"--sims","256,1024","--out",cl_targets],
      [sys.executable,ROOT/"training/build_search_priority_dataset.py",
       "--source-dataset",replay,"--positions",positions,
       "--zq-targets",zq_targets,"--claustro-targets",cl_targets,
       "--out",search_dataset,
       "--zq-policy-weight","0.25","--claustro-policy-weight","0.75",
       "--zq-value-weight","0.25","--claustro-value-weight","0.75"],
      [sys.executable,ROOT/"training/mix_teaching_datasets.py",
       "--replay",replay,"--teaching",search_dataset,"--out",final_dataset,
       "--teaching-fraction",a.teaching_fraction],
    ]
    payload={"positions":str(positions),"zq_targets":str(zq_targets),
             "claustro_targets":str(cl_targets),"search_dataset":str(search_dataset),
             "final_dataset":str(final_dataset),"commands":[[str(x) for x in cmd] for cmd in cmds]}
    print(json.dumps(payload,indent=2))
    if a.dry_run: return 0
    for required in (a.zq_bridge,a.claustro_bridge,a.claustro_checkpoint,a.zq_nnue,replay):
        if not Path(required).is_file():
            raise SystemExit(f"missing required input: {required}")
    for cmd in cmds: call(cmd)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
