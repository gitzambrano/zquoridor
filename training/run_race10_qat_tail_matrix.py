#!/usr/bin/env python3
"""Run race512-search10 QAT-tail ablations from the promoted race10 checkpoint."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/teaching/historical2m-search10-conservative/dataset.npz"
INIT=ROOT/"results/experiments/race512-search10-ft-s20260917/student.bin"
VARIANTS={"qat-tail20":20,"qat-tail10":30}

def make_cmd(label,start,device):
    out=ROOT/"results/experiments"/f"race512-search10-{label}-s20260919"
    return [
        sys.executable,str(ROOT/"training/run_experiment.py"),
        "--data",str(DATA),"--out-dir",str(out),
        "--architecture","race","--hidden","512",
        "--init-from",str(INIT),"--init-architecture","race","--init-hidden","512",
        "--no-from-scratch","--qat","--qat-start-epoch",str(start),
        "--epochs","40","--batch-size","4096","--lr","3e-5",
        "--schedule","cosine","--warmup-epochs","2","--min-lr","3e-6",
        "--trunk-lr-scale","0.1","--weight-decay","1e-5",
        "--policy-weight","1.0","--value-weight","1.0",
        "--train-scope","full","--grad-clip","1.0","--patience","14",
        "--seed","20260929","--device",device,"--cpu-threads","4",
        "--build","--no-benchmark"
    ]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only",choices=sorted(VARIANTS))
    p.add_argument("--device",default="cuda")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    labels=[a.only] if a.only else list(VARIANTS)
    payload=[]
    for label in labels:
        cmd=make_cmd(label,VARIANTS[label],a.device)
        payload.append({"label":label,"qat_start_epoch":VARIANTS[label],"command":cmd})
        if not a.dry_run:
            if not DATA.is_file(): raise SystemExit(f"dataset missing: {DATA}")
            subprocess.run(cmd,cwd=ROOT,check=True)
    print(json.dumps(payload,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
