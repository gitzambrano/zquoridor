#!/usr/bin/env python3
"""Same-runner matched MCGS simulation throughput: pinned main vs candidate.

Uses actual UCI 'go movetime 250' and root_json simulations/elapsed_ms,
not raw fixed simulation budgets. Alternates engine order; every
candidate and baseline is tested on exactly the same legal positions.
"""
import argparse
import json
import os
import random
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run_one(exe: Path, nnue: Path, moves: list[str], duration: int) -> dict:
    msg="position startpos"+(" moves "+" ".join(moves) if moves else "")+"\ngo movetime "+str(duration)+"\nquit\n"
    env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    r=subprocess.run([str(exe.resolve()),"--nnue",str(nnue.resolve()),"--dump-root"],
                     input=msg,text=True,capture_output=True,cwd=str(ROOT),env=env,timeout=40)
    if r.returncode: raise RuntimeError(f"Engine error {exe}: {r.stderr[-800:]}")
    pre="info string root_json "
    info=[json.loads(line[len(pre):]) for line in r.stdout.splitlines() if line.startswith(pre)]
    if len(info)!=1 or "bestmove " not in r.stdout:
        raise RuntimeError(f"Missing UCI result {exe}: {r.stdout[-1000:]}")
    d=info[0]
    if d["simulations"]<=0 or d["elapsed_ms"]<=0:
        raise RuntimeError(f"No MCAB work {exe}: {d['simulations']} sims in {d['elapsed_ms']}ms")
    return {"simulations":int(d["simulations"]),"elapsed_ms":int(d["elapsed_ms"]),
            "bestmove":d["bestmove"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--main",required=True);p.add_argument("--candidate",required=True)
    p.add_argument("--nnue",default="data/nnue/nnue_weights_int8.bin")
    p.add_argument("--samples-per-book",type=int,default=16)
    p.add_argument("--movetime",type=int,default=250)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    books={"normal":ROOT/"tools/external/openings_normal_confirm_400.jsonl",
           "center":ROOT/"tools/external/openings_center_rush_sound_5k.jsonl"}
    samples=[]
    for label,book in books.items():
        records=[json.loads(row) for row in book.read_text().splitlines() if row.strip()]
        indices=list(range(len(records)))
        random.Random(20261018 + (label=="center")).shuffle(indices)
        for idx in indices[:args.samples_per_book]:
            samples.append((label,idx,records[idx]["moves"]))
    records=[]
    for i,(label,idx,moves) in enumerate(samples):
        order=("main","candidate") if i%2==0 else ("candidate","main")
        row={"family":label,"opening_index":idx}
        for kind in order:
            exe=Path(args.main if kind=="main" else args.candidate)
            row[kind]=run_one(exe,Path(args.nnue),moves,args.movetime)
        records.append(row)
        if (i+1)%8==0: print(f"Profiled {i+1}/{len(samples)} paired positions",flush=True)
    metrics={}
    for family in ("normal","center","all"):
        subset=[x for x in records if family=="all" or x["family"]==family]
        rates={}
        for kind in ("main","candidate"):
            total_sims=sum(x[kind]["simulations"] for x in subset)
            total_ms=sum(x[kind]["elapsed_ms"] for x in subset)
            rates[kind]=1000.0*total_sims/total_ms
        metrics[family]={"main_simulations_per_second":rates["main"],
                         "candidate_simulations_per_second":rates["candidate"],
                         "candidate_speed_ratio":rates["candidate"]/rates["main"],
                         "pairs":len(subset)}
    report={"schema":"zquoridor.mcgs_matched_nps.v1","movetime_ms":args.movetime,
            "metrics":metrics,"samples":records}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    print("MCGS MATCHED SPEED RESULT "+json.dumps(metrics),flush=True)

if __name__=="__main__":main()
