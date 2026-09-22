#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures as cf, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.external import local_arena

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",required=True); ap.add_argument("--weights",required=True)
    ap.add_argument("--candidate-arg",action="append",default=[])
    ap.add_argument("--baseline-arg",action="append",default=[])
    ap.add_argument("--pairs",type=int,default=100); ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20260921); ap.add_argument("--openings",required=True)
    ap.add_argument("--move-time-ms",type=int,default=200)
    ap.add_argument("--base-ms",type=int,default=0); ap.add_argument("--inc-ms",type=int,default=0)
    ap.add_argument("--max-plies",type=int,default=240); ap.add_argument("--bootstrap",type=int,default=20000)
    ap.add_argument("--output",default="")
    a=ap.parse_args()
    opens=[(i,json.loads(x)["moves"]) for i,x in enumerate(Path(a.openings).read_text().splitlines()) if x.strip()]
    sel=random.Random(a.seed).sample(opens,a.pairs)
    cand=[a.binary,"--nnue",a.weights,*a.candidate_arg]
    base=[a.binary,"--nnue",a.weights,*a.baseline_arg]
    def one(t):
        idx,op,color=t
        return local_arena.play_game(
            opponent="legacy_root01",opening_index=idx,opening=op,zq_player=color,
            zq_factory=lambda: local_arena.UciPlayer(cand,"ahead",startup_timeout_s=30),
            opponent_factory=lambda: local_arena.UciPlayer(base,"legacy_root01",startup_timeout_s=30),
            zq_budget=a.move_time_ms,opponent_budget=a.move_time_ms,move_timeout_s=60,
            max_plies=a.max_plies,run_id=f"ahead-root01-{a.seed}",
            clock_initial_ms=a.base_ms,clock_increment_ms=a.inc_ms)
    tasks=[(i,o,c) for i,o in sel for c in (0,1)]
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n,row in enumerate(ex.map(one,tasks),1):
            rows.append(row)
            if n%20==0 or n==len(tasks):
                ok=[r for r in rows if r["status"]=="ok"]
                w=sum(r["result"]==1 for r in ok); d=sum(r["result"]==.5 for r in ok); l=sum(r["result"]==0 for r in ok)
                print(f"[{n}/{len(tasks)}] AHEAD W/D/L={w}/{d}/{l}",flush=True)
    s=local_arena.summarize_pairs(rows,bootstrap=a.bootstrap,seed=a.seed)
    s.update(wins=sum(r.get("result")==1 for r in rows if r.get("status")=="ok"),
             draws=sum(r.get("result")==.5 for r in rows if r.get("status")=="ok"),
             losses=sum(r.get("result")==0 for r in rows if r.get("status")=="ok"))
    print("AB_SUMMARY",json.dumps(s,sort_keys=True))
    if a.output:
        p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"summary":s,"games":rows},indent=2)+"\n")
    return 1 if s["failed_games"] else 0
if __name__=="__main__": raise SystemExit(main())
