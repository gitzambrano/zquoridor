#!/usr/bin/env python3
"""Forensic first-divergence 3+2 roots from completed games, not a tournament.

Exactly eight genuine *first ZQ divergence* positions, under identical
native-clock UCI budgets, three immutable engines, adaptive ON/OFF.
Cold-start root is intentional; this checks source/time-controller invariants
without inheriting path-dependent game-tree statistics. It does NOT claim to
reproduce real-game tree reuse or win probability.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

# (label, exact common history BEFORE first divergent ZQ move,
#  COMBO's recorded clock_before_ms, archived first-divergent moves)
CASES = [
 ("ti_loss_2_1", "d7h f4v f7h g6v e6h e7v e2 d9 e3 c9 d3 c8 d4", 160850,
  ("c8h", "c7", "c7")),
 ("ti_loss_8_0", "c7v d3v d5h a1h d7h c8h e2 f9 e3 f8 e4 f7", 156255,
  ("e5", "e5", "f6h")),
 ("ti_loss_36_0", "h4h a3h g7h e3h h1h d9 f1 d8 g1 c3v", 167454,
  ("c6h", "c6h", "d7v")),
 ("cl_loss_2_0", "d7h f4v f7h g6v e6h e7v e2 d9", 173883,
  ("d2", "e3", "e3")),
 ("cl_loss_10_0", "b6v e5h e2v f4v h7h a3h e2 e8 h1h e7", 167351,
  ("e3", "f1h", "f1h")),
 ("cl_loss_32_0", "b8h h2v a1h h8v g7h b2v e2 e8 e3 e7 e4 e6", 160677,
  ("e5", "e5", "e3h")),
 ("cl_center_7_1", "e2 e8 e3 e7 e4 e6 d3h e7h f3h d4v d6v f6 h3h g6 f7v g7 e5", 166835,
  ("f5v", "e5v", "g6h")),
 ("cl_center_29_1", "e2 e8 e3 e7 e4 e6 e3h e7h d3v d6 c5h e5h d5v a5v b6v d7 a7h", 166975,
  ("e8v", "c7h", "d8v")),
]

DIAG = re.compile(r"(?<!\S)(diag_[a-z]+|treeHit|reusedNodes)=([^\s]+)")
def search(exe, nnue, case, mode):
    label, history, clock, archived = case
    commands = "uci\nisready\nposition startpos moves " + history
    commands += f"\ngo wtime {clock} btime {clock} winc 2000 binc 2000\nquit\n"
    cmd = [str(exe), "--nnue", str(nnue), "--dump-root"]
    if mode == "off": cmd.append("--diag-no-adaptive-time")
    p = subprocess.run(cmd, input=commands, text=True, capture_output=True, timeout=75)
    if p.returncode:
        raise RuntimeError(f"{label}/{mode}: return {p.returncode} {p.stderr[-2000:]}")
    if "info string error illegal position history" in p.stdout:
        raise RuntimeError(f"{label}: invalid historical position")
    lines = p.stdout.splitlines()
    uci = [x for x in lines if x.startswith("info depth ") and "diag_sims=" in x]
    best = [x.split()[1] for x in lines if x.startswith("bestmove ")]
    roots = [json.loads(x.split("root_json ",1)[1]) for x in lines if "root_json " in x]
    if not (len(uci)==len(best)==len(roots)==1):
        raise RuntimeError(f"missing metrics {label} {mode}: {lines[-8:]}")
    pairs = dict(DIAG.findall(uci[0]))
    needed = ("diag_sims","diag_nodes","diag_target","diag_class","diag_stop",
              "diag_adaptive","diag_optimum","diag_expandcap")
    if any(x not in pairs for x in needed):
        raise RuntimeError(f"missing diag: {uci[0]}")
    if int(pairs["diag_adaptive"])!=(mode=="on"):
        raise RuntimeError(f"wrong mode: {uci[0]}")
    r=roots[0]
    if r["bestmove"]!=best[0]: raise RuntimeError(f"root mismatch {label}")
    return {"move":best[0],
            "simulations":int(pairs["diag_sims"]),
            "expanded_nodes":int(pairs["diag_nodes"]),
            "target_ms":int(pairs["diag_target"]),
            "optimum_ms":int(pairs["diag_optimum"]),
            "root_class":int(pairs["diag_class"]),
            "stop_reason":int(pairs["diag_stop"]),
            "node_cap":int(pairs["diag_expandcap"]),
            "tree_hit":int(pairs.get("treeHit",-1)),
            "elapsed_ms":float(r["elapsed_ms"]),
            "root":r}

def main():
    p=argparse.ArgumentParser()
    for eng in ("main","lazy","combo"):p.add_argument("--"+eng,type=Path,required=True)
    p.add_argument("--nnue",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    exes={e:getattr(a,e).resolve() for e in ("main","lazy","combo")}
    output=[]
    for i,case in enumerate(CASES):
        order=["main","lazy","combo"]
        order=order[i%3:]+order[:i%3]
        row={"case_id":case[0],"moves":case[1].split(),"clock_ms":case[2],
             "archived_first_divergence":{"main":case[3][0],"lazy":case[3][1],
                                           "combo":case[3][2]},
             "by_mode":{}}
        for mode in (("on","off") if i%2==0 else ("off","on")):
            row["by_mode"][mode]={}
            for eng in order:
                row["by_mode"][mode][eng]=search(exes[eng],a.nnue.resolve(),case,mode)
        # Only check the initial cap: no prior tree and no child visit escalation
        # assumed. A transient volatile classification can still escalate ON.
        output.append(row)
        print("DONE FIRST-DIVERGENCE",i+1,"/8",
              "ON moves", {e:row["by_mode"]["on"][e]["move"] for e in order},
              "OFF moves",{e:row["by_mode"]["off"][e]["move"] for e in order},
              flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({"schema":"zquoridor.diag.first_divergences.v1",
                  "positions":len(CASES),"mode":"cold-root native 3+2 ON/OFF, same runner",
                  "results":output},indent=2)+"\n")
    print("SUCCESS EIGHT FIRST-DIVERGENCE POSITIONS",flush=True)
if __name__=="__main__":main()
