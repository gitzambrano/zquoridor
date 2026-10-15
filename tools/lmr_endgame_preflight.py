#!/usr/bin/env python3
"""Independent preflight before expensive 1000-game LMR endgame screening.

Generate legal game histories with root wall stock 0/N, and probe the very
same production MCAB/NNUE binary with and without aggressive depth-2 LMR.
Require actual AB leaf usage AND a nontrivial frequency of different
decisions; otherwise a thousand color-swapped games from these positions
cannot meaningfully resolve a tiny search parameter effect.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ARGS = ["--lmr-min-depth", "2", "--lmr-min-move", "2", "--lmr-divisor", "1.20"]

def check(engine, moves, tuning):
    script = "position startpos moves " + " ".join(moves) + "\ngo movetime 200\nquit\n"
    result = subprocess.run([engine, *tuning], input=script, text=True,
                            capture_output=True, timeout=45, check=True)
    lines = result.stdout.splitlines()
    if any("info string error" in l for l in lines):
        raise RuntimeError(result.stdout)
    info = next((l for l in lines if "lmrReductions=" in l), None)
    move = next((l.rsplit(None, 1)[-1] for l in lines if l.startswith("bestmove ")), None)
    if not info or not move or move == "(none)":
        raise RuntimeError("missing UCI LMR stats / bestmove: " + result.stdout)
    def get(name):
        m = re.search(r"\b" + re.escape(name) + r"=(\d+)", info)
        if not m: raise RuntimeError("missing " + name + ": " + info)
        return int(m.group(1))
    return move, get("lmrReductions"), get("abLeaves"), get("abNodes")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--books", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-different", type=int, default=3)
    a=p.parse_args()
    results=[]
    for book in a.books:
        lines=[json.loads(x) for x in Path(book).read_text().splitlines() if x.strip()]
        for k,row in enumerate(lines):
            original=check(a.engine, row["moves"], [])
            aggressive=check(a.engine, row["moves"], ARGS)
            if original[1] != 0: raise AssertionError("Production depth-2 AB unexpectedly reduced moves")
            if original[2] < 1: raise AssertionError("No AB leaves with production gate")
            if aggressive[1] < 1: raise AssertionError("Aggressive LMR never triggered on a fixture")
            results.append({"book":str(book),"index":k,"old_move":original[0],
                            "new_move":aggressive[0],"changed":original[0]!=aggressive[0],
                            "default_ab_leaves":original[2],"aggressive_reductions":aggressive[1],
                            "default_ab_nodes":original[3],"aggressive_ab_nodes":aggressive[3]})
    total=len(results); changed=sum(row["changed"] for row in results)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"samples":total,"changed":changed,"rows":results},indent=2)+"\n")
    print(f"Production endgame LMR preflight: changed={changed}/{total}, "
          f"reductions={sum(x['aggressive_reductions'] for x in results)}",flush=True)
    if changed < a.min_different:
        raise SystemExit(f"Preflight gate not met ({changed} changed moves < {a.min_different}); "
                         "skip 1000 games, mine competitive positions from real selfplay instead")
    print("Preflight passed: candidate behavior differs in production AB endgames",flush=True)

if __name__=="__main__":
    main()
