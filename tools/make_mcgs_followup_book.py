#!/usr/bin/env python3
"""Build a disjoint follow-up opening book after both earlier MCGS screens.

Excludes all initial 25 pairs (screening seed=20261017), and all 120
positions used to draw the subsequent 100-pair H2H / 50-pair external runs.
Use explicit seed to keep 3+2 cohorts separate from 200ms combination cohorts.
This does not waive legality validation; the game runners referee every opening.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--pairs",type=int,required=True)
    p.add_argument("--seed",type=int,required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in Path(a.source).read_text(encoding="utf8").splitlines() if x.strip()]
    universe=list(range(len(rows)))
    initial=universe.copy()
    random.Random(20261017).shuffle(initial)
    excluded=set(initial[:25])
    earlier=universe.copy()
    earlier=[v for v in earlier if v not in excluded]
    random.Random(20261027).shuffle(earlier)
    excluded.update(earlier[:120])
    available=[j for j in universe if j not in excluded]
    random.Random(a.seed).shuffle(available)
    selected=available[:a.pairs]
    if len(selected)!=a.pairs or len(set(selected))!=len(selected):
        raise ValueError("Not enough unseen opening positions for disjoint confirmation")
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps({**rows[i],"original_book_index":i},separators=(",",":"))+"\n" for i in selected),encoding="utf8")
    print(f"Selected {len(selected)} distinct openings of {len(rows)} excluding {len(excluded)} prior; seed={a.seed}",flush=True)

if __name__=="__main__":
    main()
