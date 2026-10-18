#!/usr/bin/env python3
"""Deterministic disjoint-opening selector for MCGS confirmation after 100+400 games.

Screening used first 25 of random.Random(20261017).shuffle(all rows).
The previous deep 200ms confirmation generated first 120 positions of
random.Random(20261027).shuffle(remaining); H2H sampled 100 of these.
Exclude all 145 for genuinely new positions (not necessarily distinct
from unrelated historical projects). Refs and side inversion match across
candidate and control; independent referee validation happens in arena.
"""
import argparse
import json
import random
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--count",type=int,required=True)
    p.add_argument("--seed",type=int,default=20261030)
    a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.source).read_text(encoding="utf-8").splitlines() if line.strip()]
    order=list(range(len(rows)))
    random.Random(20261017).shuffle(order)
    old_screen=set(order[:25])
    remaining=[i for i in range(len(rows)) if i not in old_screen]
    random.Random(20261027).shuffle(remaining)
    old_deep=set(remaining[:120])
    valid=[i for i in range(len(rows)) if i not in old_screen and i not in old_deep]
    random.Random(a.seed).shuffle(valid)
    chosen=valid[:a.count]
    assert len(chosen)==a.count
    assert not set(chosen).intersection(old_screen|old_deep)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8") as f:
        for i in chosen:
            row=dict(rows[i])
            row["original_book_index"]=i
            f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    print(f"selected={len(chosen)} disjoint openings excluded-screen={len(old_screen)} excluded-previous={len(old_deep)} original={len(rows)}",flush=True)
if __name__=="__main__":
    main()
