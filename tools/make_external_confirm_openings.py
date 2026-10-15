#!/usr/bin/env python3
"""Independent, paired opening book excluding the previous Claustrophobia sample.

Match run_benchmark._read_openings selection exactly: shuffle original line
indices using the historical seed. Exclude the 25 opening pairs used in
the prior B/main-vs-Claustrophobia run, then draw 100 new pairs using a
different seed. Both versions and both external opponents use identical
openings and color swaps. No synthetic game states.
"""
import argparse
import json
import random
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--pairs",type=int,default=100)
    p.add_argument("--exclude-seed",type=int,default=20261015)
    p.add_argument("--exclude-pairs",type=int,default=25)
    p.add_argument("--seed",type=int,default=20261016)
    a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.source).read_text(encoding="utf-8").splitlines() if s.strip()]
    old=list(range(len(rows)))
    random.Random(a.exclude_seed).shuffle(old)
    excluded=set(old[:a.exclude_pairs])
    candidates=[i for i in range(len(rows)) if i not in excluded]
    random.Random(a.seed).shuffle(candidates)
    chosen=candidates[:a.pairs]
    if len(chosen)!=a.pairs or len(set(chosen))!=a.pairs or any(i in excluded for i in chosen):
        raise AssertionError("Invalid independent paired opening selection")
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",encoding="utf-8") as f:
        for idx in chosen:
            row=dict(rows[idx])
            row["original_opening_index"]=idx
            f.write(json.dumps(row,separators=(",",":"))+"\n")
    print(f"Independent openings: {a.pairs} chosen / {len(rows)}; "
          f"{a.exclude_pairs} historical positions excluded, selection seed={a.seed}.",flush=True)

if __name__=="__main__":
    main()
