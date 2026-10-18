#!/usr/bin/env python3
"""Make a new Quoridor confirmation book excluding initial screening openings.

The original 25 paired openings were drawn from each source file with
random.Random(20261017).shuffle, as in run_benchmark._read_openings.
This selects 120 distinct positions from the remainder, with fixed new seed.
Referee validation is independently repeated by match_finalists/run_benchmark.
"""
import argparse
import json
import random
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--out",required=True)
    p.add_argument("--count",type=int,default=120)
    a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.source).read_text(encoding="utf-8").splitlines() if s.strip()]
    all_indices=list(range(len(rows)))
    old=all_indices.copy()
    random.Random(20261017).shuffle(old)
    excluded=set(old[:25])
    eligible=[idx for idx in all_indices if idx not in excluded]
    random.Random(20261027).shuffle(eligible)
    chosen=eligible[:a.count]
    assert len(chosen)==a.count and not (set(chosen)&excluded)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for idx in chosen:
            row=dict(rows[idx]);row["original_book_index"]=idx
            f.write(json.dumps(row,separators=(",",":"))+"\n")
    print(f"New independent book: {len(chosen)} openings from {len(rows)}; excluded all previous 25 pairs",flush=True)
if __name__=="__main__": main()
