#!/usr/bin/env python3
"""Produce deterministic, nonoverlapping opening cohorts for parallel paired H2H.

Selection shuffles original opening indices once per book/seed and partitions
into disjoint consecutive shards. This avoids overlapping openings between jobs
and keeps color-swapped pairing inside each shard. Does not invent positions.
"""
import argparse
import json
import random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261015)
    a = p.parse_args()
    lines = [x for x in Path(a.source).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not (0 <= a.shard < a.shards) or a.pairs * a.shards > len(lines):
        raise ValueError(f"invalid shard or insufficient distinct openings: {len(lines)}")
    order = list(range(len(lines)))
    random.Random(a.seed).shuffle(order)
    subset = order[a.shard * a.pairs:(a.shard + 1) * a.pairs]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in subset:
            row = json.loads(lines[i])
            row["source_opening_index"] = i
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Opening shard {a.shard + 1}/{a.shards}: {len(subset)} unique source openings, seed={a.seed}", flush=True)

if __name__ == "__main__":
    main()
