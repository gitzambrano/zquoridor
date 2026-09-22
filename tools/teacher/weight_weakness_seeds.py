#!/usr/bin/env python3
"""Build a 60/40 Claustrophobia/Titanium seed schedule with crisis weighting."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def crisis_weight(row: dict) -> float:
    reasons = set(row.get("metadata", {}).get("reasons", []))
    weight = 1.0
    if "swept_opening" in reasons:
        weight *= 2.5
    if "zq_turn_in_loss" in reasons:
        weight *= 1.5
    if int(row.get("ply", 999)) <= 20:
        weight *= 1.5
    return weight


def sample_pool(rng: random.Random, rows: list[dict], count: int) -> list[dict]:
    if not rows:
        raise ValueError("cannot sample an empty opponent pool")
    return rng.choices(rows, weights=[crisis_weight(row) for row in rows], k=count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--claustro-share", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_opponent = {"claustrophobia": [], "titanium": []}
    for row in rows:
        opponent = str(row.get("metadata", {}).get("opponent", "")).lower()
        if opponent in by_opponent:
            by_opponent[opponent].append(row)

    rng = random.Random(args.seed)
    claustro_count = round(args.count * args.claustro_share)
    selected = sample_pool(rng, by_opponent["claustrophobia"], claustro_count)
    selected += sample_pool(rng, by_opponent["titanium"], args.count - claustro_count)
    rng.shuffle(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for index, original in enumerate(selected):
            row = dict(original)
            row["id"] = f"{original['id']}-w{index:05d}"
            row["source"] = "weighted-weakness-selfplay-seed"
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "zquoridor.weighted_weakness_seeds.v1",
        "source": str(args.input.resolve()),
        "output": str(args.out.resolve()),
        "total": len(selected),
        "claustrophobia": claustro_count,
        "titanium": len(selected) - claustro_count,
        "weighting": "swept_opening=2.5x,zq_turn_in_loss=1.5x,ply<=20=1.5x",
        "seed": args.seed,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
