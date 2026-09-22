#!/usr/bin/env python3
"""Prepare 3,500+ diverse weakness seeds for massive rollout generation.

Combines:
1. Center-rush tactical suite prefixes and branches (openings 0..32, plies 6..12).
2. Mined loss and swept opening positions vs Claustrophobia (plies 6..45).
3. Mined loss and swept opening positions vs Titanium (plies 6..45).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Edit this block for any compatible weakness and opening sources.
CONFIG = {
    "mined_file": str(ROOT / "data/teaching/mined_weaknesses_all.jsonl"),
    "center_rush_file": str(ROOT / "tools/external/openings_center_rush_v1.jsonl"),
    "out_file": str(ROOT / "data/teaching/massive_seeds_3500.jsonl"),
    "seed": 20260919,
    "max_seeds": 3500,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-file", default=CONFIG["mined_file"])
    parser.add_argument("--center-rush-file", default=CONFIG["center_rush_file"])
    parser.add_argument("--out-file", default=CONFIG["out_file"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--max-seeds", type=int, default=CONFIG["max_seeds"])
    args = parser.parse_args(argv)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mined_file = Path(args.mined_file)
    cr_file = Path(args.center_rush_file)

    seeds = []
    seen_histories = set()

    # 1. Center rush openings: generate prefix steps from each opening
    with open(cr_file, "r", encoding="utf-8") as fh:
        cr_openings = [json.loads(line) for line in fh if line.strip()]

    print(f"Loaded {len(cr_openings)} center rush openings.")
    for op in cr_openings:
        moves = op["moves"]
        op_idx = op["opening_index"]
        # Seed at the base rush (ply 6) and each subsequent step (plies 7, 8, 9)
        for end_ply in range(6, len(moves) + 1):
            sub_history = moves[:end_ply]
            key = tuple(sub_history)
            if key not in seen_histories:
                seen_histories.add(key)
                seeds.append({
                    "schema": "zquoridor.position.v1",
                    "id": f"cr_op{op_idx}_p{end_ply}",
                    "history": sub_history,
                    "side_to_move": end_ply % 2,
                    "split": "train" if op_idx % 6 != 0 else "val",
                    "source": "center_rush_suite",
                    "opening_index": op_idx,
                    "ply": end_ply,
                    "metadata": {"category": op.get("category", "unknown"), "reasons": ["center_rush_opening"]},
                })

    print(f"Added {len(seeds)} center-rush seeds.")

    # 2. Opponent loss crisis points
    print(f"Loading mined weaknesses from {mined_file}...")
    with open(mined_file, "r", encoding="utf-8") as fh:
        mined_rows = [json.loads(line) for line in fh if line.strip()]

    # Filter for opening/early midgame crisis (ply 6..45) where ZQ was under pressure
    claustro_candidates = []
    titanium_candidates = []

    for r in mined_rows:
        ply = r.get("ply", 0)
        if not (6 <= ply <= 45):
            continue
        reasons = r.get("metadata", {}).get("reasons", [])
        if "zq_turn_in_loss" not in reasons and "swept_opening" not in reasons:
            continue
        opp = r.get("metadata", {}).get("opponent")
        if opp == "claustrophobia":
            claustro_candidates.append(r)
        elif opp == "titanium":
            titanium_candidates.append(r)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(claustro_candidates)
    rng.shuffle(titanium_candidates)

    target_claustro = args.max_seeds // 2
    target_titanium = args.max_seeds - target_claustro

    added_cl = 0
    for r in claustro_candidates:
        key = tuple(r["history"])
        if key not in seen_histories:
            seen_histories.add(key)
            seeds.append(r)
            added_cl += 1
            if added_cl >= target_claustro:
                break

    added_ti = 0
    for r in titanium_candidates:
        key = tuple(r["history"])
        if key not in seen_histories:
            seen_histories.add(key)
            seeds.append(r)
            added_ti += 1
            if added_ti >= target_titanium:
                break

    print(f"Added {added_cl} Claustrophobia loss seeds and {added_ti} Titanium loss seeds.")
    seeds = seeds[:args.max_seeds]
    print(f"Total unique seeds assembled: {len(seeds)}")

    with open(out_file, "w", encoding="utf-8") as fh:
        for s in seeds:
            fh.write(json.dumps(s) + "\n")

    print(f"Saved {len(seeds)} seeds to {out_file}")


if __name__ == "__main__":
    main()
