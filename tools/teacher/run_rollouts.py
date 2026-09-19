#!/usr/bin/env python3
"""Run multithreaded branching rollouts from crisis positions and build dataset.npz."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from build_teacher_soft import encode_states

CONFIG = {
    "positions": str(ROOT / "data/teaching/weakness-mining-35k/positions.jsonl"),
    "nnue": str(ROOT / "results/experiments/race512-search10-ft-s20260917/student_int8.bin"),
    "out_dir": str(ROOT / "data/teaching/rollouts-500-seed"),
    "max_seeds": 500,
    "rollouts": 8,
    "nodes": 128,
    "explore_plies": 4,
    "top_k": 4,
    "gamma": 0.98,
    "max_plies": 100,
    "threads": 12,
    "seed": 20260919,
    "val_fraction": 0.15,
    "sample_weight": 6.0,
}


def filter_crisis_seeds(positions_file: Path, max_seeds: int) -> list[dict]:
    """Select top crisis seeds favoring losses against Claustrophobia."""
    with open(positions_file, "r", encoding="utf-8") as fh:
        all_positions = [json.loads(line) for line in fh if line.strip()]

    # Prioritize: claustro losses first, then wall asymmetry
    claustro_loss = []
    wall_asymmetry = []
    other = []

    for row in all_positions:
        tags = set(row.get("tags", []))
        if "loss_vs_claustrophobia" in tags:
            claustro_loss.append(row)
        elif "wall_asymmetry" in tags:
            wall_asymmetry.append(row)
        else:
            other.append(row)

    rng = np.random.default_rng(20260919)
    rng.shuffle(claustro_loss)
    rng.shuffle(wall_asymmetry)
    rng.shuffle(other)

    selected = claustro_loss[:max_seeds]
    if len(selected) < max_seeds:
        selected.extend(wall_asymmetry[:max_seeds - len(selected)])
    if len(selected) < max_seeds:
        selected.extend(other[:max_seeds - len(selected)])

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=Path(CONFIG["positions"]))
    parser.add_argument("--nnue", type=Path, default=Path(CONFIG["nnue"]))
    parser.add_argument("--out-dir", type=Path, default=Path(CONFIG["out_dir"]))
    parser.add_argument("--max-seeds", type=int, default=CONFIG["max_seeds"])
    parser.add_argument("--rollouts", type=int, default=CONFIG["rollouts"])
    parser.add_argument("--nodes", type=int, default=CONFIG["nodes"])
    parser.add_argument("--explore-plies", type=int, default=CONFIG["explore_plies"])
    parser.add_argument("--top-k", type=int, default=CONFIG["top_k"])
    parser.add_argument("--gamma", type=float, default=CONFIG["gamma"])
    parser.add_argument("--max-plies", type=int, default=CONFIG["max_plies"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--val-fraction", type=float, default=CONFIG["val_fraction"])
    parser.add_argument("--sample-weight", type=float, default=CONFIG["sample_weight"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds_file = args.out_dir / "seeds.jsonl"
    raw_rollouts_file = args.out_dir / "raw_rollouts.jsonl"
    out_dataset = args.out_dir / "dataset.npz"

    # Step 1: Select seeds
    seeds = filter_crisis_seeds(args.positions, args.max_seeds)
    print(f"Selected {len(seeds)} crisis seeds from {args.positions}", flush=True)
    with open(seeds_file, "w", encoding="utf-8") as fh:
        for s in seeds:
            fh.write(json.dumps(s) + "\n")

    # Step 2: Run C++ rollout generator
    gen_bin = ROOT / "bin" / "generate_rollouts.exe"
    cmd = [
        str(gen_bin),
        "--positions", str(seeds_file),
        "--nnue", str(args.nnue),
        "--out", str(raw_rollouts_file),
        "--rollouts", str(args.rollouts),
        "--nodes", str(args.nodes),
        "--explore-plies", str(args.explore_plies),
        "--top-k", str(args.top_k),
        "--gamma", str(args.gamma),
        "--max-plies", str(args.max_plies),
        "--threads", str(args.threads),
        "--seed", str(args.seed),
    ]
    print(f"Running rollout generator: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # Step 3: Load rollout steps
    with open(raw_rollouts_file, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    if not records:
        raise ValueError("rollout generator produced zero records")
    print(f"Generated {len(records)} total rollout steps", flush=True)

    # Step 4: Encode states via C++ state encoder
    encoder_bin = ROOT / "bin" / "teacher_encode_state.exe"
    print(f"Encoding {len(records)} states via {encoder_bin}...", flush=True)
    encoded = encode_states(records, encoder_bin)

    # Step 5: Build policy, value, weight, split
    n = len(records)
    policy = np.zeros((n, 209), dtype=np.float16)
    value = np.zeros(n, dtype=np.float16)
    seed_ids = []

    for i, r in enumerate(records):
        action = int(r["best_action"])
        policy[i, action] = 1.0
        value[i] = float(r["root_value"])
        # Extract seed id (strip _rX_pY)
        raw_id = r["id"]
        seed_id = raw_id.split("_r")[0] if "_r" in raw_id else raw_id
        seed_ids.append(seed_id)

    # Group-based split to avoid data leakage
    unique_seeds = sorted(set(seed_ids))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_seeds)
    n_val_seeds = max(1, int(len(unique_seeds) * args.val_fraction))
    val_seed_set = set(unique_seeds[:n_val_seeds])
    is_val = np.array([sid in val_seed_set for sid in seed_ids], dtype=bool)

    dataset_arrays = {
        "own_pawn": encoded["own_pawn"],
        "opp_pawn": encoded["opp_pawn"],
        "walls_h": encoded["walls_h"],
        "walls_v": encoded["walls_v"],
        "walls_left_own": encoded["walls_left_own"],
        "walls_left_opp": encoded["walls_left_opp"],
        "own_dist": encoded["own_dist"],
        "opp_dist": encoded["opp_dist"],
        "mover": encoded["mover"],
        "policy": policy,
        "value": value,
        "weight": np.full(n, args.sample_weight, dtype=np.float32),
        "is_val": is_val,
        "group_id": np.array([f"rollout:{sid}".encode("utf-8") for sid in seed_ids]),
    }

    np.savez(out_dataset, **dataset_arrays)
    manifest = {
        "schema": "zquoridor.rollouts.v1",
        "seeds": len(seeds),
        "rollout_steps": n,
        "train_samples": int((~is_val).sum()),
        "val_samples": int(is_val.sum()),
        "gamma": args.gamma,
        "explore_plies": args.explore_plies,
        "sample_weight": args.sample_weight,
        "out": str(out_dataset),
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
