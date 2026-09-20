#!/usr/bin/env python3
"""Assemble fine-tuning dataset for weakness recovery and center-rush tactics.

Combines:
1. 5,000 top hard weakness positions (from Claustrophobia & Titanium loss games,
   including 415 center-rush positions), relabeled with 1024-node MCTS search.
2. Background tactical & selfplay positions (from policy-tactical-245k), ensuring
   the network preserves its general strength and evaluation without catastrophic forgetting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from build_teacher_soft import encode_states, load_positions


def build_weakness_dataset(
    weakness_jsonl: Path,
    weakness_npz: Path,
    background_npz: Path,
    out_dir: Path,
    n_background: int = 60000,
    weakness_weight_scale: float = 3.5,
    seed: int = 20260919,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_bin = ROOT / "bin" / "teacher_encode_state.exe"
    if not encoder_bin.is_file():
        raise FileNotFoundError(f"Missing {encoder_bin}")

    print(f"Loading positions from {weakness_jsonl}...")
    positions = load_positions(weakness_jsonl)
    n_weak = len(positions)
    print(f"Loaded {n_weak} positions. Encoding states with {encoder_bin}...")
    encoded = encode_states(positions, encoder_bin)

    print(f"Loading search targets from {weakness_npz}...")
    with np.load(weakness_npz, allow_pickle=False) as targets:
        weak_policy = targets["policy"].astype(np.float32)
        weak_value = targets["value"].astype(np.float32)
        stability = targets["budget_agreement"].astype(np.float32)

    # Normalize policies just in case
    pol_sum = weak_policy.sum(axis=1, keepdims=True)
    pol_sum = np.where(pol_sum > 0, pol_sum, 1.0)
    weak_policy /= pol_sum

    # Compute weights for weakness positions: higher weight for stable search results
    weak_weight = np.clip(weakness_weight_scale * stability, 1.0, 6.0).astype(np.float32)

    # Clean train/val split based on opening index (zero leakage)
    rng = np.random.default_rng(seed)
    unique_openings = sorted({r.get("opening_index", 0) for r in positions})
    rng.shuffle(unique_openings)
    n_val_openings = max(1, int(len(unique_openings) * 0.15))
    val_openings_set = set(unique_openings[:n_val_openings])

    weak_is_val = np.asarray([r.get("opening_index", 0) in val_openings_set for r in positions], dtype=bool)
    weak_group_id = np.asarray([f"weak_op_{r.get('opening_index', 0)}" for r in positions])

    weak_data = {
        "own_pawn": encoded["own_pawn"],
        "opp_pawn": encoded["opp_pawn"],
        "walls_h": encoded["walls_h"],
        "walls_v": encoded["walls_v"],
        "walls_left_own": encoded["walls_left_own"],
        "walls_left_opp": encoded["walls_left_opp"],
        "own_dist": encoded["own_dist"],
        "opp_dist": encoded["opp_dist"],
        "policy": weak_policy,
        "value": weak_value,
        "weight": weak_weight,
        "is_val": weak_is_val,
        "group_id": weak_group_id,
    }

    print(f"Weakness dataset: {n_weak} positions ({weak_is_val.sum()} val, {(~weak_is_val).sum()} train).")

    # Now load background data
    print(f"Loading background dataset from {background_npz} (sampling {n_background} positions)...")
    with np.load(background_npz, allow_pickle=False) as bg:
        bg_n = len(bg["value"])
        if n_background < bg_n:
            bg_indices = rng.choice(bg_n, size=n_background, replace=False)
        else:
            bg_indices = np.arange(bg_n)

        bg_is_val = bg["is_val"][bg_indices].astype(bool)
        bg_group_id = np.asarray([f"bg_{i}" for i in bg_indices])

        bg_data = {
            "own_pawn": bg["own_pawn"][bg_indices],
            "opp_pawn": bg["opp_pawn"][bg_indices],
            "walls_h": bg["walls_h"][bg_indices],
            "walls_v": bg["walls_v"][bg_indices],
            "walls_left_own": bg["walls_left_own"][bg_indices],
            "walls_left_opp": bg["walls_left_opp"][bg_indices],
            "own_dist": bg["own_dist"][bg_indices],
            "opp_dist": bg["opp_dist"][bg_indices],
            "policy": bg["policy"][bg_indices].astype(np.float32),
            "value": bg["value"][bg_indices].astype(np.float32),
            "weight": np.full(len(bg_indices), 1.0, dtype=np.float32),
            "is_val": bg_is_val,
            "group_id": bg_group_id,
        }

    # Concatenate
    combined = {}
    for key in weak_data:
        combined[key] = np.concatenate([weak_data[key], bg_data[key]])

    total_samples = len(combined["value"])
    val_samples = int(combined["is_val"].sum())
    train_samples = total_samples - val_samples

    print(f"Combined total: {total_samples} samples ({train_samples} train, {val_samples} val).")

    # Verify group disjointness
    val_groups = set(combined["group_id"][combined["is_val"]])
    train_groups = set(combined["group_id"][~combined["is_val"]])
    overlap = val_groups & train_groups
    if overlap:
        raise ValueError(f"Group overlap detected: {len(overlap)} groups")

    out_file = out_dir / "dataset.npz"
    np.savez(out_file, **combined)
    print(f"Saved dataset to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "schema": "zquoridor.dataset.weakness_fine_tuning.v1",
        "total_samples": total_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "weakness_samples": n_weak,
        "background_samples": len(bg_indices),
        "weakness_weight_scale": weakness_weight_scale,
        "seed": seed,
    }
    manifest_path = out_dir / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    build_weakness_dataset(
        weakness_jsonl=ROOT / "data/teaching/weakness_top5k.jsonl",
        weakness_npz=ROOT / "data/teaching/weakness_top5k_relabeled.npz",
        background_npz=ROOT / "data/teaching/policy-tactical-245k/dataset.npz",
        out_dir=ROOT / "data/teaching/weakness-fine-tuning-champion",
        n_background=75000,
        weakness_weight_scale=3.5,
    )
