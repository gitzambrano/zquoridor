#!/usr/bin/env python3
"""Assemble massive 200k+ dataset with action-Q policy sharpening and background mix.

Features:
1. 200,000+ rollout steps with temporal game-length discounting (gamma = 0.98).
2. 5,000 deep 1024-node search relabeled positions with action-Q policy sharpening.
3. 80,000 background selfplay positions to preserve general board play.
4. Heavy sample weighting (4.0x - 6.0x) on weakness/tactical data vs 1.0x background.
5. Strict group disjointness (zero train/val data leakage).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from build_teacher_soft import encode_states, load_positions


def sharpen_policy(policy: np.ndarray, action_q: np.ndarray, beta: float = 1.5) -> np.ndarray:
    """Sharpen policy visit counts using action Q values from deep search."""
    visited = policy > 0
    q_clean = np.where(visited, action_q, 0.0)
    # Subtract row maximum for numerical stability
    q_max = np.where(visited, q_clean, -1e9).max(axis=1, keepdims=True)
    exp_term = np.exp(beta * (q_clean - q_max))
    weighted = np.where(visited, policy * exp_term, 0.0)
    total = weighted.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return (weighted / total).astype(np.float32)


def assemble(
    rollouts_jsonl: Path,
    weakness_jsonl: Path,
    weakness_npz: Path,
    background_npz: Path,
    out_dir: Path,
    n_background: int = 80000,
    seed: int = 20260919,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_bin = ROOT / "bin" / "teacher_encode_state.exe"
    rng = np.random.default_rng(seed)

    # 1. Load and encode rollout steps
    print(f"Loading rollout steps from {rollouts_jsonl}...")
    with open(rollouts_jsonl, "r", encoding="utf-8") as fh:
        rollout_records = [json.loads(line) for line in fh if line.strip()]

    n_rollouts = len(rollout_records)
    print(f"Loaded {n_rollouts} rollout steps. Encoding states with {encoder_bin}...")
    rollout_enc = encode_states(rollout_records, encoder_bin)

    # Policy for rollouts is one-hot on best_action, value is discounted root_value
    rollout_policy = np.zeros((n_rollouts, 209), dtype=np.float32)
    rollout_value = np.zeros(n_rollouts, dtype=np.float32)
    rollout_groups = []

    for i, r in enumerate(rollout_records):
        rollout_policy[i, int(r["best_action"])] = 1.0
        rollout_value[i] = float(r["root_value"])
        raw_id = r["id"]
        seed_id = raw_id.split("_r")[0] if "_r" in raw_id else raw_id
        rollout_groups.append(f"rollout:{seed_id}")

    # Seed-based train/val split (15% val)
    unique_seeds = sorted(set(rollout_groups))
    rng.shuffle(unique_seeds)
    val_seed_set = set(unique_seeds[:int(len(unique_seeds) * 0.15)])
    rollout_is_val = np.asarray([g in val_seed_set for g in rollout_groups], dtype=bool)

    rollout_data = {
        "own_pawn": rollout_enc["own_pawn"],
        "opp_pawn": rollout_enc["opp_pawn"],
        "walls_h": rollout_enc["walls_h"],
        "walls_v": rollout_enc["walls_v"],
        "walls_left_own": rollout_enc["walls_left_own"],
        "walls_left_opp": rollout_enc["walls_left_opp"],
        "own_dist": rollout_enc["own_dist"],
        "opp_dist": rollout_enc["opp_dist"],
        "policy": rollout_policy,
        "value": rollout_value,
        "weight": np.full(n_rollouts, 4.0, dtype=np.float32),
        "is_val": rollout_is_val,
        "group_id": np.asarray(rollout_groups),
    }

    # 2. Load and sharpen deep search weakness positions (5k)
    print(f"Loading 5k deep search positions from {weakness_jsonl}...")
    weak_positions = load_positions(weakness_jsonl)
    weak_enc = encode_states(weak_positions, encoder_bin)

    with np.load(weakness_npz, allow_pickle=False) as targets:
        raw_weak_policy = targets["policy"].astype(np.float32)
        raw_weak_q = targets["action_q"].astype(np.float32)
        weak_value = targets["value"].astype(np.float32)
        stability = targets["budget_agreement"].astype(np.float32)

    print("Sharpening policy targets with action Q values...")
    sharpened_weak_policy = sharpen_policy(raw_weak_policy, raw_weak_q, beta=1.5)

    weak_groups = [f"weak_op_{r.get('opening_index', 0)}" for r in weak_positions]
    unique_weak_ops = sorted(set(weak_groups))
    rng.shuffle(unique_weak_ops)
    val_weak_set = set(unique_weak_ops[:int(len(unique_weak_ops) * 0.15)])
    weak_is_val = np.asarray([g in val_weak_set for g in weak_groups], dtype=bool)

    weak_data = {
        "own_pawn": weak_enc["own_pawn"],
        "opp_pawn": weak_enc["opp_pawn"],
        "walls_h": weak_enc["walls_h"],
        "walls_v": weak_enc["walls_v"],
        "walls_left_own": weak_enc["walls_left_own"],
        "walls_left_opp": weak_enc["walls_left_opp"],
        "own_dist": weak_enc["own_dist"],
        "opp_dist": weak_enc["opp_dist"],
        "policy": sharpened_weak_policy,
        "value": weak_value,
        "weight": np.clip(6.0 * stability, 2.0, 8.0).astype(np.float32),
        "is_val": weak_is_val,
        "group_id": np.asarray(weak_groups),
    }

    # 3. Load background self-play positions
    print(f"Loading background dataset from {background_npz} (sampling {n_background} positions)...")
    with np.load(background_npz, allow_pickle=False) as bg:
        bg_n = len(bg["value"])
        bg_idx = rng.choice(bg_n, size=min(n_background, bg_n), replace=False)
        bg_is_val = bg["is_val"][bg_idx].astype(bool)
        bg_groups = np.asarray([f"bg_{i}" for i in bg_idx])

        bg_data = {
            "own_pawn": bg["own_pawn"][bg_idx],
            "opp_pawn": bg["opp_pawn"][bg_idx],
            "walls_h": bg["walls_h"][bg_idx],
            "walls_v": bg["walls_v"][bg_idx],
            "walls_left_own": bg["walls_left_own"][bg_idx],
            "walls_left_opp": bg["walls_left_opp"][bg_idx],
            "own_dist": bg["own_dist"][bg_idx],
            "opp_dist": bg["opp_dist"][bg_idx],
            "policy": bg["policy"][bg_idx].astype(np.float32),
            "value": bg["value"][bg_idx].astype(np.float32),
            "weight": np.full(len(bg_idx), 1.0, dtype=np.float32),
            "is_val": bg_is_val,
            "group_id": bg_groups,
        }

    # 4. Concatenate all datasets
    combined = {}
    for key in rollout_data:
        combined[key] = np.concatenate([rollout_data[key], weak_data[key], bg_data[key]])

    total_samples = len(combined["value"])
    val_samples = int(combined["is_val"].sum())
    train_samples = total_samples - val_samples

    print(f"Combined total: {total_samples} samples ({train_samples} train, {val_samples} val).")

    # Verify no group overlap
    val_groups = set(combined["group_id"][combined["is_val"]])
    train_groups = set(combined["group_id"][~combined["is_val"]])
    overlap = val_groups & train_groups
    if overlap:
        raise ValueError(f"Group overlap detected: {len(overlap)} groups")

    out_file = out_dir / "dataset.npz"
    np.savez(out_file, **combined)
    print(f"Saved dataset to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "schema": "zquoridor.dataset.massive_weakness_200k.v1",
        "total_samples": total_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "rollout_samples": n_rollouts,
        "search_samples": len(weak_positions),
        "background_samples": len(bg_idx),
        "rollout_weight": 4.0,
        "search_weight_max": 8.0,
        "background_weight": 1.0,
        "seed": seed,
    }
    manifest_path = out_dir / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        assemble(
            rollouts_jsonl=ROOT / "data/teaching/massive_rollouts_raw.jsonl",
            weakness_jsonl=ROOT / "data/teaching/weakness_top5k.jsonl",
            weakness_npz=ROOT / "data/teaching/weakness_top5k_relabeled.npz",
            background_npz=ROOT / "data/teaching/policy-tactical-245k/dataset.npz",
            out_dir=ROOT / "data/teaching/massive-200k-sharpened",
            n_background=80000,
        )
    else:
        print("Usage: python assemble_200k_dataset.py --run")
