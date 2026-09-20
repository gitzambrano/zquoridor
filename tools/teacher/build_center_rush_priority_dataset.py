#!/usr/bin/env python3
"""Build Center-Rush Focused Priority Teaching Dataset.

Mines and extracts:
1. 200,000 sharp Center-Rush positions from the 10.8M Claustrophobia & ZQ teaching corpus.
   Assigned HIGH weight (5.0x) so center-rush represents >90% of the training loss.
2. 5,000 deep 1024-node MCTS search relabeled weakness positions from Claustrophobia
   and Titanium games, with Action-Q policy sharpening (weight 6.0x).
3. 50,000 non-center-rush background positions (weight 1.0x) to anchor general board
   play and endgame race calculations without catastrophic forgetting.
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
    q_max = np.where(visited, q_clean, -1e9).max(axis=1, keepdims=True)
    exp_term = np.exp(beta * (q_clean - q_max))
    weighted = np.where(visited, policy * exp_term, 0.0)
    total = weighted.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return (weighted / total).astype(np.float32)


def build_center_rush_dataset(
    source_npz: Path,
    weakness_jsonl: Path,
    weakness_npz: Path,
    out_dir: Path,
    n_center_rush: int = 200000,
    n_background: int = 50000,
    cr_weight: float = 5.0,
    seed: int = 20260919,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    encoder_bin = ROOT / "bin" / "teacher_encode_state.exe"

    print(f"Opening source dataset via mmap from {source_npz}...")
    source = np.load(source_npz, mmap_mode="r")
    total_source = len(source["value"])
    print(f"Total positions in source: {total_source}")

    # 1. Identify Center-Rush candidates
    print("Identifying Center-Rush positions in source...")
    own = source["own_pawn"][:]
    opp = source["opp_pawn"][:]
    w_own = source["walls_left_own"][:]
    w_opp = source["walls_left_opp"][:]

    # Center-rush confrontation criteria:
    # Files c, d, e, f, g (columns 2..6), rows 2..5 for own pawn, rows 3..6 for opp pawn
    # With active opening/early-midgame wall stock (>= 5 walls each)
    d_cols = {2, 3, 4, 5, 6}
    own_c = np.isin(own % 9, list(d_cols)) & (own // 9 >= 2) & (own // 9 <= 5)
    opp_c = np.isin(opp % 9, list(d_cols)) & (opp // 9 >= 3) & (opp // 9 <= 6)
    walls_opening = (w_own >= 5) & (w_opp >= 5)

    cr_mask = own_c & opp_c & walls_opening
    cr_indices = np.flatnonzero(cr_mask)
    bg_indices = np.flatnonzero(~cr_mask)

    print(f"Found {len(cr_indices)} center-rush candidates and {len(bg_indices)} background candidates.")

    if len(cr_indices) < n_center_rush:
        raise ValueError(f"Requested {n_center_rush} center rush, but only {len(cr_indices)} available")

    # Sample exactly n_center_rush and n_background
    selected_cr = np.sort(rng.choice(cr_indices, size=n_center_rush, replace=False))
    selected_bg = np.sort(rng.choice(bg_indices, size=min(n_background, len(bg_indices)), replace=False))

    print(f"Selected {len(selected_cr)} center-rush positions and {len(selected_bg)} background positions.")

    # 2. Extract arrays in chunks to optimize RAM and speed
    fields = [
        "own_pawn", "opp_pawn", "walls_h", "walls_v",
        "own_dist", "opp_dist", "walls_left_own", "walls_left_opp",
        "policy", "value"
    ]

    print("Extracting center-rush arrays from source...")
    cr_data = {f: source[f][selected_cr] for f in fields}
    cr_data["weight"] = np.full(len(selected_cr), cr_weight, dtype=np.float32)

    # Clean group IDs for center rush to avoid train/val leakage
    # Group by (own_pawn, opp_pawn, walls_left_own, walls_left_opp)
    cr_pawn_combos = (
        cr_data["own_pawn"].astype(np.int64) * 81 + cr_data["opp_pawn"].astype(np.int64)
    )
    unique_combos = np.unique(cr_pawn_combos)
    rng.shuffle(unique_combos)
    val_combo_set = set(unique_combos[:int(len(unique_combos) * 0.15)])
    cr_data["is_val"] = np.isin(cr_pawn_combos, list(val_combo_set))
    cr_data["group_id"] = np.asarray([f"cr_combo_{c}" for c in cr_pawn_combos])

    print("Extracting background arrays from source...")
    bg_data = {f: source[f][selected_bg] for f in fields}
    bg_data["weight"] = np.full(len(selected_bg), 1.0, dtype=np.float32)
    bg_pawn_combos = (
        bg_data["own_pawn"].astype(np.int64) * 81 + bg_data["opp_pawn"].astype(np.int64)
    )
    bg_data["is_val"] = np.isin(bg_pawn_combos, list(val_combo_set))
    bg_data["group_id"] = np.asarray([f"bg_combo_{c}" for c in bg_pawn_combos])

    # 3. Load 5,000 deep search weakness positions with Action-Q sharpening
    print(f"Loading 5,000 deep search weakness positions from {weakness_jsonl}...")
    weak_pos = load_positions(weakness_jsonl)
    weak_enc = encode_states(weak_pos, encoder_bin)

    with np.load(weakness_npz, allow_pickle=False) as targets:
        weak_pol_raw = targets["policy"].astype(np.float32)
        weak_q = targets["action_q"].astype(np.float32)
        weak_val = targets["value"].astype(np.float32)
        weak_stab = targets["budget_agreement"].astype(np.float32)

    print("Sharpening 5k policy targets with Action-Q values...")
    weak_pol_sharp = sharpen_policy(weak_pol_raw, weak_q, beta=1.5)

    weak_groups = [f"weak_op_{r.get('opening_index', 0)}" for r in weak_pos]
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
        "policy": weak_pol_sharp,
        "value": weak_val,
        "weight": np.clip(6.0 * weak_stab, 2.0, 8.0).astype(np.float32),
        "is_val": weak_is_val,
        "group_id": np.asarray(weak_groups),
    }

    # 4. Concatenate all datasets
    all_fields = fields + ["weight", "is_val", "group_id"]
    combined = {}
    for f in all_fields:
        combined[f] = np.concatenate([cr_data[f], weak_data[f], bg_data[f]])

    total_samples = len(combined["value"])
    val_samples = int(combined["is_val"].sum())
    train_samples = total_samples - val_samples

    # Verify train/val group disjointness
    val_groups = set(combined["group_id"][combined["is_val"]])
    train_groups = set(combined["group_id"][~combined["is_val"]])
    overlap = val_groups & train_groups
    if overlap:
        raise ValueError(f"Train/validation group overlap detected: {len(overlap)} groups")

    out_file = out_dir / "dataset.npz"
    print(f"Saving combined dataset ({total_samples} samples: {train_samples} train, {val_samples} val) to {out_file}...")
    np.savez(out_file, **combined)
    print(f"Successfully saved to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "schema": "zquoridor.dataset.center_rush_priority.v1",
        "total_samples": total_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "center_rush_samples": len(selected_cr),
        "deep_search_weakness_samples": len(weak_pos),
        "background_samples": len(selected_bg),
        "center_rush_weight": cr_weight,
        "deep_search_weight_max": 8.0,
        "background_weight": 1.0,
        "effective_center_rush_share_pct": float(
            (len(selected_cr) * cr_weight + len(weak_pos) * 6.0)
            / (len(selected_cr) * cr_weight + len(weak_pos) * 6.0 + len(selected_bg) * 1.0)
            * 100.0
        ),
        "seed": seed,
    }
    manifest_path = out_dir / "dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    print(f"Effective Center-Rush signal share: {manifest['effective_center_rush_share_pct']:.2f}%")


if __name__ == "__main__":
    build_center_rush_dataset(
        source_npz=ROOT / "data/teaching/multitier-final-dataset/dataset.npz",
        weakness_jsonl=ROOT / "data/teaching/weakness_top5k.jsonl",
        weakness_npz=ROOT / "data/teaching/weakness_top5k_relabeled.npz",
        out_dir=ROOT / "data/teaching/center-rush-200k-priority",
        n_center_rush=200000,
        n_background=50000,
        cr_weight=5.0,
    )
