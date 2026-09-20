#!/usr/bin/env python3
"""Calibrate and boost sample weights specifically targeting Claustrophobia weaknesses.

Targets:
1. Center-Rush frontal clashes with active wall stocks (both pawns in cols 2..6, rows 2..6, >=4 walls).
2. Reed rear-wall encirclements and bottleneck corridor states (own_exits <= 2 with opponent nearby).
3. Mover race disadvantage in center collisions (own_dist > opp_dist with high wall stocks).
4. Retains Tier 1 background replay at weight 1.0 as regularizing anchor against catastrophic forgetting.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from student_model import _make_edge_tables

_ORTH_NEIGHBORS, _EDGE_H_MASKS, _EDGE_V_MASKS = _make_edge_tables()


def compute_exits(pawn: np.ndarray, wh: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """Compute number of unblocked directional exits (1..4) for given pawn cells."""
    p = pawn.astype(np.int64)
    h = wh.astype(np.uint64)
    v = wv.astype(np.uint64)
    fwd = (_ORTH_NEIGHBORS[p, 1] >= 0) & (((h & _EDGE_H_MASKS[p, 1]) | (v & _EDGE_V_MASKS[p, 1])) == 0)
    bwd = (_ORTH_NEIGHBORS[p, 0] >= 0) & (((h & _EDGE_H_MASKS[p, 0]) | (v & _EDGE_V_MASKS[p, 0])) == 0)
    lft = (_ORTH_NEIGHBORS[p, 2] >= 0) & (((h & _EDGE_H_MASKS[p, 2]) | (v & _EDGE_V_MASKS[p, 2])) == 0)
    rgt = (_ORTH_NEIGHBORS[p, 3] >= 0) & (((h & _EDGE_H_MASKS[p, 3]) | (v & _EDGE_V_MASKS[p, 3])) == 0)
    return fwd.astype(np.int32) + bwd.astype(np.int32) + lft.astype(np.int32) + rgt.astype(np.int32)


def calibrate_weights(in_path: Path, out_path: Path, max_weight: float = 30.0):
    print(f"Loading master dataset from {in_path}...", flush=True)
    with np.load(in_path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}

    n = len(data["value"])
    weights = data["weight"].astype(np.float32)
    own_pawn = data["own_pawn"].astype(np.int64)
    opp_pawn = data["opp_pawn"].astype(np.int64)
    ow = data["walls_left_own"].astype(np.int64)
    pw = data["walls_left_opp"].astype(np.int64)
    od = data["own_dist"].astype(np.int64)
    pd = data["opp_dist"].astype(np.int64)

    # 1. Center Rush Geometry
    own_r, own_c = own_pawn // 9, own_pawn % 9
    opp_r, opp_c = opp_pawn // 9, opp_pawn % 9
    dr = np.abs(opp_r - own_r)
    dc = np.abs(opp_c - own_c)
    manhattan = dr + dc

    in_center = (own_c >= 2) & (own_c <= 6) & (opp_c >= 2) & (opp_c <= 6) & \
                (own_r >= 2) & (own_r <= 6) & (opp_r >= 2) & (opp_r <= 6)
    active_walls = (ow >= 4) & (pw >= 4)
    cr_crisis = in_center & active_walls

    # 2. Exit Bottlenecks / Reed Rear-Wall
    print("Computing unblocked directional exit counts...", flush=True)
    chunk_size = 500_000
    exits = np.zeros(n, dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        exits[start:end] = compute_exits(own_pawn[start:end], data["walls_h"][start:end], data["walls_v"][start:end])

    # Encirclement: 1 or 2 exits, nearby opponent, active walls
    bottleneck_crisis = (exits <= 2) & (manhattan <= 3) & ((ow >= 3) | (pw >= 3))
    severe_bottleneck = (exits == 1) & ((ow >= 2) | (pw >= 2))

    # 3. Disadvantaged Center Clash
    race_behind_in_center = cr_crisis & (od > pd)

    # Boost factors
    multipliers = np.ones(n, dtype=np.float32)

    # Apply multiplicative boosts
    # Center-rush crisis: boost by 2.0x (or set min 6.0)
    cr_boost = np.where(cr_crisis, np.maximum(2.0, 6.0 / np.maximum(weights, 1.0)), 1.0)
    multipliers *= cr_boost

    # Bottleneck / Rear-wall encirclement: boost by 2.5x (or set min 8.0)
    bottle_boost = np.where(bottleneck_crisis, np.maximum(2.5, 8.0 / np.maximum(weights, 1.0)), 1.0)
    multipliers *= bottle_boost

    # Severe 1-exit bottleneck: boost by 3.0x (or set min 12.0)
    severe_boost = np.where(severe_bottleneck, np.maximum(3.0, 12.0 / np.maximum(weights, 1.0)), 1.0)
    multipliers *= severe_boost

    # Disadvantaged race in center: extra 1.5x
    disadv_boost = np.where(race_behind_in_center, 1.5, 1.0)
    multipliers *= disadv_boost

    # Calibrate final weights
    new_weights = np.clip(weights * multipliers, 1.0, max_weight).astype(np.float32)

    # Protect pure Tier 1 background anchor: positions with 4 exits and no crisis keep weight 1.0
    pure_open = (exits == 4) & ~cr_crisis & (weights <= 1.01)
    new_weights[pure_open] = 1.0

    print(f"Original weights: min={weights.min():.2f}, mean={weights.mean():.2f}, max={weights.max():.2f}")
    print(f"Boosted weights:  min={new_weights.min():.2f}, mean={new_weights.mean():.2f}, max={new_weights.max():.2f}")
    print(f"CR Crisis count: {cr_crisis.sum()} (avg weight: {new_weights[cr_crisis].mean():.2f})")
    print(f"Bottleneck count: {bottleneck_crisis.sum()} (avg weight: {new_weights[bottleneck_crisis].mean():.2f})")
    print(f"Severe Bottleneck count: {severe_bottleneck.sum()} (avg weight: {new_weights[severe_bottleneck].mean():.2f})")
    print(f"High-weight samples (>= 8.0): {(new_weights >= 8.0).sum()}")
    print(f"Anchor samples (== 1.0): {(new_weights == 1.0).sum()}")

    data["weight"] = new_weights
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out_path.with_suffix(".tmp.npz")
    print(f"Saving calibrated dataset to {out_path}...", flush=True)
    np.savez_compressed(temp_out, **data)
    temp_out.replace(out_path)
    print(f"Calibrated dataset written successfully ({out_path.stat().st_size / (1024**3):.2f} GB).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-data", type=Path, default=ROOT / "data/teaching/multipath-master-11m/dataset.npz")
    parser.add_argument("--out-data", type=Path, default=ROOT / "data/teaching/multipath-weakness-boosted-11m/dataset.npz")
    parser.add_argument("--max-weight", type=float, default=30.0)
    args = parser.parse_args()

    calibrate_weights(args.in_data, args.out_data, args.max_weight)


if __name__ == "__main__":
    main()
