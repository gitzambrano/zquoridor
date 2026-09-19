#!/usr/bin/env python3
"""Mine 100,000 generic critical positions from self-play for Tier 3.5 curriculum.

Selects critical states directly from the generic canonical self-play corpus
using vectorized evaluation across four dimensions:
1. High policy entropy (competitive top moves, high Shannon entropy)
2. Race and value uncertainty (|V| near 0, close BFS distance margin)
3. Critical wall dynamics (wall depletion <= 3, high CAT corridor pressure, maze density)
4. Game turning swings (evaluation vs outcome divergence)

Outputs architecture-neutral zquoridor.position.v1 JSONL with canonical @state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from read_selfplay import _detect_format, SAMPLE_DTYPE, SAMPLE_DTYPE_V2


def mine_tier3_5_critical(
    selfplay_dir: Path,
    out_file: Path,
    target_count: int = 100000,
    val_fraction: float = 0.15,
    seed: int = 20260920,
) -> dict:
    rng = np.random.default_rng(seed)
    shards = sorted(selfplay_dir.rglob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"no selfplay shards in {selfplay_dir}")

    rng.shuffle(shards)
    print(f"Mining {target_count:,} generic critical positions from {len(shards)} shards...", flush=True)

    seen_keys = set()
    selected_records = []
    quota_per_shard = max(200, math.ceil(target_count * 1.5 / len(shards)))

    for s_idx, shard_path in enumerate(shards):
        try:
            dtype, size = _detect_format(str(shard_path))
            if size not in (32, 64):
                continue
            data = np.memmap(shard_path, dtype=dtype, mode="r")
            n = len(data)
            if n == 0:
                continue

            # Vectorized pre-filters
            own_p = data["own_pawn"].astype(np.int32)
            opp_p = data["opp_pawn"].astype(np.int32)
            ow = data["walls_left_own"].astype(np.int32)
            pw = data["walls_left_opp"].astype(np.int32)
            own_dist = data["own_dist"].astype(np.int32)
            opp_dist = data["opp_dist"].astype(np.int32)
            ev = data["nnue_eval"].astype(np.int32)
            res = data["game_result"].astype(np.int32)
            cat_own = data["own_cat_total"].astype(np.int32)
            cat_opp = data["opp_cat_total"].astype(np.int32)

            # 1. Basic validity: not terminal, valid walls
            valid_mask = (own_p < 72) & (opp_p > 8) & (ow >= 0) & (ow <= 10) & (pw >= 0) & (pw <= 10)

            # 2. Race / Value Uncertainty: |ev - 32768| <= 12000 and tight distance margin <= 2
            uncertainty_mask = (np.abs(ev - 32768) <= 12000) & (np.abs(own_dist - opp_dist) <= 2)

            # 3. Critical Wall Dynamics: low reserve or dense board with corridor pressure
            placed_walls = 20 - ow - pw
            wall_crit_mask = (ow <= 3) | (pw <= 3) | ((placed_walls >= 8) & ((cat_own >= 35) | (cat_opp >= 35)))

            # 4. Swings: engine thought one side was ahead but the other won
            swing_mask = ((ev >= 39000) & (res < 0)) | ((ev <= 26000) & (res > 0))

            # Filter for candidates matching at least (uncertainty AND wall_crit) OR swing
            critical_mask = valid_mask & ((uncertainty_mask & wall_crit_mask) | swing_mask)

            cand_indices = np.flatnonzero(critical_mask)
            if len(cand_indices) == 0:
                continue

            # Shuffle candidates from this shard
            rng.shuffle(cand_indices)
            accepted = 0

            for idx in cand_indices:
                row = data[idx]
                p_own = int(row["own_pawn"])
                p_opp = int(row["opp_pawn"])
                w_h = int(row["walls_h"])
                w_v = int(row["walls_v"])
                w_ow = int(row["walls_left_own"])
                w_pw = int(row["walls_left_opp"])

                tup = (p_own, p_opp, w_h, w_v, w_ow, w_pw)
                if tup in seen_keys:
                    continue
                seen_keys.add(tup)

                reasons = []
                if uncertainty_mask[idx]:
                    reasons.append("value_uncertainty")
                if wall_crit_mask[idx]:
                    reasons.append("wall_critical")
                if swing_mask[idx]:
                    reasons.append("game_swing")

                # If size == 64, check entropy of top-8 policy
                if size == 64:
                    top_p = row["policy_top_prob"]
                    if top_p[0] > 0 and top_p[1] > 0:
                        ratio = float(top_p[1]) / float(top_p[0])
                        if ratio >= 0.5:
                            reasons.append("high_entropy")

                sid = hashlib.sha256(f"{p_own}_{p_opp}_{w_h}_{w_v}_{w_ow}_{w_pw}".encode()).hexdigest()[:24]
                state_str = f"@state {p_own} {p_opp} {w_h} {w_v} {w_ow} {w_pw}"
                selected_records.append({
                    "schema": "zquoridor.position.v1",
                    "id": sid,
                    "history": [state_str],
                    "side_to_move": 0,
                    "source": f"generic-critical-{'_'.join(reasons[:2])}",
                    "opening_index": -1,
                    "ply": int(20 - w_ow - w_pw + 10),
                    "metadata": {
                        "reasons": reasons,
                        "nnue_eval": int(row["nnue_eval"]),
                        "own_dist": int(row["own_dist"]),
                        "opp_dist": int(row["opp_dist"]),
                        "placed_walls": int(20 - w_ow - w_pw),
                    },
                })
                accepted += 1
                if accepted >= quota_per_shard:
                    break

            if len(selected_records) >= target_count:
                break
        except Exception:
            continue

    print(f"Total mined generic critical candidates: {len(selected_records):,}", flush=True)
    rng.shuffle(selected_records)
    final_selection = selected_records[:target_count]

    # Assign train/val splits
    n_val = int(len(final_selection) * val_fraction)
    for idx, item in enumerate(final_selection):
        item["split"] = "val" if idx < n_val else "train"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for item in final_selection:
            fh.write(json.dumps(item, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "zquoridor.position_manifest.v1",
        "out": str(out_file),
        "total_positions": len(final_selection),
        "train_positions": len(final_selection) - n_val,
        "val_positions": n_val,
        "target_count": target_count,
        "seed": seed,
    }
    manifest_path = out_file.with_suffix(out_file.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfplay-dir", type=Path, default=ROOT / "data/selfplay_canonical_v3")
    parser.add_argument("--out", type=Path, default=ROOT / "data/teaching/tier3-5-generic-critical-100k/positions.jsonl")
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)

    manifest = mine_tier3_5_critical(
        selfplay_dir=args.selfplay_dir,
        out_file=args.out,
        target_count=args.count,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
