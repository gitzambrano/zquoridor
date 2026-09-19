#!/usr/bin/env python3
"""Assemble hierarchical 5-tier training dataset:
- Tier 1: Massive background replay (10M samples, soft champion net targets, weight ~1.0)
- Tier 2: Reused past search cases (mixed/priority search datasets, weight ~2.0)
- Tier 3: 250,000 generic search positions (Claustrophobia + ZQuoridor search, weight ~3.5)
- Tier 4: Dual-crisis positions (20,000 critical loss/asymmetry states, 75% Claustrophobia MCTS, weight ~6.0 - 8.0)
- Tier 5: Dynamic branching rollouts (10,000 games, temporal discounting gamma=0.98, weight ~6.0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FIELDS = (
    "own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
    "walls_left_own", "walls_left_opp", "policy", "value", "weight", "is_val", "group_id"
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arch:
        data = {k: arch[k] for k in arch.files}
    return data


def assemble_5tier_dataset(
    tiers: list[tuple[str, Path, float]],
    out_path: Path,
) -> dict:
    combined = {f: [] for f in REQUIRED_FIELDS}
    tier_stats = {}

    for name, path, scale in tiers:
        if not path.exists():
            print(f"Skipping absent tier {name}: {path}", flush=True)
            continue

        print(f"Loading {name} from {path} (weight scale: {scale})...", flush=True)
        s = load_npz(path)

        for field in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "policy", "value"):
            if field not in s:
                raise ValueError(f"dataset {name} lacks required field: {field}")

        n = len(s["value"])
        w = s["weight"].astype(np.float32) * float(scale) if "weight" in s else np.ones(n, dtype=np.float32) * float(scale)
        if "is_val" in s and s["is_val"].sum() > 0:
            val = s["is_val"].astype(bool)
        elif "id" in s:
            val = np.array([int(str(x.decode() if isinstance(x, bytes) else x)[:8], 16) % 5 == 0 for x in s["id"]], dtype=bool)
        else:
            val = (np.arange(n) % 5 == 0)

        if "group_id" in s:
            groups = np.array([f"{name}:" + (g.decode('utf-8') if isinstance(g, bytes) else str(g)) for g in s["group_id"]])
        else:
            groups = np.array([f"{name}:{i // 1000}" for i in range(n)])

        tier_stats[name] = {
            "path": str(path),
            "samples": n,
            "mean_weight": float(w.mean()),
            "total_gradient_mass": float(w.sum()),
            "val_fraction": float(val.mean()),
        }

        combined["own_pawn"].append(s["own_pawn"].astype(np.uint8))
        combined["opp_pawn"].append(s["opp_pawn"].astype(np.uint8))
        combined["walls_h"].append(s["walls_h"].astype(np.uint64))
        combined["walls_v"].append(s["walls_v"].astype(np.uint64))
        combined["own_dist"].append(s["own_dist"].astype(np.uint8))
        combined["opp_dist"].append(s["opp_dist"].astype(np.uint8))
        combined["walls_left_own"].append(s["walls_left_own"].astype(np.int8))
        combined["walls_left_opp"].append(s["walls_left_opp"].astype(np.int8))
        combined["policy"].append(s["policy"].astype(np.float16))
        combined["value"].append(s["value"].astype(np.float16))
        combined["weight"].append(w.astype(np.float32))
        combined["is_val"].append(val.astype(bool))
        combined["group_id"].append(groups)

    if not combined["value"]:
        raise ValueError("no valid tiers loaded")

    final_data = {}
    for f in REQUIRED_FIELDS:
        if f in ("group_id",):
            final_data[f] = np.concatenate(combined[f], axis=0).astype("S64")
        else:
            final_data[f] = np.concatenate(combined[f], axis=0)

    total_n = len(final_data["value"])
    total_mass = float(final_data["weight"].sum())
    train_count = int((~final_data["is_val"]).sum())
    val_count = int(final_data["is_val"].sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **final_data)

    manifest = {
        "schema": "zquoridor.5tier_dataset.v1",
        "out": str(out_path),
        "total_samples": total_n,
        "train_samples": train_count,
        "val_samples": val_count,
        "total_gradient_mass": total_mass,
        "tier_stats": tier_stats,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=ROOT / "data/teaching/massive-background-10m/dataset.npz")
    parser.add_argument("--tier1-scale", type=float, default=1.0)
    parser.add_argument("--tier2", type=Path, default=ROOT / "data/teaching/mixed-gen1-500k-search20/dataset.npz")
    parser.add_argument("--tier2-scale", type=float, default=2.0)
    parser.add_argument("--tier3", type=Path, default=ROOT / "data/teaching/generic-search-100k-zq/dataset.npz")
    parser.add_argument("--tier3-scale", type=float, default=3.5)
    parser.add_argument("--tier4", type=Path, default=ROOT / "data/teaching/tier4-dual-crisis-100k/dataset.npz")
    parser.add_argument("--tier4-scale", type=float, default=6.0)
    parser.add_argument("--tier5", type=Path, default=ROOT / "data/teaching/rollouts-10k-games/dataset.npz")
    parser.add_argument("--tier5-scale", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/teaching/5tier-final-dataset/dataset.npz")
    args = parser.parse_args()

    tiers = [
        ("tier1_massive_background", args.tier1, args.tier1_scale),
        ("tier2_reused_past_search", args.tier2, args.tier2_scale),
        ("tier3_generic_search_100k", args.tier3, args.tier3_scale),
        ("tier4_dual_crisis_100k", args.tier4, args.tier4_scale),
        ("tier5_branching_rollouts", args.tier5, args.tier5_scale),
    ]

    manifest = assemble_5tier_dataset(tiers, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
