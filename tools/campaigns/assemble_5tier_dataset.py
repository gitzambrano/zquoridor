#!/usr/bin/env python3
"""Assemble hierarchical multi-tier training dataset for Zquoridor.

Combines dataset tiers into one canonical dataset.npz:
- Tier 1: Massive background replay (10M samples, soft champion targets, weight ~1.0)
- Tier 2: Reused past search cases (500k samples, weight ~2.0)
- Tier 3: Generic search positions (100k samples, bilateral search, weight ~3.5)
- Tier 3.5: Generic critical search (100k samples, 80% Claustro + 20% ZQ, weight ~4.5)
- Tier 4: Dual-crisis search (100k samples, 75% Claustro + 25% ZQ, weight ~6.0)
- Tier 5 Deep Search: Deep search positions on CUDA (10k samples, 512 sims, weight ~6.0)
- Tier 5 Rollouts: Dynamic branching rollouts from crisis seeds (discounted gamma=0.98, weight ~6.0)

Writes arrays into the target NPZ archive sequentially using chunked streaming
to minimize peak memory usage.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import zipfile
import numpy as np
import numpy.lib.format as nformat

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FIELDS = (
    "own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
    "walls_left_own", "walls_left_opp", "policy", "value", "weight", "is_val", "group_id"
)

FIELD_DTYPES = {
    "own_pawn": np.uint8,
    "opp_pawn": np.uint8,
    "walls_h": np.uint64,
    "walls_v": np.uint64,
    "own_dist": np.uint8,
    "opp_dist": np.uint8,
    "walls_left_own": np.int8,
    "walls_left_opp": np.int8,
    "policy": np.float16,
    "value": np.float16,
    "weight": np.float32,
    "is_val": bool,
    "group_id": "S64",
}


def inspect_tier(name: str, path: Path, scale: float) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load metadata, scaled weights, validation flags, and group IDs for a tier."""
    with np.load(path, allow_pickle=False) as s:
        for field in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "policy", "value"):
            if field not in s:
                raise ValueError(f"dataset {name} lacks required field: {field}")
        n = len(s["value"])
        if "weight" in s:
            w = s["weight"].astype(np.float32) * float(scale)
        else:
            w = np.full(n, float(scale), dtype=np.float32)

        if "is_val" in s and s["is_val"].sum() > 0:
            val = s["is_val"].astype(bool)
        elif "id" in s:
            val = np.array([int(str(x.decode() if isinstance(x, bytes) else x)[:8], 16) % 5 == 0 for x in s["id"]], dtype=bool)
        else:
            val = (np.arange(n) % 5 == 0)

        if "group_id" in s and len(s["group_id"]) == n:
            raw_groups = s["group_id"]
            g_train = set(raw_groups[~val])
            g_val = set(raw_groups[val])
            if not (g_train & g_val):
                groups = np.array([f"{name}:" + (g.decode('utf-8') if isinstance(g, bytes) else str(g)) for g in raw_groups], dtype="S64")
            else:
                groups = np.array([f"{name}:val_{i // 500}" if val[i] else f"{name}:train_{i // 500}" for i in range(n)], dtype="S64")
        else:
            groups = np.array([f"{name}:val_{i // 500}" if val[i] else f"{name}:train_{i // 500}" for i in range(n)], dtype="S64")

    stats = {
        "path": str(path),
        "samples": n,
        "mean_weight": float(w.mean()),
        "total_gradient_mass": float(w.sum()),
        "val_fraction": float(val.mean()),
    }
    return stats, w, val, groups


def stream_write_array(dest, field: str, total_n: int, active_tiers: list[tuple[str, Path, int]]):
    """Write an NPY header and stream tier contents in chunks directly to the archive."""
    dtype = np.dtype(FIELD_DTYPES[field])
    shape = (total_n, 209) if field == "policy" else (total_n,)
    header = {
        "descr": nformat.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    nformat.write_array_header_2_0(dest, header)

    chunk_size = 50_000
    for name, path, n in active_tiers:
        with np.load(path, allow_pickle=False) as s:
            src = s[field]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = src[start:end].astype(dtype, copy=False)
                dest.write(chunk.tobytes())
        gc.collect()


def assemble_multitier_dataset(
    tiers: list[tuple[str, Path, float]],
    out_path: Path,
) -> dict:
    """Assemble all valid tiers into a single NPZ archive."""
    active_tiers: list[tuple[str, Path, int]] = []
    tier_stats = {}
    weights_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    groups_list: list[np.ndarray] = []
    total_n = 0

    for name, path, scale in tiers:
        if path is None or not path.exists():
            print(f"Skipping absent tier {name}: {path}", flush=True)
            continue
        print(f"Inspecting {name} from {path} (weight scale: {scale})...", flush=True)
        stats, w, val, groups = inspect_tier(name, path, scale)
        tier_stats[name] = stats
        weights_list.append(w)
        val_list.append(val)
        groups_list.append(groups)
        active_tiers.append((name, path, stats["samples"]))
        total_n += stats["samples"]

    if not active_tiers:
        raise ValueError("no valid tiers loaded")

    print(f"Assembling {len(active_tiers)} tiers into {out_path} (total samples: {total_n:,})...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        # 1. Write weight
        print("Streaming field: weight...", flush=True)
        w_all = np.concatenate(weights_list, axis=0).astype(np.float32)
        total_mass = float(w_all.sum())
        with zf.open("weight.npy", "w", force_zip64=True) as dest:
            np.lib.format.write_array(dest, w_all, allow_pickle=False)
        del w_all, weights_list
        gc.collect()

        # 2. Write is_val
        print("Streaming field: is_val...", flush=True)
        val_all = np.concatenate(val_list, axis=0).astype(bool)
        train_count = int((~val_all).sum())
        val_count = int(val_all.sum())
        with zf.open("is_val.npy", "w", force_zip64=True) as dest:
            np.lib.format.write_array(dest, val_all, allow_pickle=False)
        del val_all, val_list
        gc.collect()

        # 3. Write group_id
        print("Streaming field: group_id...", flush=True)
        groups_all = np.concatenate(groups_list, axis=0).astype("S64")
        with zf.open("group_id.npy", "w", force_zip64=True) as dest:
            np.lib.format.write_array(dest, groups_all, allow_pickle=False)
        del groups_all, groups_list
        gc.collect()

        # 4. Stream remaining state and target fields
        direct_fields = (
            "own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
            "walls_left_own", "walls_left_opp", "policy", "value"
        )
        for field in direct_fields:
            print(f"Streaming field: {field}...", flush=True)
            with zf.open(f"{field}.npy", "w", force_zip64=True) as dest:
                stream_write_array(dest, field, total_n, active_tiers)

    manifest = {
        "schema": "zquoridor.multitier_dataset.v1",
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
    parser.add_argument("--tier3-5", type=Path, default=ROOT / "data/teaching/tier3-5-generic-critical-100k/dataset.npz")
    parser.add_argument("--tier3-5-scale", type=float, default=4.5)
    parser.add_argument("--tier4", type=Path, default=ROOT / "data/teaching/tier4-dual-crisis-100k/dataset.npz")
    parser.add_argument("--tier4-scale", type=float, default=6.0)
    parser.add_argument("--tier5-deep-search", type=Path, default=ROOT / "data/teaching/tier5-deep-search-10k/dataset.npz")
    parser.add_argument("--tier5-deep-search-scale", type=float, default=6.0)
    parser.add_argument("--tier5-rollouts", type=Path, default=ROOT / "data/teaching/rollouts-500-seed/dataset.npz")
    parser.add_argument("--tier5-rollouts-scale", type=float, default=6.0)
    parser.add_argument("--tier5", type=Path, default=None)
    parser.add_argument("--tier5-scale", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/teaching/multitier-final-dataset/dataset.npz")
    args = parser.parse_args()

    tiers = [
        ("tier1_massive_background", args.tier1, args.tier1_scale),
        ("tier2_reused_past_search", args.tier2, args.tier2_scale),
        ("tier3_generic_search_100k", args.tier3, args.tier3_scale),
        ("tier3_5_generic_critical_100k", args.tier3_5, args.tier3_5_scale),
        ("tier4_dual_crisis_100k", args.tier4, args.tier4_scale),
        ("tier5_deep_search_10k", args.tier5_deep_search, args.tier5_deep_search_scale),
        ("tier5_branching_rollouts", args.tier5_rollouts, args.tier5_rollouts_scale),
    ]
    if args.tier5 is not None:
        tiers.append(("tier5_legacy", args.tier5, args.tier5_scale))

    manifest = assemble_multitier_dataset(tiers, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
