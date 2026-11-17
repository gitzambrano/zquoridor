#!/usr/bin/env python3
"""Assemble the full multi-tier master dataset for the experimental multipath architecture.

Combines all previous knowledge:
- Multi-tier consolidated dataset (Tiers 1, 2, 3, 3.5, 4, 5: 10,810,000 samples)
- Center-Rush priority dataset with Action-Q sharpening (255,000 samples)

Streams uncompressed arrays in chunks to keep RAM usage low.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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


def inspect_source(name: str, path: Path, scale: float) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as s:
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
            def group_key(raw):
                text = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else str(raw)
                value = f"{name}:{text}"
                if len(value.encode('utf-8')) <= 63:
                    return value
                return f"{name}:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
            groups = np.array([group_key(g) for g in raw_groups], dtype="S64")
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


def stream_write_array(dest, field: str, total_n: int, sources: list[tuple[str, Path, int]]):
    dtype = np.dtype(FIELD_DTYPES[field])
    shape = (total_n, 209) if field == "policy" else (total_n,)
    header = {
        "descr": nformat.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    nformat.write_array_header_2_0(dest, header)

    chunk_size = 50_000
    for name, path, n in sources:
        with np.load(path, allow_pickle=False) as s:
            src = s[field]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = src[start:end].astype(dtype, copy=False)
                dest.write(chunk.tobytes())
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/teaching/multipath-master-11m/dataset.npz",
    )
    parser.add_argument(
        "--multitier",
        type=Path,
        default=ROOT / "data/teaching/multitier-final-dataset/dataset.npz",
    )
    parser.add_argument(
        "--center-rush",
        type=Path,
        default=ROOT / "data/teaching/center-rush-200k-priority/dataset.npz",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    sources_def = [
        ("multitier_all", args.multitier, 1.0),
        ("center_rush_priority", args.center_rush, 2.0),
    ]

    active_sources: list[tuple[str, Path, int]] = []
    source_stats = {}
    weights_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    groups_list: list[np.ndarray] = []
    total_n = 0

    for name, path, scale in sources_def:
        if not path.exists():
            print(f"Skipping absent source {name}: {path}", flush=True)
            continue
        print(f"Inspecting {name} from {path} (scale: {scale})...", flush=True)
        stats, w, val, groups = inspect_source(name, path, scale)
        source_stats[name] = stats
        weights_list.append(w)
        val_list.append(val)
        groups_list.append(groups)
        active_sources.append((name, path, stats["samples"]))
        total_n += stats["samples"]

    print(f"Streaming {total_n:,} samples into {args.out}...", flush=True)
    temp_zip = args.out.with_suffix(".tmp.npz")

    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        # Stream raw state fields and policy/value
        for field in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                      "walls_left_own", "walls_left_opp", "policy", "value"):
            print(f"  Streaming {field}...", flush=True)
            with zf.open(f"{field}.npy", "w", force_zip64=True) as dest:
                stream_write_array(dest, field, total_n, active_sources)

        # Write merged weights, val split, and group_ids
        print("  Writing merged weights...", flush=True)
        merged_weights = np.concatenate(weights_list, axis=0)
        with zf.open("weight.npy", "w", force_zip64=True) as dest:
            nformat.write_array(dest, merged_weights, version=(2, 0))
        del merged_weights
        del weights_list
        gc.collect()

        print("  Writing merged is_val...", flush=True)
        merged_val = np.concatenate(val_list, axis=0)
        with zf.open("is_val.npy", "w", force_zip64=True) as dest:
            nformat.write_array(dest, merged_val, version=(2, 0))
        total_val = int(merged_val.sum())
        total_train = total_n - total_val
        del merged_val
        del val_list
        gc.collect()

        print("  Writing merged group_id...", flush=True)
        merged_groups = np.concatenate(groups_list, axis=0)
        with zf.open("group_id.npy", "w", force_zip64=True) as dest:
            nformat.write_array(dest, merged_groups, version=(2, 0))
        del merged_groups
        del groups_list
        gc.collect()

    temp_zip.replace(args.out)

    manifest = {
        "schema": "zquoridor.multipath_master_dataset.v1",
        "out": str(args.out),
        "total_samples": total_n,
        "train_samples": total_train,
        "val_samples": total_val,
        "sources": source_stats,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Assembly complete: {args.out} ({total_n:,} samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
