#!/usr/bin/env python3
"""Assemble the unified clean master dataset for Zquoridor NNUE training.

Combines:
- 11.065M weakness-boosted master dataset (tactical recovery, Claustrophobia defense, Center Rush)
- 4.572M clean canonical stored-search self-play (stored_outcome_weight: 0.0, unpolluted search value targets)

Streams uncompressed arrays in chunks of 50,000 samples to keep RAM usage under 500 MB.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import numpy.lib.format as nformat

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "weakness_master": str(ROOT / "data/teaching/multipath-weakness-boosted-11m/dataset.npz"),
    "clean_replay": str(ROOT / "data/teaching/replay_clean_stored/dataset.npz"),
    "out": str(ROOT / "data/teaching/multipath_unified_clean_15m/dataset.npz"),
    "weakness_scale": 1.0,
    "replay_scale": 1.0,
}

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
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                value = f"{name}:{text}"
                if len(value.encode("utf-8")) <= 63:
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


def run_assembly(config: dict) -> int:
    out_path = Path(config["out"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources_def = [
        ("weakness_master", Path(config["weakness_master"]).resolve(), float(config["weakness_scale"])),
        ("clean_selfplay", Path(config["clean_replay"]).resolve(), float(config["replay_scale"])),
    ]

    active_sources: list[tuple[str, Path, int]] = []
    source_stats = {}
    weights_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    groups_list: list[np.ndarray] = []
    total_n = 0

    for name, path, scale in sources_def:
        if not path.exists():
            raise FileNotFoundError(f"Missing required source {name}: {path}")
        print(f"Inspecting {name} from {path} (scale: {scale})...", flush=True)
        stats, w, val, groups = inspect_source(name, path, scale)
        source_stats[name] = stats
        weights_list.append(w)
        val_list.append(val)
        groups_list.append(groups)
        active_sources.append((name, path, stats["samples"]))
        total_n += stats["samples"]

    print(f"Streaming {total_n:,} samples into {out_path}...", flush=True)
    temp_zip = out_path.with_suffix(".tmp.npz")

    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for field in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                      "walls_left_own", "walls_left_opp", "policy", "value"):
            print(f"  Streaming {field}...", flush=True)
            with zf.open(f"{field}.npy", "w", force_zip64=True) as dest:
                stream_write_array(dest, field, total_n, active_sources)

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

    temp_zip.replace(out_path)

    manifest = {
        "schema": "zquoridor.unified_clean_master_dataset.v1",
        "out": str(out_path),
        "total_samples": total_n,
        "train_samples": total_train,
        "val_samples": total_val,
        "sources": source_stats,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Assembly complete: {out_path} ({total_n:,} samples: {total_train:,} train, {total_val:,} val)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args(argv)
    cfg = copy.deepcopy(CONFIG)
    cfg.update(vars(args))
    return run_assembly(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
