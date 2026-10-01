#!/usr/bin/env python3
"""Build a disk-backed, split-safe experimental training mixture.

Inputs are canonical teaching NPZs or directory datasets produced by this
module.  Repeated canonical positions are merged by weighted target average;
source fractions are enforced independently in train and validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE_FIELDS = ("own_pawn", "opp_pawn", "walls_h", "walls_v",
                "walls_left_own", "walls_left_opp")
GROUP_FIELDS = ("group_id", "opening_index")
REQUIRED = (*STATE_FIELDS, "own_dist", "opp_dist", "policy", "value", "weight", "is_val")
POLICY_SIZE = 209
SCHEMA = "zquoridor.experimental_mixture.v1"
CONFIG = {
    "sources": [],
    "out_dir": "results/experiments/experimental-mix/dataset",
    "chunk_size": 8192,
    "resume": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _source_identity(path: Path) -> dict:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"input file must be NPZ: {path}")
        return {"kind": "npz", "path": str(path.resolve()), "sha256": _sha256(path)}
    manifest_path = path / "dataset.manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset directory lacks dataset.manifest.json: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"dataset directory is not marked complete: {path}")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError(f"dataset directory has no arrays mapping: {path}")
    records = []
    for name, info in sorted(arrays.items()):
        file_name = info.get("filename") if isinstance(info, dict) else None
        rel = Path(file_name) if file_name else None
        if not file_name or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid array filename for {name!r}")
        file_path = path / file_name
        if not file_path.is_file():
            raise ValueError(f"missing dataset array: {file_path}")
        actual = _sha256(file_path)
        if info.get("sha256") != actual:
            raise ValueError(f"array SHA-256 mismatch: {file_path}")
        records.append({"name": name, "file": file_name, "sha256": actual})
    return {"kind": "directory", "path": str(path.resolve()),
            "manifest_sha256": _sha256(manifest_path), "arrays": records}


def _npz_cache(npz_path: Path, cache_dir: Path) -> Path:
    digest = _sha256(npz_path)
    cache = cache_dir / digest
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".npy")]
        for member in members:
            leaf = Path(member).name
            if leaf != member or not leaf:
                raise ValueError(f"unsafe NPZ member path: {member}")
            dest = cache / leaf
            if dest.exists():
                continue
            with archive.open(member) as src, tempfile.NamedTemporaryFile(
                    dir=cache, prefix=leaf + ".", suffix=".tmp", delete=False) as dst:
                tmp = Path(dst.name)
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    dst.write(block)
            tmp.replace(dest)
    return cache


def load_arrays(path, cache_dir=None) -> dict[str, np.ndarray]:
    """Load NPZ/directory fields as read-only arrays, memmapping large fields."""
    path = Path(path)
    if path.is_file():
        if cache_dir is None:
            cache_dir = path.parent / ".input_cache"
        cache = _npz_cache(path, Path(cache_dir))
        arrays = {file.stem: np.load(file, mmap_mode="r", allow_pickle=False)
                  for file in cache.glob("*.npy")}
    else:
        manifest = json.loads((path / "dataset.manifest.json").read_text(encoding="utf-8"))
        identity = _source_identity(path)
        del identity  # validates every recorded array before exposing it
        arrays = {name: np.load(path / info["filename"], mmap_mode="r", allow_pickle=False)
                  for name, info in manifest["arrays"].items()}
    _validate_arrays(arrays, str(path))
    return arrays


def _validate_arrays(data: dict, label: str) -> int:
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError(f"{label} lacks required fields: {', '.join(missing)}")
    n = len(data["value"])
    if n < 2 or any(len(data[key]) != n for key in REQUIRED):
        raise ValueError(f"{label} has empty or inconsistent fields")
    if data["policy"].shape != (n, POLICY_SIZE):
        raise ValueError(f"{label} policy must have shape (N, 209)")
    if not np.issubdtype(data["is_val"].dtype, np.bool_):
        raise ValueError(f"{label} is_val must be boolean")
    for start in range(0, n, 500_000):
        stop = min(n, start + 500_000)
        values = np.asarray(data["value"][start:stop])
        weights = np.asarray(data["weight"][start:stop])
        policies = np.asarray(data["policy"][start:stop], dtype=np.float32)
        if not np.isfinite(values).all() or (np.abs(values) > 1.001).any():
            raise ValueError(f"{label} has invalid signed value targets")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(f"{label} has nonpositive or nonfinite sample weights")
        if (not np.isfinite(policies).all() or (policies < 0).any()
                or not np.allclose(policies.sum(axis=1), 1, atol=0.005)):
            raise ValueError(f"{label} has invalid policy targets")
        for field, maximum in (("own_pawn", 80), ("opp_pawn", 80),
                               ("walls_left_own", 10), ("walls_left_opp", 10),
                               ("own_dist", 81), ("opp_dist", 81)):
            values = np.asarray(data[field][start:stop])
            if (values < 0).any() or (values > maximum).any():
                raise ValueError(f"{label} has invalid {field} values")
    return n


def _key_dtype() -> np.dtype:
    return np.dtype([(name, "<u8" if name in ("walls_h", "walls_v") else "<i2")
                     for name in STATE_FIELDS])


def _make_keys(data, rows) -> np.ndarray:
    keys = np.empty(len(rows), dtype=_key_dtype())
    for name in STATE_FIELDS:
        keys[name] = data[name][rows]
    return keys


def _bad_groups(data, n: int, chunk_size: int) -> list[tuple[str, set]]:
    result = []
    for field in GROUP_FIELDS:
        if field not in data:
            continue
        train_groups, val_groups = set(), set()
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            groups = np.asarray(data[field][start:stop]).astype(str)
            split = np.asarray(data["is_val"][start:stop], dtype=bool)
            train_groups.update(groups[~split].tolist())
            val_groups.update(groups[split].tolist())
        result.append((field, train_groups.intersection(val_groups)))
    return result


def _row_keep(data, start, stop, bad_groups):
    keep = np.ones(stop - start, dtype=bool)
    for field, conflict_groups in bad_groups:
        if conflict_groups:
            groups = np.asarray(data[field][start:stop]).astype(str)
            keep &= ~np.isin(groups, np.asarray(list(conflict_groups), dtype=str))
    return keep


def _atomic_json(path: Path, value: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_existing(out_dir: Path, identity_hash: str):
    manifest_path = out_dir / "dataset.manifest.json"
    if not manifest_path.exists():
        if any(p.name != ".input_cache" for p in out_dir.iterdir()):
            raise FileExistsError(f"output directory has incomplete prior output: {out_dir}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("identity_sha256") != identity_hash:
        raise ValueError("output exists but its completed identity does not match these inputs")
    for name, info in manifest["arrays"].items():
        file_path = out_dir / info["filename"]
        if not file_path.is_file() or _sha256(file_path) != info["sha256"]:
            raise ValueError(f"completed output array failed SHA-256 validation: {name}")
    return manifest


def mix_datasets(sources, out_dir, chunk_size=8192, resume=True):
    """Mix canonical datasets with neutral weighted deduplication.

    Explicit sources use `{name, path, fraction, weight_cap}`. Dynamic pools
    use `{name, path, fraction: null, pool, pool_fraction, weight_cap}`;
    members split the pool budget by eligible capped raw mass. Shares are
    enforced independently in train and validation.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not sources:
        raise ValueError("at least one source is required")
    normalized = []
    names = set()
    explicit_fraction_sum = 0.0
    pool_budgets = {}
    for source in sources:
        name = str(source["name"])
        pool = source.get("pool")
        fraction = source.get("fraction")
        fraction = None if fraction is None else float(fraction)
        pool_fraction = source.get("pool_fraction")
        pool_fraction = None if pool_fraction is None else float(pool_fraction)
        cap = source.get("weight_cap")
        cap = None if cap is None else float(cap)
        if not name or name in names:
            raise ValueError("source names must be nonempty and unique")
        if pool is None:
            if fraction is None or not math.isfinite(fraction) or fraction <= 0:
                raise ValueError("explicit source fractions must be finite and positive")
            if pool_fraction is not None:
                raise ValueError("pool_fraction requires a pool name")
            explicit_fraction_sum += fraction
        else:
            pool = str(pool)
            if fraction is not None:
                raise ValueError("pooled sources must set fraction=null")
            if pool_fraction is None or not math.isfinite(pool_fraction) or pool_fraction <= 0:
                raise ValueError("pooled sources require a positive pool_fraction")
            if pool in pool_budgets and not math.isclose(pool_budgets[pool], pool_fraction, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"inconsistent pool_fraction for pool {pool!r}")
            pool_budgets[pool] = pool_fraction
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("weight_cap must be finite and positive or null")
        names.add(name)
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        ident = _source_identity(path)
        normalized.append({"name": name, "path": path, "fraction": fraction,
                           "pool": pool, "pool_fraction": pool_fraction,
                           "weight_cap": cap, "identity": ident})
    fraction_sum = explicit_fraction_sum + sum(pool_budgets.values())
    if not math.isclose(fraction_sum, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"source fractions must sum to 1 (got {fraction_sum})")

    identity = {"sources": [{"name": s["name"], "fraction": s["fraction"],
                             "pool": s["pool"], "pool_fraction": s["pool_fraction"],
                             "weight_cap": s["weight_cap"], "identity": s["identity"]}
                            for s in normalized]}
    identity_hash = _json_sha(identity)
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(out_dir, identity_hash)
    if existing is not None:
        if not resume:
            raise FileExistsError(f"completed output already exists: {out_dir}")
        return existing
    cache_dir = out_dir / ".input_cache"
    cache_dir.mkdir(exist_ok=True)

    # Count surviving rows before allocating one compact state-key table.
    source_data, source_bad = [], []
    total_rows = 0
    for source in normalized:
        arrays = load_arrays(source["path"], cache_dir)
        n = len(arrays["value"])
        if not any(field in arrays for field in GROUP_FIELDS):
            raise ValueError(f"source {source['name']} must carry group_id or opening_index for split verification")
        bad = _bad_groups(arrays, n, chunk_size)
        kept = 0
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            kept += int(_row_keep(arrays, start, stop, bad).sum())
        if kept < 2:
            raise ValueError(f"source {source['name']} has fewer than two rows after group leakage removal")
        source_data.append(arrays)
        source_bad.append(bad)
        total_rows += kept

    all_keys = np.empty(total_rows, dtype=_key_dtype())
    all_split = np.empty(total_rows, dtype=np.uint8)
    offsets = []
    cursor = 0
    for i, arrays in enumerate(source_data):
        n = len(arrays["value"])
        begin = cursor
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            keep = _row_keep(arrays, start, stop, source_bad[i])
            if not keep.any():
                continue
            rows = np.flatnonzero(keep) + start
            count = len(rows)
            all_keys[cursor:cursor + count] = _make_keys(arrays, rows)
            all_split[cursor:cursor + count] = np.where(arrays["is_val"][rows], 2, 1)
            cursor += count
        offsets.append((begin, cursor))
    if cursor != total_rows:
        raise RuntimeError("source row count changed while mixing")

    unique_keys, inverse = np.unique(all_keys, return_inverse=True)
    split_flags = np.zeros(len(unique_keys), dtype=np.uint8)
    np.bitwise_or.at(split_flags, inverse, all_split)
    split_conflict = split_flags == 3
    keep_key = ~split_conflict
    train_unique = int(np.count_nonzero(keep_key & (split_flags == 1)))
    val_unique = int(np.count_nonzero(keep_key & (split_flags == 2)))
    if train_unique == 0 or val_unique == 0:
        raise ValueError("both train and validation must retain nonconflicting positions")
    unique_keys = unique_keys[keep_key]
    global_splits = split_flags[keep_key] == 2
    del inverse, split_flags, split_conflict, keep_key, all_keys, all_split

    n_unique, n_sources = len(unique_keys), len(normalized)
    fields_dir = out_dir / "fields"
    fields_dir.mkdir(exist_ok=True)
    temp_dir = out_dir / ".mix_work"
    if temp_dir.exists() and any(temp_dir.iterdir()):
        raise FileExistsError(f"incomplete temporary mixture exists: {temp_dir}")
    temp_dir.mkdir(exist_ok=True)

    def mmap(name, dtype, shape):
        return np.lib.format.open_memmap(temp_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    weight_sum = mmap("weight_sum", np.float64, (n_unique,))
    value_sum = mmap("value_sum", np.float64, (n_unique,))
    policy_sum = mmap("policy_sum", np.float32, (n_unique, POLICY_SIZE))
    source_mass = mmap("source_mass", np.float64, (n_unique, n_sources))
    own_dist_out = mmap("own_dist", np.uint8, (n_unique,))
    opp_dist_out = mmap("opp_dist", np.uint8, (n_unique,))
    own_dist_out[:] = 255
    opp_dist_out[:] = 255
    raw_masses = np.zeros((n_sources, 2), dtype=np.float64)
    group_removed = [sum(len(groups) for _, groups in bad) for bad in source_bad]
    conflict_removed_rows = [0] * n_sources

    # First count per-source available mass after global split conflicts.
    for si, arrays in enumerate(source_data):
        n = len(arrays["value"])
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            keep = _row_keep(arrays, start, stop, source_bad[si])
            rows = np.flatnonzero(keep) + start
            if not len(rows):
                continue
            ids = np.searchsorted(unique_keys, _make_keys(arrays, rows))
            valid = (ids < n_unique)
            valid &= unique_keys[np.minimum(ids, n_unique - 1)] == _make_keys(arrays, rows)
            # Split-conflicting keys were removed from unique_keys; membership verifies them.
            conflict_removed_rows[si] += int((~valid).sum())
            if not valid.any():
                continue
            rows, ids = rows[valid], ids[valid]
            val_mask = np.asarray(arrays["is_val"][rows], dtype=bool)
            raw_w = np.asarray(arrays["weight"][rows], dtype=np.float64)
            cap = normalized[si]["weight_cap"]
            if cap is not None:
                raw_w = np.minimum(raw_w, cap)
            raw_masses[si, 0] += raw_w[~val_mask].sum()
            raw_masses[si, 1] += raw_w[val_mask].sum()
    if np.any(raw_masses <= 0):
        raise ValueError("a source has no train or validation mass after conflict removal")

    # Explicit sources keep their specified share; each pool shares its total
    # budget in proportion to its eligible capped raw weight, separately by split.
    effective_fractions = np.zeros((n_sources, 2), dtype=np.float64)
    for si, source in enumerate(normalized):
        if source["pool"] is None:
            effective_fractions[si, :] = source["fraction"]
        else:
            members = [k for k, candidate in enumerate(normalized) if candidate["pool"] == source["pool"]]
            for split_i in range(2):
                denominator = raw_masses[members, split_i].sum()
                if denominator <= 0:
                    raise ValueError(f"pool {source['pool']!r} has no eligible mass in a split")
                effective_fractions[si, split_i] = (
                    source["pool_fraction"] * raw_masses[si, split_i] / denominator)
    if not np.allclose(effective_fractions.sum(axis=0), 1.0, rtol=0, atol=1e-8):
        raise RuntimeError("effective source fractions do not sum to one in each split")
    target_split_mass = np.asarray([train_unique, val_unique], dtype=np.float64)
    scales = np.asarray([[effective_fractions[i, j] * target_split_mass[j] / raw_masses[i, j]
                          for j in range(2)] for i in range(n_sources)])

    # Accumulate weighted targets by global state id; policy rows stay chunk-sized in RAM.
    for si, arrays in enumerate(source_data):
        n = len(arrays["value"])
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            keep = _row_keep(arrays, start, stop, source_bad[si])
            rows = np.flatnonzero(keep) + start
            if not len(rows):
                continue
            keys = _make_keys(arrays, rows)
            ids = np.searchsorted(unique_keys, keys)
            valid = ids < n_unique
            valid &= unique_keys[np.minimum(ids, n_unique - 1)] == keys
            if not valid.any():
                continue
            rows, ids = rows[valid], ids[valid]
            is_val = np.asarray(arrays["is_val"][rows], dtype=bool)
            raw_w = np.asarray(arrays["weight"][rows], dtype=np.float64)
            cap = normalized[si]["weight_cap"]
            if cap is not None:
                raw_w = np.minimum(raw_w, cap)
            normalized_w = raw_w * np.where(is_val, scales[si, 1], scales[si, 0])
            vals = np.asarray(arrays["value"][rows], dtype=np.float64)
            pol = np.asarray(arrays["policy"][rows], dtype=np.float64)
            if not np.isfinite(vals).all() or (np.abs(vals) > 1.001).any():
                raise ValueError(f"invalid value target in {normalized[si]['name']}")
            if not np.isfinite(pol).all() or (pol < 0).any() or not np.allclose(pol.sum(axis=1), 1, atol=0.005):
                raise ValueError(f"invalid policy target in {normalized[si]['name']}")
            np.add.at(weight_sum, ids, normalized_w)
            np.add.at(value_sum, ids, vals * normalized_w)
            np.add.at(policy_sum, ids, pol.astype(np.float32) * normalized_w.astype(np.float32)[:, None])
            np.add.at(source_mass[:, si], ids, normalized_w)

            # Derived distances must agree for identical canonical states.
            for field, target in (("own_dist", own_dist_out), ("opp_dist", opp_dist_out)):
                incoming = np.asarray(arrays[field][rows], dtype=np.uint8)
                order = np.argsort(ids, kind="stable")
                sorted_ids, sorted_values = ids[order], incoming[order]
                starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
                local_ids, local_values = sorted_ids[starts], sorted_values[starts]
                if np.any(sorted_values != local_values[np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(sorted_ids)]))]):
                    raise ValueError(f"inconsistent {field} for duplicate canonical state")
                old = target[local_ids]
                if np.any((old != 255) & (old != local_values)):
                    raise ValueError(f"inconsistent {field} across sources")
                target[local_ids] = local_values

    # Materialize final fields and hash every file before the completion marker.
    output_specs = {
        **{name: (dtype, (n_unique,)) for name, dtype in (("own_pawn", np.uint8), ("opp_pawn", np.uint8),
            ("walls_h", np.uint64), ("walls_v", np.uint64), ("walls_left_own", np.int8), ("walls_left_opp", np.int8))},
        "own_dist": (np.uint8, (n_unique,)), "opp_dist": (np.uint8, (n_unique,)),
        "policy": (np.float32, (n_unique, POLICY_SIZE)), "value": (np.float32, (n_unique,)),
        "weight": (np.float32, (n_unique,)), "is_val": (np.bool_, (n_unique,)),
        "source_mass": (np.float32, (n_unique, n_sources)),
    }
    outputs = {name: np.lib.format.open_memmap(fields_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
               for name, (dtype, shape) in output_specs.items()}
    for field in STATE_FIELDS:
        outputs[field][:] = unique_keys[field]
    outputs["own_dist"][:] = own_dist_out
    outputs["opp_dist"][:] = opp_dist_out
    outputs["is_val"][:] = global_splits
    for start in range(0, n_unique, chunk_size):
        stop = min(n_unique, start + chunk_size)
        w = np.asarray(weight_sum[start:stop])
        if not np.isfinite(w).all() or (w <= 0).any():
            raise RuntimeError("a merged state has no positive target mass")
        outputs["weight"][start:stop] = w
        outputs["value"][start:stop] = value_sum[start:stop] / w
        p = policy_sum[start:stop] / w[:, None]
        psum = p.sum(axis=1)
        if (psum <= 0).any():
            raise RuntimeError("merged policy has zero total probability")
        outputs["policy"][start:stop] = (p / psum[:, None]).astype(np.float32)
        outputs["source_mass"][start:stop] = source_mass[start:stop]
    for arr in outputs.values():
        arr.flush()

    arrays_manifest = {}
    for name in output_specs:
        file_path = fields_dir / f"{name}.npy"
        arrays_manifest[name] = {"filename": str(Path("fields") / file_path.name),
                                 "sha256": _sha256(file_path),
                                 "shape": list(outputs[name].shape), "dtype": str(outputs[name].dtype)}
    source_summary = []
    for si, source in enumerate(normalized):
        row = {"name": source["name"], "fraction": source["fraction"],
               "pool": source["pool"], "pool_fraction": source["pool_fraction"],
               "effective_fraction_train": float(effective_fractions[si, 0]),
               "effective_fraction_val": float(effective_fractions[si, 1]),
               "weight_cap": source["weight_cap"], "identity": source["identity"],
               "removed_group_count": group_removed[si],
               "group_fields_checked": [field for field, _ in source_bad[si]],
               "removed_split_conflict_rows": conflict_removed_rows[si],
               "train_raw_weight_after_filter": float(raw_masses[si, 0]),
               "val_raw_weight_after_filter": float(raw_masses[si, 1]),
               "train_weight_scale": float(scales[si, 0]),
               "val_weight_scale": float(scales[si, 1]),
               "train_effective_mass": float(source_mass[:, si][~global_splits].sum()),
               "val_effective_mass": float(source_mass[:, si][global_splits].sum())}
        source_summary.append(row)
    manifest = {"schema": SCHEMA, "complete": True, "identity_sha256": identity_hash,
                "group_separation_verified": True,
                "sources": source_summary, "source_names": [s["name"] for s in normalized],
                "source_mass": {"array": "source_mass", "source_names": [s["name"] for s in normalized],
                                "row_sum_matches_weight": True},
                "deduplication": {"key_fields": list(STATE_FIELDS), "target_merge": "weighted_mean_by_capped_source_weight",
                                  "cross_split_conflicts_removed": True, "group_leakage_policy": "remove_entire_group"},
                "samples": n_unique, "train_samples": train_unique, "val_samples": val_unique,
                "train_weight_sum": float(weight_sum[~global_splits].sum()),
                "val_weight_sum": float(weight_sum[global_splits].sum()),
                "source_mass_mean_error": float(np.max(np.abs(source_mass.sum(axis=1) - weight_sum))),
                "arrays": arrays_manifest}
    # Assert the advertised proportions against actual float64 mass.
    for si, source in enumerate(normalized):
        expected_train = effective_fractions[si, 0] * train_unique
        expected_val = effective_fractions[si, 1] * val_unique
        if not math.isclose(source_summary[si]["train_effective_mass"], expected_train, rel_tol=2e-6, abs_tol=2e-4):
            raise RuntimeError(f"train effective mass mismatch for {source['name']}")
        if not math.isclose(source_summary[si]["val_effective_mass"], expected_val, rel_tol=2e-6, abs_tol=2e-4):
            raise RuntimeError(f"validation effective mass mismatch for {source['name']}")
    _atomic_json(out_dir / "dataset.manifest.json", manifest)
    for arr in (weight_sum, value_sum, policy_sum, source_mass, own_dist_out, opp_dist_out):
        arr.flush()
        arr._mmap.close()
    for file_path in temp_dir.iterdir():
        file_path.unlink()
    temp_dir.rmdir()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", help="JSON file or JSON array of source records")
    parser.add_argument("--out-dir", default=CONFIG["out_dir"])
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=CONFIG["resume"])
    args = parser.parse_args(argv)
    sources = CONFIG["sources"]
    if args.sources:
        candidate = Path(args.sources)
        sources = json.loads(candidate.read_text(encoding="utf-8")) if candidate.is_file() else json.loads(args.sources)
    result = mix_datasets(sources, args.out_dir, args.chunk_size, args.resume)
    print(json.dumps({k: result[k] for k in ("schema", "samples", "train_samples", "val_samples", "identity_sha256")}, indent=2))


if __name__ == "__main__":
    main()
