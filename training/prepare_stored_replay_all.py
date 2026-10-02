#!/usr/bin/env python3
"""Prepare all frozen accepted stored-search shards into a disk-backed replay.

This is preparation only: no engine search, external teacher, or network
training is run. SQLite holds the deduplicated canonical states and dense
policy sums so RAM use is bounded by the input chunk.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.read_selfplay import SAMPLE_DTYPE

if SAMPLE_DTYPE.itemsize != 64:
    raise RuntimeError("stored replay preparation requires the 64-byte V3 sample layout")

META_DTYPE = np.dtype([("game", "<u8"), ("root", "<f4"), ("plies", "<u2"),
                       ("length", "<u2"), ("source", "u1"), ("flags", "u1"), ("reserved", "<u2")])
META_ROOT_MISSING = 1
META_ROOT_VALID = 2
STATE_FIELDS = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own", "walls_left_opp")
POLICY_SIZE = 209
SCHEMA = "zquoridor.stored_replay_all.v1"
CONFIG = {
    "source": "data/selfplay/central-weakness-rollouts-16m",
    "out_dir": "data/teaching/replay-all-stored",
    "seed": 20261001,
    "val_fraction": 0.2,
    "chunk_size": 8192,
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _safe_child(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"shard path escapes source directory: {value}")
    return path


def _resolve_source(source):
    """Return frozen shard descriptors plus their source manifest metadata."""
    root = Path(source)
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve()
    if root.is_file():
        if root.name.endswith("campaign_manifest.json"):
            manifest_path = root
            root = root.parent
        elif root.name in ("manifest.json", "replay_manifest.json", "dataset.manifest.json",
                           "frozen_partial_manifest.json"):
            manifest_path = root
            root = root.parent
        else:
            raise ValueError(f"source must be a frozen manifest or source directory: {root}")
    else:
        campaign_path = root / "campaign_manifest.json"
        manifest_path = campaign_path if campaign_path.is_file() else None
        if manifest_path is None:
            for name in ("frozen_partial_manifest.json", "manifest.json", "replay_manifest.json",
                         "dataset.manifest.json"):
                candidate = root / name
                if candidate.is_file():
                    manifest_path = candidate
                    break
    if manifest_path is None:
        raise ValueError(f"no frozen accepted-shard manifest found in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    campaign = manifest_path.name == "campaign_manifest.json"
    if campaign:
        status = str(manifest.get("status", ""))
        if "assigned_sources_complete" not in status:
            raise ValueError("campaign is not frozen: assigned sources are incomplete")
        if int(manifest.get("reserve_unassigned", 0)) <= 0:
            raise ValueError("campaign manifest does not retain an unassigned reserve")
        entries = manifest.get("shards")
    else:
        if manifest.get("complete") is not True:
            raise ValueError("generic source manifest must declare complete=true")
        entries = manifest.get("accepted_shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("frozen source manifest has no accepted shards")
    shards = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("accepted shard entry must be an object")
        binary_name = entry.get("v3") or entry.get("bin") or entry.get("path")
        if not isinstance(binary_name, str):
            raise ValueError("accepted shard lacks a V3 binary path")
        binary = _safe_child(root, binary_name)
        meta_name = entry.get("meta_path") or entry.get("meta")
        if isinstance(meta_name, dict):
            meta_name = meta_name.get("path")
        if isinstance(meta_name, str):
            meta_candidate = _safe_child(root, meta_name)
            if meta_candidate.is_file():
                meta = meta_candidate
            elif Path(meta_name).name == meta_name:
                meta = _safe_child(root, str(binary.relative_to(root).parent / meta_name))
            else:
                meta = meta_candidate
        else:
            meta = binary.with_suffix(".meta")
        bin_sha = entry.get("bin_sha256") or entry.get("v3_sha256") or entry.get("sha256")
        meta_sha = entry.get("meta_sha256")
        if not isinstance(bin_sha, str) or not isinstance(meta_sha, str):
            raise ValueError(f"accepted shard lacks binary/metadata SHA provenance: {binary_name}")
        if not binary.is_file() or _sha(binary) != bin_sha:
            raise ValueError(f"accepted V3 shard missing or SHA mismatch: {binary}")
        if not meta.is_file() or _sha(meta) != meta_sha:
            raise ValueError(f"accepted metadata shard missing or SHA mismatch: {meta}")
        if binary.stat().st_size % SAMPLE_DTYPE.itemsize:
            raise ValueError(f"V3 shard is not 64-byte aligned: {binary}")
        if meta.stat().st_size % META_DTYPE.itemsize:
            raise ValueError(f"metadata shard is not 20-byte aligned: {meta}")
        if binary.stat().st_size // SAMPLE_DTYPE.itemsize != meta.stat().st_size // META_DTYPE.itemsize:
            raise ValueError(f"V3 and metadata counts are not aligned: {binary}")
        shards.append({"path": binary, "meta_path": meta, "bin_sha256": bin_sha,
                       "meta_sha256": meta_sha, "source": str(entry.get("source", root.name)),
                       "seed": entry.get("seed"), "entry": entry})
    return {"root": root, "manifest_path": manifest_path, "manifest": manifest,
            "campaign": campaign, "shards": shards}


def _policy_rows(rows: np.ndarray):
    indices = rows["policy_top_idx"].astype(np.int64)
    visits = rows["policy_top_prob"].astype(np.float32)
    totals = visits.sum(axis=1)
    good = totals > 0
    if not good.any():
        return np.empty((0, POLICY_SIZE), dtype=np.float32), good
    idx, probs = indices[good], visits[good]
    if np.any((probs > 0) & ((idx < 0) | (idx >= POLICY_SIZE))):
        raise ValueError("stored policy contains an out-of-range action index")
    dense = np.zeros((len(idx), POLICY_SIZE), dtype=np.float32)
    row_ids = np.arange(len(idx), dtype=np.int64)
    for column in range(idx.shape[1]):
        active = probs[:, column] > 0
        np.add.at(dense, (row_ids[active], idx[active, column]), probs[active, column])
    dense /= totals[good, None]
    return dense, good


def _target(root, result, plies, gamma, outcome_weight):
    signed_root = 2.0 * root - 1.0
    return np.clip((1.0 - outcome_weight) * signed_root
                   + outcome_weight * np.power(gamma, plies) * result, -1.0, 1.0)


def _state_key(row) -> bytes:
    return struct.pack("<BBQQbb", int(row["own_pawn"]), int(row["opp_pawn"]),
                       int(row["walls_h"]), int(row["walls_v"]),
                       int(row["walls_left_own"]), int(row["walls_left_opp"]))


def _partition(seed: int, game_token: str, val_fraction: float) -> tuple[bool, str]:
    token = hashlib.sha256(f"{seed}:{game_token}".encode()).hexdigest()
    val = int(token, 16) < int(val_fraction * (1 << 256))
    return val, token


def _opening_bank(source_info):
    """Index opening-bank line offsets without retaining its rows in RAM."""
    manifest = source_info["manifest"]
    if source_info["campaign"]:
        path = source_info["root"] / "opening_bank_8to10ply.jsonl"
        expected = manifest.get("bank_sha256")
    else:
        info = manifest.get("opening_bank")
        if not isinstance(info, dict) or not info.get("available"):
            return None
        path = _safe_child(source_info["root"], str(info.get("path", "")))
        expected = info.get("sha256")
    if not path.is_file():
        return None
    if expected and _sha(path) != expected:
        raise ValueError(f"opening-bank SHA mismatch: {path}")
    offsets = array("Q")
    with path.open("rb") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                offsets.append(position)
    if not offsets:
        return None
    return {"path": path, "offsets": offsets,
            "sha256": expected or _sha(path)}


def _opening_family(bank, game_id, shard, source_info):
    if bank is None:
        return "unknown"
    games_per_shard = source_info["manifest"].get("games_per_shard")
    shard_seed = shard.get("seed")
    if games_per_shard is None or shard_seed is None:
        return "unknown"
    local_game = int(game_id) % int(games_per_shard)
    index = ((int(shard_seed) + local_game * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)) % len(bank["offsets"])
    with bank["path"].open("rb") as stream:
        stream.seek(bank["offsets"][index])
        record = json.loads(stream.readline())
    label = record.get("opening_index")
    if label is None:
        label = record.get("source")
    return str(label) if label is not None else "unknown"


def _replay_manifest(dataset_manifest, manifest_path, out_dir, chunk_size):
    return {"schema": "zquoridor.stored_replay_manifest.v1", "complete": True,
            "config": {"seed": dataset_manifest["seed"],
                       "val_fraction": dataset_manifest["val_fraction"],
                       "mode": "stored_search", "stored_gamma": 1.0,
                       "stored_outcome_weight": 0.5, "max_positions": None,
                       "chunk_size": int(chunk_size)},
            "dataset_manifest_sha256": _sha(manifest_path),
            "dataset_identity_sha256": dataset_manifest["identity_sha256"],
            "dataset": str(out_dir.resolve()),
            "provenance": {"sources": dataset_manifest["sources"],
                           "shards": dataset_manifest["shards"]},
            "counts": {key: dataset_manifest[key] for key in (
                "raw_records", "accepted_records", "skipped_missing_root", "skipped_zero_visits",
                "duplicate_records_merged", "cross_split_conflict_states_removed",
                "cross_split_conflict_records_removed", "samples", "train_samples", "val_samples")}}


def _write_npy_arrays(db: sqlite3.Connection, fields_dir: Path, chunk_size: int, source_names):
    total = db.execute("SELECT COUNT(*) FROM states WHERE split_mask IN (1,2)").fetchone()[0]
    train = db.execute("SELECT COUNT(*) FROM states WHERE split_mask=1").fetchone()[0]
    val = db.execute("SELECT COUNT(*) FROM states WHERE split_mask=2").fetchone()[0]
    if total < 2 or train < 1 or val < 1:
        raise ValueError("deterministic game split has insufficient states")
    fields_dir.mkdir(parents=True, exist_ok=True)
    shapes = {
        "own_pawn": (np.uint8, (total,)), "opp_pawn": (np.uint8, (total,)),
        "walls_h": (np.uint64, (total,)), "walls_v": (np.uint64, (total,)),
        "walls_left_own": (np.int8, (total,)), "walls_left_opp": (np.int8, (total,)),
        "own_dist": (np.uint8, (total,)), "opp_dist": (np.uint8, (total,)),
        "policy": (np.float32, (total, POLICY_SIZE)), "value": (np.float32, (total,)),
        "weight": (np.float32, (total,)), "is_val": (np.bool_, (total,)),
        "group_id": ("S64", (total,)), "stored_root": (np.float32, (total,)),
        "game_result": (np.float32, (total,)), "duplicate_count": (np.int32, (total,)),
        "network_source_count": (np.int32, (total, len(source_names))),
    }
    arrays = {name: np.lib.format.open_memmap(fields_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (dtype, shape) in shapes.items()}
    src_index = {name: i for i, name in enumerate(source_names)}
    source_rows = iter(db.execute("SELECT key,source_name,count FROM source_roots ORDER BY key,source_name"))
    source_row = next(source_rows, None)
    cursor = db.execute("SELECT key,own_dist,opp_dist,value_sum,root_sum,result_sum,count,split_mask,first_group,policy_sum "
                        "FROM states WHERE split_mask IN (1,2) ORDER BY key")
    offset = 0
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        end = offset + len(rows)
        policies = np.empty((len(rows), POLICY_SIZE), dtype=np.float32)
        for j, row in enumerate(rows):
            key, od, pd, vsum, rsum, gsum, count, is_val, group, psum = row
            if od > 81 or pd > 81:
                raise ValueError("stored shard contains invalid canonical distances")
            own_pawn, opp_pawn, walls_h, walls_v, own_walls, opp_walls = struct.unpack("<BBQQbb", key)
            for field, value in (("own_pawn", own_pawn), ("opp_pawn", opp_pawn),
                                 ("walls_h", walls_h), ("walls_v", walls_v),
                                 ("walls_left_own", own_walls), ("walls_left_opp", opp_walls),
                                 ("own_dist", od), ("opp_dist", pd)):
                arrays[field][offset + j] = value
            arrays["value"][offset + j] = float(vsum) / count
            arrays["stored_root"][offset + j] = float(rsum) / count
            arrays["game_result"][offset + j] = float(gsum) / count
            arrays["duplicate_count"][offset + j] = count
            arrays["is_val"][offset + j] = int(is_val) == 2
            arrays["group_id"][offset + j] = str(group).encode("ascii")
            arrays["weight"][offset + j] = 1.0
            policy = np.frombuffer(psum, dtype=np.float32).copy() / count
            mass = float(policy.sum())
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("deduplicated policy has no positive mass")
            policies[j] = policy / mass
            arrays["policy"][offset + j] = policies[j]
            arrays["network_source_count"][offset + j] = 0
            # Source rows also include states excluded for split conflicts.
            while source_row is not None and source_row[0] < key:
                source_row = next(source_rows, None)
            while source_row is not None and source_row[0] == key:
                arrays["network_source_count"][offset + j, src_index[source_row[1]]] = source_row[2]
                source_row = next(source_rows, None)
        offset = end
    if offset != total:
        raise RuntimeError("SQLite export row count changed")
    for array in arrays.values():
        array.flush()
    return arrays, {"samples": total, "train_samples": train, "val_samples": val,
                    "network_source_names": source_names}


def prepare(source, out_dir, seed=20261001, val_fraction=0.2, chunk_size=8192):
    """Prepare a complete frozen stored-search corpus without loading it in RAM."""
    if not 0 < val_fraction < 1 or chunk_size < 1:
        raise ValueError("val_fraction must be in (0,1) and chunk_size positive")
    sources = list(source) if isinstance(source, (list, tuple)) else [source]
    if not sources:
        raise ValueError("at least one frozen source manifest is required")
    resolved = [_resolve_source(item) for item in sources]
    shard_entries = []
    for source_index, item in enumerate(resolved):
        for shard in item["shards"]:
            shard = dict(shard)
            shard["source_index"] = source_index
            shard_entries.append(shard)
    identity = {"seed": int(seed), "val_fraction": float(val_fraction), "gamma": 1.0,
                "outcome_weight": 0.5,
                "sources": [{"manifest": str(item["manifest_path"]),
                             "manifest_sha256": _sha(item["manifest_path"]),
                             "campaign": item["campaign"]} for item in resolved],
                "shards": [{"path": str(s["path"]), "bin_sha256": s["bin_sha256"],
                            "meta_path": str(s["meta_path"]), "meta_sha256": s["meta_sha256"],
                            "source": s["source"]} for s in shard_entries]}
    identity_sha = _json_digest(identity)
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "dataset.manifest.json"
    replay_path = out_dir.parent / "replay_manifest.json"
    if replay_path.exists():
        replay_old = json.loads(replay_path.read_text(encoding="utf-8"))
        if replay_old.get("dataset_identity_sha256") not in (None, identity_sha):
            raise ValueError("sibling replay_manifest.json belongs to a different dataset identity")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not existing.get("complete") or existing.get("identity_sha256") != identity_sha:
            raise ValueError("completed output exists with a different or incomplete identity")
        for name, info in existing.get("arrays", {}).items():
            output = out_dir / info["filename"]
            if not output.is_file() or _sha(output) != info["sha256"]:
                raise ValueError(f"output array SHA mismatch: {name}")
        if replay_path.exists():
            replay_old = json.loads(replay_path.read_text(encoding="utf-8"))
            if replay_old.get("dataset_manifest_sha256") != _sha(manifest_path):
                raise ValueError("sibling replay_manifest.json does not match the completed dataset manifest")
        else:
            _atomic_json(replay_path, _replay_manifest(existing, manifest_path, out_dir, chunk_size))
        return existing
    if any(p.name != ".input_cache" for p in out_dir.iterdir()):
        raise FileExistsError(f"output directory contains partial data: {out_dir}")

    db_path = out_dir / ".prepare.sqlite"
    db = sqlite3.connect(db_path)
    db.create_function("policy_add", 2, lambda a, b: (np.frombuffer(a, np.float32) + np.frombuffer(b, np.float32)).tobytes())
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
        CREATE TABLE states(
            key BLOB PRIMARY KEY, split_mask INTEGER NOT NULL,
            own_dist INTEGER NOT NULL, opp_dist INTEGER NOT NULL,
            distance_conflict INTEGER NOT NULL DEFAULT 0,
            value_sum REAL NOT NULL, root_sum REAL NOT NULL, result_sum REAL NOT NULL,
            count INTEGER NOT NULL, first_group TEXT NOT NULL, policy_sum BLOB NOT NULL);
        CREATE TABLE source_roots(
            key BLOB NOT NULL, source_name TEXT NOT NULL, root_sum REAL NOT NULL,
            count INTEGER NOT NULL, PRIMARY KEY(key,source_name));
        CREATE TABLE games(token TEXT PRIMARY KEY, is_val INTEGER NOT NULL);
    """)
    skipped_missing = skipped_zero_visits = accepted = total_raw = 0
    shard_reports = []
    source_raw = {}
    profile_counts, rollout_ply_counts, opening_family_counts = {}, {}, {}
    opening_banks = [_opening_bank(item) for item in resolved]
    opening_bank_provenance = [
        {"available": bank is not None, "sha256": bank["sha256"] if bank else None,
         "path": str(bank["path"]) if bank else None} for bank in opening_banks]
    alpha, gamma = 0.5, 1.0
    try:
        for shard in shard_entries:
            data = np.memmap(shard["path"], dtype=SAMPLE_DTYPE, mode="r")
            meta = np.memmap(shard["meta_path"], dtype=META_DTYPE, mode="r")
            source_name = shard["source"]
            source_info = resolved[shard["source_index"]]
            profile = str(shard["entry"].get("profile") or "unknown")
            source_raw.setdefault(source_name, 0)
            valid_shard, shard_missing, shard_zero = 0, 0, 0
            previous_game = None
            current_is_val = False
            current_group = ""
            for start in range(0, len(data), chunk_size):
                stop = min(len(data), start + chunk_size)
                record_chunk = data[start:stop]
                meta_chunk = meta[start:stop]
                roots = meta_chunk["root"].astype(np.float64)
                flags = meta_chunk["flags"].astype(np.uint8)
                missing = (flags & META_ROOT_MISSING) != 0
                shard_missing += int(missing.sum())
                root_valid = (flags & META_ROOT_VALID) != 0
                nonmissing = ~missing
                if np.any(nonmissing & root_valid & ~np.isfinite(roots)):
                    raise ValueError(f"flagged root value is nonfinite in {shard['path']}")
                if np.any(nonmissing & ~root_valid):
                    raise ValueError(f"nonmissing root value is not flagged valid in {shard['path']}")
                if np.any(nonmissing & ((roots < 0) | (roots > 1))):
                    raise ValueError(f"root value is outside [0,1] in {shard['path']}")
                if np.any(nonmissing & (meta_chunk["length"] > 0)
                          & (meta_chunk["plies"] > meta_chunk["length"])):
                    raise ValueError(f"remaining plies exceed game length in {shard['path']}")
                policies, visits_ok = _policy_rows(record_chunk)
                eligible = nonmissing & visits_ok
                shard_zero += int((nonmissing & ~visits_ok).sum())
                rows_i = np.flatnonzero(eligible)
                if not len(rows_i):
                    continue
                eligible_rows = record_chunk[rows_i]
                eligible_meta = meta_chunk[rows_i]
                eligible_roots = roots[rows_i]
                eligible_policies = policies[np.searchsorted(np.flatnonzero(visits_ok), rows_i)]
                for field, maximum in (("own_pawn", 80), ("opp_pawn", 80),
                                       ("walls_left_own", 10), ("walls_left_opp", 10),
                                       ("own_dist", 81), ("opp_dist", 81)):
                    if np.any(eligible_rows[field] > maximum):
                        raise ValueError(f"invalid {field} in V3 shard {shard['path']}")
                if np.any(eligible_rows["walls_left_own"] < 0) or np.any(eligible_rows["walls_left_opp"] < 0):
                    raise ValueError(f"invalid remaining wall count in V3 shard {shard['path']}")
                results = eligible_rows["game_result"].astype(np.float64)
                if np.any((results < -1) | (results > 1)):
                    raise ValueError(f"invalid terminal game result in {shard['path']}")
                targets = _target(eligible_roots, results, eligible_meta["plies"], gamma, alpha)

                keys, is_vals, group_ids = [], [], []
                for meta_row in eligible_meta:
                    game = int(meta_row["game"])
                    if game != previous_game:
                        source_manifest = source_info
                        if source_manifest["campaign"]:
                            game_token = (f"{source_manifest['root']}:{game}"
                                          if len(resolved) > 1 else str(game))
                        else:
                            game_token = f"{source_name}:{shard['bin_sha256']}:{game}"
                        current_is_val, game_hash = _partition(int(seed), game_token, val_fraction)
                        current_group = game_hash
                        family = _opening_family(opening_banks[shard["source_index"]], game, shard, source_info)
                        opening_family_counts[family] = opening_family_counts.get(family, 0) + 1
                        previous_game = game
                        db.execute("INSERT OR IGNORE INTO games(token,is_val) VALUES(?,?)",
                                   (game_token, int(current_is_val)))
                        previous = db.execute("SELECT is_val FROM games WHERE token=?", (game_token,)).fetchone()[0]
                        if bool(previous) != bool(current_is_val):
                            raise RuntimeError("deterministic game assignment changed for repeated token")
                    is_vals.append(2 if current_is_val else 1)
                    group_ids.append(current_group)
                batch = []
                for j, row in enumerate(eligible_rows):
                    keys.append(_state_key(row))
                    batch.append((keys[-1], is_vals[j], int(row["own_dist"]), int(row["opp_dist"]),
                                  float(targets[j]), float(eligible_roots[j]), float(results[j]), 1,
                                  group_ids[j], eligible_policies[j].astype(np.float32, copy=False).tobytes()))
                db.executemany("""
                    INSERT INTO states(key,split_mask,own_dist,opp_dist,value_sum,root_sum,result_sum,count,first_group,policy_sum)
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET
                      split_mask=states.split_mask | excluded.split_mask,
                      distance_conflict=MAX(states.distance_conflict,
                          states.own_dist != excluded.own_dist OR states.opp_dist != excluded.opp_dist),
                      value_sum=states.value_sum+excluded.value_sum,
                      root_sum=states.root_sum+excluded.root_sum,
                      result_sum=states.result_sum+excluded.result_sum,
                      count=states.count+1,
                      policy_sum=policy_add(states.policy_sum,excluded.policy_sum)
                """, batch)
                source_batch = [(keys[j], source_name, float(eligible_roots[j]), 1) for j in range(len(keys))]
                db.executemany("""INSERT INTO source_roots(key,source_name,root_sum,count) VALUES(?,?,?,?)
                    ON CONFLICT(key,source_name) DO UPDATE SET root_sum=root_sum+excluded.root_sum,count=count+1""",
                    source_batch)
                accepted += len(batch)
                valid_shard += len(batch)
                source_raw[source_name] += len(batch)
                profile_counts[profile] = profile_counts.get(profile, 0) + len(batch)
                plies_left = eligible_meta["length"].astype(np.int64) - eligible_meta["plies"].astype(np.int64)
                for ply, count in zip(*np.unique(plies_left, return_counts=True)):
                    ply_key = str(int(ply))
                    rollout_ply_counts[ply_key] = rollout_ply_counts.get(ply_key, 0) + int(count)
                db.commit()
            del data, meta
            total_raw += int(shard["path"].stat().st_size // SAMPLE_DTYPE.itemsize)
            shard_reports.append({"path": str(shard["path"]), "meta": str(shard["meta_path"]),
                                  "bin_sha256": shard["bin_sha256"], "meta_sha256": shard["meta_sha256"],
                                  "source": source_name, "records": int(shard["path"].stat().st_size // SAMPLE_DTYPE.itemsize),
                                  "profile": profile,
                                  "accepted_records": valid_shard, "skipped_missing_root": shard_missing,
                                  "skipped_zero_visits": shard_zero})
            skipped_missing += shard_missing
            skipped_zero_visits += shard_zero
            progress = dict(status="aggregating", shards_completed=len(shard_reports),
                            shards_total=len(shard_entries), raw_records_processed=total_raw,
                            accepted_records=accepted, skipped_missing_root=skipped_missing,
                            skipped_zero_visits=skipped_zero_visits,
                            last_shard=str(shard["path"]), identity_sha256=identity_sha)
            _atomic_json(out_dir / "prepare_progress.json", progress)
            print(json.dumps(progress), flush=True)
        conflicts = db.execute("SELECT COUNT(*) FROM states WHERE split_mask=3").fetchone()[0]
        bad_distances = db.execute("SELECT COUNT(*) FROM states WHERE distance_conflict=1").fetchone()[0]
        if bad_distances:
            raise ValueError(f"{bad_distances} canonical states have inconsistent distances")
        if db.execute("SELECT COUNT(*) FROM games WHERE is_val=0").fetchone()[0] == 0:
            raise ValueError("no training games in deterministic split")
        if db.execute("SELECT COUNT(*) FROM games WHERE is_val=1").fetchone()[0] == 0:
            raise ValueError("no validation games in deterministic split")
        eligible_states = db.execute("SELECT COUNT(*) FROM states WHERE split_mask IN (1,2)").fetchone()[0]
        if eligible_states < 2:
            raise ValueError("insufficient nonconflicting stored-search states")
        cross_split_rows = db.execute("SELECT COALESCE(SUM(count),0) FROM states WHERE split_mask=3").fetchone()[0]
        game_counts = {
            "total": int(db.execute("SELECT COUNT(*) FROM games").fetchone()[0]),
            "train": int(db.execute("SELECT COUNT(*) FROM games WHERE is_val=0").fetchone()[0]),
            "validation": int(db.execute("SELECT COUNT(*) FROM games WHERE is_val=1").fetchone()[0]),
        }
        source_names = sorted(source_raw)
        fields_dir = out_dir / "fields"
        arrays, counts = _write_npy_arrays(db, fields_dir, chunk_size, source_names)
        arrays_manifest = {}
        for name, array in arrays.items():
            path = fields_dir / f"{name}.npy"
            arrays_manifest[name] = {"filename": str(Path("fields") / path.name), "sha256": _sha(path),
                                     "shape": list(array.shape), "dtype": str(array.dtype)}
        # Per-source root agreement, accumulated from the disk table only.
        shared, disagreement, pairs = 0, 0.0, 0
        current_key, roots_for_state = None, []
        for key, source_name, root_sum, count in db.execute(
                "SELECT r.key,r.source_name,r.root_sum,r.count FROM source_roots r "
                "JOIN states s ON s.key=r.key WHERE s.split_mask IN (1,2) ORDER BY r.key,r.source_name"):
            if current_key is not None and key != current_key:
                if len(roots_for_state) > 1:
                    shared += 1
                    means = [total / n for total, n in roots_for_state]
                    for i in range(len(means)):
                        for j in range(i + 1, len(means)):
                            disagreement += abs(means[i] - means[j])
                            pairs += 1
                roots_for_state = []
            current_key = key
            roots_for_state.append((root_sum, count))
        if len(roots_for_state) > 1:
            shared += 1
            means = [total / n for total, n in roots_for_state]
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    disagreement += abs(means[i] - means[j])
                    pairs += 1
        del arrays
        db.close()
        db_path.unlink(missing_ok=True)
        manifest = {"schema": SCHEMA, "complete": True, "identity_sha256": identity_sha,
                    "group_separation_verified": True, "seed": int(seed), "val_fraction": float(val_fraction),
                    "mode": "stored_search", "value_target": "0.5*(2*root-1)+0.5*game_result; gamma=1",
                    "stored_gamma": 1.0, "stored_outcome_weight": 0.5,
                    "deduplication": "canonical_state_weighted_mean_all_eligible_records",
                    "all_records_used_except_invalid_and_cross_split_conflicts": True,
                    "max_positions": None,
                    "samples": counts["samples"], "train_samples": counts["train_samples"],
                    "val_samples": counts["val_samples"], "raw_records": total_raw,
                    "accepted_records": accepted, "skipped_missing_root": skipped_missing,
                    "skipped_zero_visits": skipped_zero_visits,
                    "duplicate_records_merged": accepted - int(eligible_states + conflicts),
                    "cross_split_conflict_states_removed": int(conflicts),
                    "cross_split_conflict_records_removed": int(cross_split_rows),
                    "games": game_counts,
                    "source_names": source_names, "source_record_counts": source_raw,
                    "shared_states_across_sources": shared,
                    "mean_pairwise_root_disagreement": disagreement / pairs if pairs else None,
                    "root_disagreement_pairs": pairs, "sources": identity["sources"],
                    "profile_counts": profile_counts, "rollout_ply_counts": rollout_ply_counts,
                    "opening_family_counts": opening_family_counts or {"unknown": 0},
                    "opening_family_provenance": opening_bank_provenance,
                    "shards": shard_reports, "arrays": arrays_manifest}
        # SQLite was closed above; query game split totals from the exported group IDs.
        _atomic_json(manifest_path, manifest)
        _atomic_json(replay_path, _replay_manifest(manifest, manifest_path, out_dir, chunk_size))
        _atomic_json(out_dir / "prepare_progress.json", dict(
            status="complete", shards_completed=len(shard_reports), shards_total=len(shard_entries),
            raw_records_processed=total_raw, accepted_records=accepted,
            unique_states=counts["samples"], identity_sha256=identity_sha))
        return manifest
    except Exception:
        db.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=CONFIG["source"])
    parser.add_argument("--out-dir", default=CONFIG["out_dir"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--val-fraction", type=float, default=CONFIG["val_fraction"])
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"])
    args = parser.parse_args(argv)
    result = prepare(args.source, args.out_dir, args.seed, args.val_fraction, args.chunk_size)
    print(json.dumps({k: result[k] for k in ("samples", "train_samples", "val_samples", "identity_sha256")}, indent=2))


if __name__ == "__main__":
    main()
