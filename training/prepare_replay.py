#!/usr/bin/env python3
"""Build a replay dataset from existing canonical self-play shards.

Input:
    ``data/selfplay_canonical_v3/<generation>/selfplay_*.bin`` (64-byte V3
    records), plus the old NNUE weights and the pinned Claustrophobia
    checkpoint. With ``--mode stored_search``, each V3 shard must instead
    have an aligned 20-byte ``.meta`` sidecar; stored MCAB visits and root
    values are consumed directly and no teacher inference runs.

Output:
    ``<out-dir>/dataset.npz`` with train/validation splits and blended policy
    and value targets.  Resumable ``direct_*.npz`` caches, SHA-256 files, and
    ``replay_manifest.json`` are written beside it.  This script never creates
    self-play games, runs search, or trains a student network.

The configuration block is the default; every field can be overridden with a
matching CLI option.  Use a new output directory when inputs or settings
change.

The default ``--mode inference`` behavior is unchanged.  Stored replay uses
``--stored-gamma`` for symmetric terminal-result discount and
``--stored-outcome-weight`` to blend it with the signed root value. It
aggregates duplicate canonical states across games by averaging their visit
policies and blended value targets without selection bias. It assigns game IDs
to train or validation. The stored mode shuffles shard order before its bounded
scan. The resulting sample is reproducible but is not globally uniform.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import deque
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "training/teachers"))
from read_selfplay import _detect_format
from student_model import Student, encode_features
from teachers.claustrophobia_relabel import encode_positions
from teachers.claustrophobia_inference_worker import infer
from tools.external.bot_setup import ensure_bot

CONFIG = {
    "source": str(ROOT / "data/selfplay_canonical_v3"),
    "out_dir": str(ROOT / "data/teaching/replay"),
    "max_positions": 50000,
    "seed": 20260914,
    "val_fraction": 0.2,
    "chunk_size": 8192,
    "batch_size": 4096,
    "device": "auto",
    "old_nnue": str(ROOT / "checkpoints/gen9-teacher-head/nnue_weights.bin"),
    "old_policy_weight": 0.5,
    "claustro_policy_weight": 0.5,
    "old_value_weight": 0.75,
    "claustro_value_weight": 0.25,
    "outcome_weight": 0.0,
    "resume": True,
    # Legacy inference remains the default.  stored_search consumes aligned
    # V3/.meta pairs and never invokes either teacher.
    "mode": "inference",
    "stored_gamma": 0.99,
    "stored_outcome_weight": 0.5,
}
STATE_FIELDS = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own", "walls_left_opp")
POLICY_DIM = 209
# Names match the controller's public META_DTYPE while the packed layout
# matches TrainingMetaV1 exactly.
META_DTYPE = np.dtype([("game", "<u8"), ("root", "<f4"),
                       ("plies", "<u2"), ("length", "<u2"),
                       ("source", "u1"), ("flags", "u1"),
                       ("reserved", "<u2")])
METADATA_DTYPE = META_DTYPE
META_ROOT_MISSING = 1
META_ROOT_VALID = 2


def normalize_visits(indices, visits) -> np.ndarray:
    """Expand a V3 top-eight visit distribution into a dense policy target."""
    result = np.zeros(POLICY_DIM, dtype=np.float32)
    indices = np.asarray(indices)
    visits = np.asarray(visits, dtype=np.float64)
    if indices.shape != visits.shape or indices.ndim != 1:
        raise ValueError("stored visits and indices must be one-dimensional and aligned")
    positive = visits > 0
    if (not np.isfinite(visits).all() or np.any(visits < 0)
            or np.any(indices[positive] < 0) or np.any(indices[positive] >= POLICY_DIM)):
        raise ValueError("stored visits must be finite, non-negative, and use legal policy indices")
    total = float(visits.sum())
    if total <= 0:
        raise ValueError("stored visits must contain positive mass")
    for index, visit in zip(indices, visits):
        if visit > 0:
            result[int(index)] += float(visit)
    result /= total
    return result


def stored_value_target(root_value: float, terminal_result: float, remaining_plies: int,
                        gamma: float, outcome_weight: float = 0.5) -> float:
    """Blend signed root search value with discounted terminal evidence."""
    root = float(root_value)
    if not math.isfinite(root) or not 0.0 <= root <= 1.0:
        raise ValueError("stored root value must be finite and in [0,1]")
    res = float(terminal_result)
    if not math.isfinite(res) or not -1.0 <= res <= 1.0:
        raise ValueError("stored terminal result must be finite and in [-1, 1]")
    plies = int(remaining_plies)
    if plies != remaining_plies or plies < 0:
        raise ValueError("stored remaining plies must be a non-negative integer")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("stored gamma must be finite and in [0,1]")
    alpha = float(outcome_weight)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("stored outcome weight must be finite and in [0,1]")
    root_signed = 2.0 * root - 1.0
    return float(np.clip((1.0 - alpha) * root_signed
                         + alpha * (gamma ** plies) * res, -1.0, 1.0))


def legal_wall_topology(walls_h: int, walls_v: int, own_pawn: int, opp_pawn: int) -> bool:
    """Validate a recorded board before sending it to the Claustrophobia bridge.

    Historical shards may contain a few corrupt wall sets.  The direct bridge
    correctly rejects them, but filtering at sampling time lets a million-row
    relabel run replace the bad row instead of stopping after hours.  A wall
    crossing uses the same anchor in both bitboards; both pawns must retain a
    path to their respective goal rows.
    """
    walls_h, walls_v = int(walls_h), int(walls_v)
    if walls_h & walls_v:
        return False

    def blocked(row, col, next_row, next_col):
        if row == next_row:
            anchor_row, anchor_col = row, min(col, next_col)
            return ((anchor_row > 0 and (walls_v >> ((anchor_row - 1) * 8 + anchor_col)) & 1)
                    or (anchor_row < 8 and (walls_v >> (anchor_row * 8 + anchor_col)) & 1))
        anchor_row, anchor_col = min(row, next_row), col
        return ((anchor_col > 0 and (walls_h >> (anchor_row * 8 + anchor_col - 1)) & 1)
                or (anchor_col < 8 and (walls_h >> (anchor_row * 8 + anchor_col)) & 1))

    def has_path(cell, goal_row):
        queue, seen = deque([cell]), {cell}
        while queue:
            current = queue.popleft()
            row, col = divmod(current, 9)
            if row == goal_row:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                nxt = nr * 9 + nc
                if 0 <= nr < 9 and 0 <= nc < 9 and nxt not in seen and not blocked(row, col, nr, nc):
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    return has_path(int(own_pawn), 8) and has_path(int(opp_pawn), 0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_states(config, blocked=()):
    files = sorted(Path(config["source"]).rglob("*.bin"))
    valid, skipped = [], []
    for path in files:
        dtype, size = _detect_format(path)
        if size == 27:
            skipped.append(str(path))
        else:
            valid.append((path, dtype, size))
    if len(valid) < 2:
        raise ValueError("replay requires at least two canonical 32/64-byte shards; ambiguous 27-byte data is excluded")
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(valid)
    n_val = max(1, min(len(valid)-1, round(len(valid)*config["val_fraction"])))
    quota = math.ceil(config["max_positions"] / len(valid))
    rows, ids, groups, splits, provenance = [], [], [], [], []
    seen = set(blocked)
    unique_counts = {}
    result_sums = {}
    for file_index, (path, dtype, size) in enumerate(valid):
        digest = sha(path)
        provenance.append(dict(path=str(path.resolve()), sha256=digest, record_bytes=size))
        data = np.memmap(path, dtype=dtype, mode="r")
        accepted = 0
        # Use a per-shard random permutation rather than a fixed oversample.
        # Small runs still stop as soon as their quota is filled, while a large
        # run can consume the long tail of a shard when duplicate openings or
        # holdouts reject more rows than the former 4x bound allowed.
        candidates = rng.permutation(len(data))
        for index in candidates:
            row = data[index]
            key = tuple(int(row[name]) for name in STATE_FIELDS)
            if key in seen:
                if key in unique_counts:
                    unique_counts[key] += 1
                    result_sums[key] += int(row["game_result"])
                continue
            if (row["own_pawn"] >= 72 or row["opp_pawn"] <= 8
                    or not (0 <= row["walls_left_own"] <= 10)
                    or not (0 <= row["walls_left_opp"] <= 10)
                    or not legal_wall_topology(row["walls_h"], row["walls_v"], row["own_pawn"], row["opp_pawn"])):
                continue
            seen.add(key)
            unique_counts[key] = 1
            result_sums[key] = int(row["game_result"])
            rows.append(tuple(int(row[name]) for name in (*STATE_FIELDS, "own_dist", "opp_dist")))
            ids.append(hashlib.sha256(f"{digest}:{index}".encode()).hexdigest()[:24])
            groups.append(digest)
            splits.append(file_index < n_val)
            accepted += 1
            if accepted >= quota or len(rows) >= config["max_positions"]:
                break
        del data
        if len(rows) >= config["max_positions"]:
            break
    # Some shards contain fewer eligible, distinct positions than the uniform
    # quota. Borrow the shortfall from the training shards first, then from
    # validation shards if necessary. `seen` keeps this second pass disjoint
    # from the balanced first pass and preserves whole-shard group identities.
    if len(rows) < config["max_positions"]:
        for validation in (False, True):
            for file_index, (path, dtype, _size) in enumerate(valid):
                if (file_index < n_val) != validation:
                    continue
                digest = provenance[file_index]["sha256"]
                data = np.memmap(path, dtype=dtype, mode="r")
                for index in rng.permutation(len(data)):
                    row = data[index]
                    key = tuple(int(row[name]) for name in STATE_FIELDS)
                    if key in seen:
                        if key in unique_counts:
                            unique_counts[key] += 1
                            result_sums[key] += int(row["game_result"])
                        continue
                    if (row["own_pawn"] >= 72 or row["opp_pawn"] <= 8
                            or not (0 <= row["walls_left_own"] <= 10)
                            or not (0 <= row["walls_left_opp"] <= 10)
                            or not legal_wall_topology(row["walls_h"], row["walls_v"], row["own_pawn"], row["opp_pawn"])):
                        continue
                    seen.add(key)
                    unique_counts[key] = 1
                    result_sums[key] = int(row["game_result"])
                    rows.append(tuple(int(row[name]) for name in (*STATE_FIELDS, "own_dist", "opp_dist")))
                    ids.append(hashlib.sha256(f"{digest}:{index}".encode()).hexdigest()[:24])
                    groups.append(digest)
                    splits.append(validation)
                    if len(rows) >= config["max_positions"]:
                        break
                del data
                if len(rows) >= config["max_positions"]:
                    break
            if len(rows) >= config["max_positions"]:
                break
    if len(rows) < config["max_positions"]:
        raise ValueError(f"requested {config['max_positions']} positions but found only {len(rows)} distinct eligible samples")
    if len(rows) < 2 or not any(splits) or all(splits):
        raise ValueError("insufficient independent train/validation replay positions")
    names = (*STATE_FIELDS, "own_dist", "opp_dist")
    arrays = {name: np.asarray([r[i] for r in rows], dtype=np.uint64 if name in ("walls_h", "walls_v") else np.int64)
              for i, name in enumerate(names)}
    game_results = np.asarray([float(result_sums[tuple(r[:6])]) / unique_counts[tuple(r[:6])] for r in rows], dtype=np.float32)
    arrays.update(game_result=game_results, id=np.asarray(ids, dtype="S24"), group_id=np.asarray(groups, dtype="S64"),
                  is_val=np.asarray(splits, dtype=bool))
    return arrays, dict(shards=provenance, skipped_legacy_shards=skipped, samples=len(rows))


def sample_stored_states(config, blocked=()):
    """Load V3 states and aligned TrainingMetaV1 sidecars as whole-game groups."""
    source_root = Path(config["source"])
    manifest_path = source_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("accepted_shards")
            if not isinstance(entries, list):
                raise ValueError("managed self-play manifest has no accepted_shards list")
            files = []
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("v3"), str):
                    raise ValueError("managed self-play manifest has an invalid accepted shard")
                path = (source_root / entry["v3"]).resolve()
                if source_root.resolve() not in path.parents:
                    raise ValueError("managed self-play manifest shard leaves source directory")
                files.append(path)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid managed self-play manifest: {manifest_path}") from error
    else:
        # Flat/generic sources have no controller staging tree. Managed
        # corpora always take the manifest branch above.
        files = sorted(source_root.glob("*.bin"))
    if not files:
        raise ValueError("stored_search requires at least one V3 .bin shard")
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(files)
    groups = {}
    provenance = []
    blocked_keys = set(blocked)
    unique_records = {}
    accepted = 0
    skipped_missing = 0
    skipped_no_visits = 0
    duplicates_merged = 0
    stop = False

    gamma = float(config.get("stored_gamma", 0.99))
    outcome_weight = float(config.get("stored_outcome_weight", 0.5))

    for path in files:
        dtype, size = _detect_format(path)
        if size != 64:
            raise ValueError(f"stored_search requires V3 64-byte shards: {path}")
        meta_path = path.with_suffix(".meta")
        if not meta_path.exists():
            raise ValueError(f"stored_search metadata sidecar is missing: {meta_path}")
        if meta_path.stat().st_size % META_DTYPE.itemsize:
            raise ValueError(f"stored_search metadata sidecar is not 20-byte aligned: {meta_path}")
        data = np.memmap(path, dtype=dtype, mode="r")
        metadata = np.memmap(meta_path, dtype=META_DTYPE, mode="r")
        if len(data) != len(metadata):
            raise ValueError(f"stored_search shard and metadata counts differ: {path}")
        path_sha = sha(path)
        meta_sha = sha(meta_path)
        provenance.append(dict(path=str(path.resolve()), meta=str(meta_path.resolve()),
                               sha256=path_sha, meta_sha256=meta_sha, record_bytes=size))
        for index, (row, meta) in enumerate(zip(data, metadata)):
            flags = int(meta["flags"])
            root = float(meta["root"])
            if flags & META_ROOT_MISSING:
                skipped_missing += 1
                continue
            if not math.isfinite(root) or not 0 <= root <= 1:
                raise ValueError(f"stored_search root value is missing or invalid at {path}:{index}")
            if not flags & META_ROOT_VALID:
                raise ValueError(f"stored_search root value is unflagged at {path}:{index}")
            if int(meta["length"]) and int(meta["plies"]) > int(meta["length"]):
                raise ValueError(f"stored_search terminal distance exceeds game length at {path}:{index}")
            key = tuple(int(row[name]) for name in STATE_FIELDS)
            if key in blocked_keys or not legal_wall_topology(row["walls_h"], row["walls_v"], row["own_pawn"], row["opp_pawn"]):
                continue
            visits = np.asarray(row["policy_top_prob"], dtype=np.float64)
            if not np.any(visits > 0):
                skipped_no_visits += 1
                continue
            policy = normalize_visits(row["policy_top_idx"], visits)
            val_target = stored_value_target(root, row["game_result"], meta["plies"], gamma, outcome_weight)

            if key in unique_records:
                rec = unique_records[key]
                rec["count"] += 1
                rec["policy_sum"] += policy
                rec["value_sum"] += val_target
                rec["root_sum"] += root
                rec["plies_sum"] += int(meta["plies"])
                rec["result_sum"] += int(row["game_result"])
                duplicates_merged += 1
                continue

            rec = {
                "row": row.copy(),
                "meta": meta.copy(),
                "index": index,
                "path": path,
                "count": 1,
                "policy_sum": policy.astype(np.float64).copy(),
                "value_sum": float(val_target),
                "root_sum": float(root),
                "plies_sum": int(meta["plies"]),
                "result_sum": int(row["game_result"]),
            }
            unique_records[key] = rec
            game = int(meta["game"])
            groups.setdefault(game, []).append(rec)
            accepted += 1
            if accepted >= config["max_positions"] and len(groups) >= 2:
                stop = True
                break
        del metadata, data
        if stop:
            break
    if len(groups) < 2:
        raise ValueError("stored_search requires at least two independent game groups")
    game_ids = np.asarray(list(groups), dtype=np.uint64)
    rng.shuffle(game_ids)
    n_val = max(1, min(len(game_ids) - 1, round(len(game_ids) * config["val_fraction"])))
    val_games = set(int(x) for x in game_ids[:n_val])
    records, splits = [], []
    for game in game_ids:
        for rec in groups[int(game)]:
            records.append(rec)
            splits.append(int(game) in val_games)
    if len(records) > config["max_positions"]:
        order = rng.permutation(len(records))[:config["max_positions"]]
        records = [records[int(i)] for i in order]
        splits = [splits[int(i)] for i in order]
    if not any(splits) or all(splits):
        raise ValueError("stored_search split must contain train and validation games")

    n_samples = len(records)
    names = (*STATE_FIELDS, "own_dist", "opp_dist")
    file_hashes = {path: item["sha256"] for path, item in zip(files, provenance)}

    policies = np.empty((n_samples, POLICY_DIM), dtype=np.float32)
    values = np.empty(n_samples, dtype=np.float32)
    stored_roots = np.empty(n_samples, dtype=np.float32)
    stored_plies = np.empty(n_samples, dtype=np.int32)
    game_results = np.empty(n_samples, dtype=np.float32)
    policy_top_idx = np.zeros((n_samples, 8), dtype=np.uint16)
    policy_top_prob = np.zeros((n_samples, 8), dtype=np.uint16)

    for i, rec in enumerate(records):
        cnt = rec["count"]
        p_avg = (rec["policy_sum"] / cnt).astype(np.float32)
        p_sum = float(p_avg.sum())
        if p_sum > 0:
            p_avg /= p_sum
        policies[i] = p_avg

        top8 = np.argsort(-p_avg)[:8]
        policy_top_idx[i] = top8.astype(np.uint16)
        policy_top_prob[i] = np.round(p_avg[top8] * 65535).astype(np.uint16)

        values[i] = float(np.clip(rec["value_sum"] / cnt, -1.0, 1.0))
        stored_roots[i] = float(rec["root_sum"] / cnt)
        stored_plies[i] = int(round(rec["plies_sum"] / cnt))
        game_results[i] = float(rec["result_sum"] / cnt)

    arrays = {name: np.asarray([int(rec["row"][name]) for rec in records],
                               dtype=np.uint64 if name in ("walls_h", "walls_v") else np.int64)
              for name in names}
    arrays.update(
        game_result=game_results,
        id=np.asarray([hashlib.sha256(f"{file_hashes[rec['path']]}:{rec['index']}".encode()).hexdigest()[:24]
                       for rec in records], dtype="S24"),
        group_id=np.asarray([str(int(rec["meta"]["game"])) for rec in records], dtype="S64"),
        is_val=np.asarray(splits, dtype=bool),
        policy=policies,
        value=values,
        weight=np.ones(n_samples, dtype=np.float32),
        stored_root=stored_roots,
        stored_plies=stored_plies,
        source_class=np.asarray([int(rec["meta"]["source"]) for rec in records], dtype=np.uint8),
        policy_top_idx=policy_top_idx,
        policy_top_prob=policy_top_prob,
        duplicate_count=np.asarray([rec["count"] for rec in records], dtype=np.int32),
    )
    return arrays, dict(shards=provenance, samples=n_samples, games=len(groups), stored_search=True,
                        duplicates_merged=duplicates_merged,
                        skipped_missing_root=skipped_missing, skipped_zero_visits=skipped_no_visits)


def raw_positions(data, start, stop):
    return [dict(id=data["id"][i].decode(), side_to_move=0,
                 history=["@state", *(str(int(data[key][i])) for key in STATE_FIELDS)])
            for i in range(start, stop)]


def run(config):
    if config.get("mode", "inference") not in ("inference", "stored_search"):
        raise ValueError("mode must be 'inference' or 'stored_search'")
    if config.get("mode") == "stored_search":
        if not math.isfinite(float(config["stored_gamma"])) or not 0.0 <= float(config["stored_gamma"]) <= 1.0:
            raise ValueError("stored_gamma must be finite and in [0,1]")
        if not math.isfinite(float(config["stored_outcome_weight"])) or not 0.0 <= float(config["stored_outcome_weight"]) <= 1.0:
            raise ValueError("stored_outcome_weight must be finite and in [0,1]")
    for key in ("max_positions", "batch_size", "chunk_size"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < config["val_fraction"] < 1:
        raise ValueError("val_fraction must be in (0,1)")
    for key in ("old_policy_weight", "claustro_policy_weight", "old_value_weight", "claustro_value_weight", "outcome_weight"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError("teacher weights must be finite and nonnegative")
    pw = config["old_policy_weight"] + config["claustro_policy_weight"]
    vw = config["old_value_weight"] + config["claustro_value_weight"] + config["outcome_weight"]
    if pw <= 0 or vw <= 0:
        raise ValueError("at least one teacher weight per head must be positive")
    folder = Path(config["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    # Exclude all historical benchmark opening states, including prefixes.
    from teachers.teaching_pipeline import _tool
    from build_teacher_soft import encode_states
    openings = [json.loads(line)["moves"] for line in (ROOT / "tools/external/openings_titanium.jsonl").read_text().splitlines() if line.strip()]
    histories = {tuple(moves[:end]) for moves in openings for end in range(len(moves)+1)}
    blocked_data = encode_states([{"history": list(h)} for h in histories],
                                _tool("teacher_encode_state", "tools/teacher/encode_state.cpp"))
    blocked = {tuple(int(blocked_data[k][i]) for k in STATE_FIELDS) for i in range(len(histories))}
    if config.get("mode") == "stored_search":
        data, provenance = sample_stored_states(config, blocked)
        n = len(data["id"])
        if "policy" not in data:
            data["policy"] = np.asarray([normalize_visits(data["policy_top_idx"][i], data["policy_top_prob"][i]) for i in range(n)], dtype=np.float32)
        if "value" not in data:
            data["value"] = np.asarray([stored_value_target(data["stored_root"][i], data["game_result"][i], data["stored_plies"][i], config["stored_gamma"], config["stored_outcome_weight"]) for i in range(n)], dtype=np.float32)
        if "weight" not in data:
            data["weight"] = np.ones(n, np.float32)
        identity = dict(provenance=provenance, code_sha256=sha(__file__))
        manifest = dict(fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                        code_sha256=identity["code_sha256"],
                        config={k: v for k, v in config.items() if k != "resume"}, provenance=provenance,
                        complete=True, dataset=str((folder / "dataset.npz").resolve()),
                        train_samples=int((~data["is_val"]).sum()), val_samples=int(data["is_val"].sum()))
        manifest_path = folder / "replay_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != manifest["fingerprint"] or previous.get("config") != manifest["config"]:
                raise ValueError("stored replay inputs or settings changed; use a new out_dir")
        np.savez(folder / "dataset.npz", **data)
        manifest["dataset_sha256"] = sha(folder / "dataset.npz")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    data, provenance = sample_states(config, blocked)
    bots = ensure_bot("claustrophobia")
    identity = dict(config={k:v for k,v in config.items() if k != "resume"}, provenance=provenance,
                    old_sha=sha(config["old_nnue"]), claustro_sha=sha(bots["checkpoint"]),
                    encoder_sha=sha(bots["encode_bridge"]), code_sha=sha(__file__),
                    dependencies={str(p.relative_to(ROOT)): sha(p) for p in (
                        ROOT/"training/student_model.py", ROOT/"training/train_teacher_policy.py",
                        ROOT/"training/teachers/claustrophobia_inference_worker.py",
                        ROOT/"training/teachers/claustrophobia_relabel.py")})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = folder / "replay_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] != fingerprint:
            stable = (previous.get("config") == identity["config"]
                      and previous.get("provenance") == identity["provenance"]
                      and previous.get("old_sha") == identity["old_sha"]
                      and previous.get("claustro_sha") == identity["claustro_sha"]
                      and previous.get("encoder_sha") == identity["encoder_sha"])
            if not (config["resume"] and stable):
                raise ValueError("replay inputs changed; use a new out_dir")
            print("Replay code changed; verifying resumable chunks by sample id.", flush=True)
    manifest = dict(fingerprint=fingerprint, **identity, complete=False)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    old = Student("base", 256, qat=True)
    old.load_float(config["old_nnue"])
    old.to(device).eval()
    teacher = torch.jit.load(str(bots["checkpoint"]), map_location=device).eval()
    n = len(data["id"])
    chunks = []
    for start in range(0, n, config["chunk_size"]):
        stop = min(n, start + config["chunk_size"])
        cache = folder / f"direct_{start:08d}.npz"
        checksum = cache.with_suffix(".sha256")
        reuse = False
        if config["resume"] and cache.exists() and checksum.exists() and checksum.read_text().strip() == sha(cache):
            with np.load(cache, allow_pickle=False) as stored:
                chunk = {k:stored[k] for k in stored.files}
            reuse = np.array_equal(chunk["id"], data["id"][start:stop])
        if not reuse:
            planes, legal = encode_positions(raw_positions(data, start, stop), bots["encode_bridge"])
            cp, cv, op, ov = [], [], [], []
            for b in range(0, stop-start, config["batch_size"]):
                end = min(stop-start, b+config["batch_size"])
                p, v = infer(teacher, planes[b:end], legal[b:end], device)
                cp.append(p); cv.append(v)
                x = torch.from_numpy(encode_features(data, np.arange(start+b,start+end), "base")).to(device)
                with torch.inference_mode():
                    v, p = old(x)
                    p = p.masked_fill(~torch.from_numpy(legal[b:end]).to(device), -torch.inf).softmax(1)
                    op.append(p.cpu().numpy()); ov.append((v.sigmoid()*2-1).cpu().numpy())
            chunk = dict(id=data["id"][start:stop], old_policy=np.concatenate(op), old_value=np.concatenate(ov),
                         claustro_policy=np.concatenate(cp), claustro_value=np.concatenate(cv))
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as out:
                np.savez(out, **chunk)
            temporary.replace(cache)
            checksum.write_text(sha(cache)+"\n", encoding="ascii")
        chunks.append(chunk)
        print(f"Replay teaching: {stop}/{n} positions ({device})", flush=True)
    joined = {key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    data["policy"] = (joined["old_policy"]*config["old_policy_weight"] + joined["claustro_policy"]*config["claustro_policy_weight"])/pw
    data["value"] = (joined["old_value"]*config["old_value_weight"] + joined["claustro_value"]*config["claustro_value_weight"] + data["game_result"]*config["outcome_weight"])/vw
    data["weight"] = np.ones(n, np.float32)
    np.savez(folder / "dataset.npz", **data)
    manifest.update(complete=True, dataset=str((folder/"dataset.npz").resolve()),
                    train_samples=int((~data["is_val"]).sum()), val_samples=int(data["is_val"].sum()),
                    dataset_sha256=sha(folder/"dataset.npz"))
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(value)
        parser.add_argument("--"+key.replace("_","-"), **kwargs)
    config = dict(CONFIG, **vars(parser.parse_args(argv)))
    print(json.dumps(run(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
