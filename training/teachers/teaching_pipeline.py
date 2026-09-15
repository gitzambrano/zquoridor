#!/usr/bin/env python3
"""Create a cached multi-teacher policy and value dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = ROOT / "training"
EXTERNAL = ROOT / "tools" / "external"
for module_path in (str(HERE), str(TRAINING), str(EXTERNAL)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from targets import POLICY_DIM, dumps, sample_id  # noqa: E402
from value_targets import (  # noqa: E402
    blend_value_targets,
    discount_game_outcome,
    temper_policy,
    temper_value,
    validate_gamma,
)

CACHE_SCHEMA = "zquoridor.teaching.cache.v1"
TARGET_SCHEMA = "zquoridor.teacher.ensemble.v3"

# Edit this block for a persistent local configuration. Each field also has a
# command-line override for reproducible one-off runs.
CONFIG = {
    "mode": "direct",
    "positions": None,
    "out_dir": ROOT / "data" / "teaching" / "pilot",
    "games": 24,
    "opening_plies": 4,
    "max_plies": 180,
    "max_positions": 512,
    "trajectory_movetime_ms": 30,
    "workers": 2,
    "chunk_size": 128,
    "seed": 20260914,
    "val_fraction": 0.20,
    "gamma": 1.0,
    "outcome_weight": 0.0,
    "bootstrap_weight": 0.0,
    "policy_temperature": 1.0,
    "value_temperature": 1.0,
    "batch_size": 1024,
    "device": "auto",
    "claustro_sims": 512,
    "claustro_cpuct": 1.5,
    "zq_nodes": 100000,
    "zq_time_ms": 0,
    "zq_leaf_depth": 0,
    "old_nnue": ROOT / "data" / "nnue" / "nnue_weights.bin",
    "zq_nnue": ROOT / "data" / "nnue" / "nnue_weights_int8.bin",
    "claustro_checkpoint": None,
    "claustro_encode_bridge": None,
    "claustro_search_bridge": None,
    "zq_search_bridge": None,
    "state_encoder": None,
    "targets_out": None,
    "dataset_out": None,
    "policy_weights": {},
    "value_weights": {},
}

MODE_SOURCES = {
    "direct": ("old-direct", "claustro-direct"),
    "search": ("zq-search", "claustro-search"),
    "mixed": ("old-direct", "claustro-direct", "zq-search", "claustro-search"),
}


@dataclass(frozen=True)
class CacheIdentity:
    """Identify all inputs that affect one source cache."""

    input_sha256: str
    source: str
    checkpoint_sha256: str
    params_sha256: str
    schema: str = CACHE_SCHEMA
    tool_sha256: str = ""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: object) -> str:
    """Return the SHA-256 hash of canonical JSON data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(identity: CacheIdentity) -> str:
    """Return a compact key for one cache identity."""
    return hash_json(asdict(identity))[:24]


def _decode_ids(values: np.ndarray) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("ascii"))
        else:
            result.append(str(value))
    return result


def _validate_target_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    if "id" not in arrays:
        raise ValueError("target cache lacks id")
    ids = _decode_ids(np.asarray(arrays["id"]))
    if not ids:
        raise ValueError("target cache is empty")
    if len(set(ids)) != len(ids):
        raise ValueError("target cache contains duplicate ids")
    count = len(ids)
    present = False
    if "policy" in arrays:
        present = True
        policy = np.asarray(arrays["policy"], dtype=np.float32)
        if policy.shape != (count, POLICY_DIM):
            raise ValueError(f"invalid policy shape: {policy.shape}")
        if not np.isfinite(policy).all() or np.any(policy < 0.0):
            raise ValueError("policy contains invalid values")
        if np.max(np.abs(policy.sum(axis=1) - 1.0)) > 5e-3:
            raise ValueError("policy is not normalized")
    if "value" in arrays:
        present = True
        value = np.asarray(arrays["value"], dtype=np.float32)
        if value.shape != (count,):
            raise ValueError(f"invalid value shape: {value.shape}")
        if not np.isfinite(value).all() or np.any(np.abs(value) > 1.001):
            raise ValueError("value must be finite and in [-1,1]")
    if not present:
        raise ValueError("target cache contains no policy or value")
    for name, array in arrays.items():
        if name != "id" and len(np.asarray(array)) != count:
            raise ValueError(f"target field {name} has the wrong row count")
    return count


def write_cache(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    identity: CacheIdentity,
    *,
    completed_chunks: Sequence[int],
    total_chunks: int,
) -> None:
    """Write one cache and its validation manifest."""
    clean = {name: np.asarray(value) for name, value in arrays.items()}
    count = _validate_target_arrays(clean)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **clean)
    manifest = {
        "schema": CACHE_SCHEMA,
        "identity": asdict(identity),
        "cache_key": cache_key(identity),
        "samples": count,
        "completed_chunks": sorted({int(index) for index in completed_chunks}),
        "total_chunks": int(total_chunks),
        "npz_sha256": sha256_file(path),
    }
    Path(str(path) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def cache_valid(path: Path, identity: CacheIdentity, expected_ids: Sequence[str]) -> bool:
    """Return true if a cache is complete and matches all declared inputs."""
    manifest_path = Path(str(path) + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != CACHE_SCHEMA or manifest.get("identity") != asdict(identity):
            return False
        total = int(manifest["total_chunks"])
        if total <= 0 or manifest.get("completed_chunks") != list(range(total)):
            return False
        if manifest.get("npz_sha256") != sha256_file(path):
            return False
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        if _validate_target_arrays(arrays) != len(expected_ids):
            return False
        return _decode_ids(arrays["id"]) == list(expected_ids)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def load_positions(path: Path) -> list[dict]:
    """Load exact move histories from a position JSONL file."""
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "zquoridor.position.v1":
                raise ValueError(f"{path}:{line_number}: unsupported position schema")
            history = row.get("history")
            if not isinstance(history, list) or not all(isinstance(move, str) for move in history):
                raise ValueError(f"{path}:{line_number}: an exact move history is required")
            side = row.get("side_to_move")
            if side not in (0, 1) or side != (len(history) & 1):
                raise ValueError(f"{path}:{line_number}: side and history parity differ")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}:{line_number}: id is required")
            if identifier != sample_id(history):
                raise ValueError(f"{path}:{line_number}: id does not match the move history")
            if identifier in seen:
                raise ValueError(f"{path}:{line_number}: duplicate position id")
            seen.add(identifier)
            rows.append(row)
    if not rows:
        raise ValueError("position corpus is empty")
    return rows


def group_id(row: Mapping[str, object]) -> str:
    """Return the trajectory group for one position."""
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("game_id", "trajectory_id"):
            value = metadata.get(key)
            if value is not None and str(value):
                return str(value)
    opening = int(row.get("opening_index", -1))
    if opening < 0:
        raise ValueError(f"position {row.get('id')} lacks a trajectory group")
    return f"opening:{opening}"


def sample_balanced_groups(rows, limit, seed):
    if limit <= 0:
        raise ValueError("position limit must be positive")
    groups = {}
    for row in rows:
        groups.setdefault(group_id(row), []).append(row)
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(len(items))) for items in groups.values()]
    items = list(groups.values())
    selected = []
    while len(selected) < limit:
        changed = False
        for index in rng.permutation(len(items)):
            if queues[index] and len(selected) < limit:
                selected.append(items[index][queues[index].pop()])
                changed = True
        if not changed:
            break
    return selected


def assign_group_splits(rows: Sequence[dict], val_fraction: float, seed: int) -> list[dict]:
    """Assign train and validation splits to complete trajectory groups."""
    fraction = float(val_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be finite and in (0,1)")
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(group_id(row), []).append(row)
    if len(by_group) < 2:
        raise ValueError("a group split requires at least two trajectory groups")

    assigned: dict[str, str] = {}
    for name, members in by_group.items():
        declared = {str(row["split"]) for row in members if row.get("split") in ("train", "val")}
        if len(declared) > 1:
            raise ValueError(f"group {name} has conflicting split values")
        if declared:
            assigned[name] = declared.pop()

    missing = [name for name in sorted(by_group) if name not in assigned]
    if missing:
        rng = np.random.default_rng(int(seed))
        order = [missing[index] for index in rng.permutation(len(missing))]
        desired_val = max(1, min(len(by_group) - 1, int(round(len(by_group) * fraction))))
        current_val = sum(value == "val" for value in assigned.values())
        for name in order:
            assigned[name] = "val" if current_val < desired_val else "train"
            current_val += assigned[name] == "val"
    if set(assigned.values()) != {"train", "val"}:
        raise ValueError("declared group splits must contain train and validation groups")

    result = []
    for row in rows:
        clean = dict(row)
        clean["split"] = assigned[group_id(row)]
        clean["group_id"] = group_id(row)
        result.append(clean)
    return result


def outcome_targets(rows: Sequence[dict], gamma: float) -> np.ndarray:
    """Build discounted mover outcomes from explicit game metadata."""
    validate_gamma(gamma)
    result = np.empty(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or "game_outcome" not in metadata:
            raise ValueError(f"position {row.get('id')} lacks game_outcome metadata")
        if "remaining_plies" not in metadata:
            raise ValueError(f"position {row.get('id')} lacks remaining_plies metadata")
        result[index] = discount_game_outcome(
            metadata["game_outcome"],
            int(row["side_to_move"]),
            metadata["remaining_plies"],
            gamma,
        )
    return result


def _head_weights(names: Sequence[str], configured: Mapping[str, float], head: str) -> np.ndarray:
    result = np.asarray([float(configured.get(name, 1.0)) for name in names], dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{head} weight must be finite and non-negative")
    if float(result.sum()) <= 0.0:
        raise ValueError(f"at least one {head} weight must be positive")
    return result


def join_sources(
    rows: Sequence[dict],
    sources: Mapping[str, Mapping[str, np.ndarray]],
    policy_weights: Mapping[str, float],
    value_weights: Mapping[str, float],
    *,
    policy_temperature: float = 1.0,
    value_temperature: float = 1.0,
    outcomes: np.ndarray | None = None,
    outcome_weight: float = 0.0,
    bootstrap: np.ndarray | None = None,
    bootstrap_weight: float = 0.0,
) -> dict[str, np.ndarray]:
    """Align sources by id and blend each target head independently."""
    expected_ids = [str(row["id"]) for row in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("position corpus contains duplicate ids")
    aligned = {}
    for name, raw in sources.items():
        arrays = {key: np.asarray(value) for key, value in raw.items()}
        _validate_target_arrays(arrays)
        ids = _decode_ids(arrays["id"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"source {name} contains duplicate ids")
        lookup = {identifier: index for index, identifier in enumerate(ids)}
        missing = [identifier for identifier in expected_ids if identifier not in lookup]
        if missing:
            raise ValueError(f"source {name} lacks {len(missing)} position ids")
        order = np.asarray([lookup[identifier] for identifier in expected_ids], dtype=np.int64)
        aligned[name] = {key: value[order] for key, value in arrays.items()}

    policy_names = [name for name, item in aligned.items() if "policy" in item]
    if not policy_names:
        raise ValueError("at least one source must contain policy targets")
    pw = _head_weights(policy_names, policy_weights, "policy")
    tempered_policies = np.stack([
        np.stack([temper_policy(row, policy_temperature) for row in aligned[name]["policy"]])
        for name in policy_names
    ])
    policy = np.tensordot(pw / pw.sum(), tempered_policies, axes=(0, 0)).astype(np.float32)
    masks = [item["legal_mask"].astype(bool) for item in aligned.values() if "legal_mask" in item]
    if masks:
        if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
            raise ValueError("teachers disagree about legal actions")
        policy *= masks[0]
        if (policy.sum(1) <= 0).any():
            raise ValueError("merged policy has no legal mass")
    policy /= policy.sum(axis=1, keepdims=True)

    value_names = [name for name, item in aligned.items() if "value" in item]
    if not value_names:
        raise ValueError("at least one source must contain value targets")
    vw = _head_weights(value_names, value_weights, "value")
    teacher_values = np.stack([
        np.asarray([temper_value(value, value_temperature) for value in aligned[name]["value"]])
        for name in value_names
    ])
    outcome_values = None if outcomes is None else np.asarray(outcomes, dtype=np.float32)
    bootstrap_values = None if bootstrap is None else np.asarray(bootstrap, dtype=np.float32)
    if outcome_values is not None and outcome_values.shape != (len(rows),):
        raise ValueError("outcome target count does not match the positions")
    if bootstrap_values is not None and bootstrap_values.shape != (len(rows),):
        raise ValueError("bootstrap target count does not match the positions")
    value = np.asarray([
        blend_value_targets(
            teacher_values[:, index],
            vw,
            outcome_value=None if outcome_values is None else outcome_values[index],
            outcome_weight=outcome_weight,
            bootstrap_value=None if bootstrap_values is None else bootstrap_values[index],
            bootstrap_weight=bootstrap_weight,
        )
        for index in range(len(rows))
    ], dtype=np.float32)

    top = tempered_policies.argmax(axis=2)
    consensus = policy.argmax(axis=1)
    agreement = (top == consensus[None, :]).mean(axis=0).astype(np.float32)
    value_std = teacher_values.std(axis=0).astype(np.float32)
    confidence = (agreement / (1.0 + value_std)).astype(np.float32)
    return {
        "id": np.asarray(expected_ids, dtype="S24"),
        "policy": policy,
        "value": value,
        "confidence": confidence,
        "policy_agreement": agreement,
        "value_std": value_std,
    }


Relabeler = Callable[[Sequence[dict]], Mapping[str, np.ndarray]]


def _concatenate(chunks: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks[1:]):
        raise ValueError("relabel chunks contain different fields")
    return {name: np.concatenate([np.asarray(chunk[name]) for chunk in chunks]) for name in keys}


def relabel_in_chunks(
    rows: Sequence[dict],
    relabeler: Relabeler,
    cache_dir: Path,
    identity: CacheIdentity,
    chunk_size: int,
) -> Path:
    """Relabel a source in resumable chunks and return its validated cache."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    expected_ids = [str(row["id"]) for row in rows]
    final = cache_dir / f"{identity.source}-{cache_key(identity)}.npz"
    if cache_valid(final, identity, expected_ids):
        return final
    total = (len(rows) + chunk_size - 1) // chunk_size
    chunk_dir = cache_dir / f"{identity.source}-{cache_key(identity)}.chunks"
    chunks = []
    for chunk_index in range(total):
        start = chunk_index * chunk_size
        stop = min(len(rows), start + chunk_size)
        chunk_rows = rows[start:stop]
        ids = expected_ids[start:stop]
        path = chunk_dir / f"{chunk_index:06d}.npz"
        if not cache_valid(path, identity, ids):
            arrays = {name: np.asarray(value) for name, value in relabeler(chunk_rows).items()}
            if _decode_ids(arrays.get("id", np.asarray([]))) != ids:
                raise ValueError(f"source {identity.source} returned the wrong ids for chunk {chunk_index}")
            write_cache(path, arrays, identity, completed_chunks=[0], total_chunks=1)
        with np.load(path, allow_pickle=False) as data:
            chunks.append({name: np.asarray(data[name]) for name in data.files})
    merged = _concatenate(chunks)
    write_cache(final, merged, identity, completed_chunks=range(total), total_chunks=total)
    return final


def _compile(source: Path, output: Path) -> Path:
    """Compile one standalone C++ teaching tool."""
    if output.is_file() and output.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in [source, *list((ROOT / "src").glob("*.hpp"))]):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    compiler = os.environ.get("CXX") or shutil.which("g++")
    if compiler is None and Path("C:/mingw64/bin/g++.exe").is_file():
        compiler = "C:/mingw64/bin/g++.exe"
    if compiler is None:
        raise RuntimeError("g++ is required for the teaching tools")
    command = [compiler, "-O3", "-std=c++17", "-march=native"]
    if os.name == "nt":
        command.extend(["-mavx2", "-mfma"])
    command.extend(["-I", str(ROOT / "src"), str(source), "-o", str(output)])
    subprocess.run(command, check=True)
    return output


def _tool(name: str, source: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _compile(ROOT / source, ROOT / "bin" / f"{name}{suffix}")


def _write_positions(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")


def _enrich_outcomes(rows: Sequence[dict], games_path: Path) -> list[dict]:
    games = {}
    for line in games_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            game = json.loads(line)
            games[int(game["opening_index"])] = game
    result = []
    for row in rows:
        clean = dict(row)
        metadata = dict(clean.get("metadata") or {})
        game = games.get(int(clean.get("opening_index", -1)))
        if game and game.get("termination") == "goal":
            final_plies = int(game["final_plies"])
            winner = (final_plies - 1) & 1
            metadata["game_outcome"] = 1.0 if winner == 0 else -1.0
            metadata["remaining_plies"] = final_plies - int(clean["ply"])
        clean["metadata"] = metadata
        result.append(clean)
    return result


def generate_positions(args: argparse.Namespace, out: Path) -> list[dict]:
    """Generate fresh trajectories with the compiled ZQuoridor UCI engine."""
    from collect import CollectorConfig, run_collection
    from common import load_independent_openings
    from export_positions import export_positions

    generator = _tool("teaching_generate_openings", "tools/teacher/generate_openings.cpp")
    engine = _tool("teaching_zquoridor_uci", "tools/external/zquoridor_uci.cpp")
    identity = dict(games=args.games, opening_plies=args.opening_plies, seed=args.seed,
                    time_ms=args.trajectory_movetime_ms, max_plies=args.max_plies,
                    workers=args.workers, val_fraction=args.val_fraction,
                    engine=sha256_file(engine), weights=sha256_file(args.zq_nnue),
                    collector=sha256_file(HERE / "collect.py"))
    generation_path = out / "generation.json"
    complete_positions = out / "positions.completed.jsonl"
    if generation_path.exists():
        previous = json.loads(generation_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("trajectory settings changed; choose a new out_dir")
        if complete_positions.exists() and previous.get("positions_sha256") == sha256_file(complete_positions):
            return load_positions(complete_positions)
    generation_path.write_text(json.dumps(dict(identity=identity)), encoding="utf-8")
    openings_path = out / "generated_openings.jsonl"
    with openings_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [str(generator), str(args.games), str(args.opening_plies), str(args.seed)],
            stdout=stream,
            text=True,
            check=True,
        )
    frozen = EXTERNAL / "openings_titanium.jsonl"
    openings = load_independent_openings(openings_path, frozen)
    raw_dir = out / "trajectories"
    config = CollectorConfig(
        teacher="zquoridor-old",
        protocol="uci",
        command=(str(engine), "--nnue", str(args.zq_nnue)),
        budgets=(args.trajectory_movetime_ms,),
        repeats=1,
        max_plies=args.max_plies,
        out_dir=raw_dir,
        val_mod=max(2, int(round(1.0 / args.val_fraction))),
    )
    run_collection(config, openings, args.workers)
    exported = out / "positions.full.jsonl"
    export_positions(raw_dir, exported)
    rows = _enrich_outcomes(load_positions(exported), raw_dir / "games.jsonl")
    _write_positions(complete_positions, rows)
    generation_path.write_text(json.dumps(dict(identity=identity,
        positions_sha256=sha256_file(complete_positions))), encoding="utf-8")
    return rows


def _encode_states(rows: Sequence[dict], encoder: Path) -> dict[str, np.ndarray]:
    from build_teacher_soft import encode_states

    return encode_states(rows, encoder)


def independent_states(rows, encoder):
    """Remove frozen opening states and states shared across data splits."""
    fields = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own", "walls_left_opp")
    states = _encode_states(rows, encoder)
    book = [json.loads(line)["moves"] for line in (EXTERNAL / "openings_titanium.jsonl").read_text().splitlines() if line.strip()]
    histories = {tuple(moves[:end]) for moves in book for end in range(len(moves)+1)}
    frozen = _encode_states([dict(history=list(h)) for h in histories], encoder)
    blocked = {tuple(int(frozen[k][i]) for k in fields) for i in range(len(histories))}
    keys = [tuple(int(states[k][i]) for k in fields) for i in range(len(rows))]
    first = {}
    for i, key in enumerate(keys):
        if key in first and rows[first[key]]["split"] != rows[i]["split"]:
            blocked.add(key)
        first.setdefault(key, i)
    selected = [rows[i] for key, i in first.items() if key not in blocked]
    if {row["split"] for row in selected} != {"train", "val"}:
        raise ValueError("no independent train/validation states remain")
    return selected


def old_direct_relabeler(checkpoint: Path, encoder: Path, batch_size: int, device_name: str) -> Relabeler:
    """Create a relabeler for the current baseline NNUE."""
    import torch
    import train_nnue as base
    from train_teacher_policy import dense_features

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    model = base.QuoridorNNUE()
    base._load_into_model(model, base._load_raw_weights(checkpoint))
    model.to(device).eval()

    def relabel(rows: Sequence[dict]) -> Mapping[str, np.ndarray]:
        compact = _encode_states(rows, encoder)
        features = dense_features(compact, np.arange(len(rows), dtype=np.int64))
        policies = []
        values = []
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                inputs = torch.from_numpy(features[start:start + batch_size]).to(device)
                value_logits, policy_logits = model(inputs)
                policies.append(torch.softmax(policy_logits.float(), dim=1).cpu().numpy())
                values.append((2.0 * torch.sigmoid(value_logits.float()) - 1.0).cpu().numpy())
        return {
            "id": np.asarray([row["id"] for row in rows], dtype="S24"),
            "policy": np.concatenate(policies).astype(np.float32),
            "value": np.concatenate(values).astype(np.float32),
        }

    return relabel


def claustro_direct_relabeler(
    checkpoint: Path, bridge: Path, batch_size: int, device_name: str
) -> Relabeler:
    """Create a direct Claustrophobia relabeler."""
    import torch
    import claustrophobia_relabel as module

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )

    teacher = torch.jit.load(str(checkpoint), map_location=device).eval()

    def relabel(rows: Sequence[dict]) -> Mapping[str, np.ndarray]:
        planes, legal = module.encode_positions(rows, bridge)
        policy, value, _logits, _illegal = module.forward_teacher(
            planes, legal, checkpoint, batch_size, device, False, teacher=teacher
        )
        return {
            "id": np.asarray([row["id"] for row in rows], dtype="S24"),
            "policy": module.convert_policy_frames(policy, rows),
            "value": value,
            "legal_mask": module.convert_policy_frames(legal.astype(np.float32), rows).astype(np.uint8),
        }

    return relabel


def claustro_search_relabeler(checkpoint: Path, bridge: Path, sims: int, cpuct: float) -> Relabeler:
    """Create a Claustrophobia search relabeler."""
    import claustrophobia_search_relabel as module

    def relabel(rows: Sequence[dict]) -> Mapping[str, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="zq_teaching_search_") as directory:
            path = Path(directory) / "positions.tsv"
            module.write_tsv(rows, path)
            policy, value, best = module.run_budget(bridge, checkpoint, path, sims, cpuct, rows)
        return {
            "id": np.asarray([row["id"] for row in rows], dtype="S24"),
            "policy": policy,
            "value": value,
            "best_action": best,
        }

    return relabel


def zq_search_relabeler(
    nnue: Path, bridge: Path, nodes: int, time_ms: int, leaf_depth: int
) -> Relabeler:
    """Create a ZQuoridor MCAB search relabeler."""
    import zq_deep_relabel as module

    def relabel(rows: Sequence[dict]) -> Mapping[str, np.ndarray]:
        policy, value, action_q, best = module.run_budget(
            bridge, nnue, rows, nodes, time_ms, leaf_depth
        )
        return {
            "id": np.asarray([row["id"] for row in rows], dtype="S24"),
            "policy": policy,
            "value": value,
            "action_q": action_q,
            "best_action": best,
        }

    return relabel


def parse_weight(text: str) -> tuple[str, float]:
    """Parse one NAME=WEIGHT source setting."""
    try:
        name, raw = text.split("=", 1)
        weight = float(raw)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("weight must use NAME=WEIGHT") from error
    if not name.strip() or not math.isfinite(weight) or weight < 0.0:
        raise argparse.ArgumentTypeError("source weight must be finite and non-negative")
    return name.strip(), weight


def _weights(values: Sequence[tuple[str, float]], defaults: Mapping[str, float]) -> dict[str, float]:
    result = {str(name): float(weight) for name, weight in defaults.items()}
    for name, weight in values:
        result[name] = weight
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional overrides for the top-level configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_SOURCES), default=CONFIG["mode"])
    parser.add_argument("--positions", type=Path, default=CONFIG["positions"])
    parser.add_argument("--out-dir", type=Path, default=CONFIG["out_dir"])
    parser.add_argument("--games", type=int, default=CONFIG["games"])
    parser.add_argument("--opening-plies", type=int, default=CONFIG["opening_plies"])
    parser.add_argument("--max-plies", type=int, default=CONFIG["max_plies"])
    parser.add_argument("--max-positions", type=int, default=CONFIG["max_positions"])
    parser.add_argument("--trajectory-movetime-ms", type=int, default=CONFIG["trajectory_movetime_ms"])
    parser.add_argument("--workers", type=int, default=CONFIG["workers"])
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--val-fraction", type=float, default=CONFIG["val_fraction"])
    parser.add_argument("--gamma", type=float, default=CONFIG["gamma"])
    parser.add_argument("--outcome-weight", type=float, default=CONFIG["outcome_weight"])
    parser.add_argument("--bootstrap-weight", type=float, default=CONFIG["bootstrap_weight"])
    parser.add_argument("--policy-temperature", type=float, default=CONFIG["policy_temperature"])
    parser.add_argument("--value-temperature", type=float, default=CONFIG["value_temperature"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--device", default=CONFIG["device"])
    parser.add_argument("--claustro-sims", type=int, default=CONFIG["claustro_sims"])
    parser.add_argument("--claustro-cpuct", type=float, default=CONFIG["claustro_cpuct"])
    parser.add_argument("--zq-nodes", type=int, default=CONFIG["zq_nodes"])
    parser.add_argument("--zq-time-ms", type=int, default=CONFIG["zq_time_ms"])
    parser.add_argument("--zq-leaf-depth", type=int, default=CONFIG["zq_leaf_depth"])
    parser.add_argument("--old-nnue", type=Path, default=CONFIG["old_nnue"])
    parser.add_argument("--zq-nnue", type=Path, default=CONFIG["zq_nnue"])
    parser.add_argument("--claustro-checkpoint", type=Path, default=CONFIG["claustro_checkpoint"])
    parser.add_argument("--claustro-encode-bridge", type=Path, default=CONFIG["claustro_encode_bridge"])
    parser.add_argument("--claustro-search-bridge", type=Path, default=CONFIG["claustro_search_bridge"])
    parser.add_argument("--zq-search-bridge", type=Path, default=CONFIG["zq_search_bridge"])
    parser.add_argument("--state-encoder", type=Path, default=CONFIG["state_encoder"])
    parser.add_argument("--policy-source-weight", action="append", type=parse_weight, default=[])
    parser.add_argument("--value-source-weight", action="append", type=parse_weight, default=[])
    parser.add_argument("--targets-out", type=Path, default=CONFIG["targets_out"])
    parser.add_argument("--dataset-out", type=Path, default=CONFIG["dataset_out"])
    return parser.parse_args(argv)


def _check_args(args: argparse.Namespace) -> None:
    validate_gamma(args.gamma)
    if args.claustro_sims <= 0 or args.zq_nodes < 2 or args.zq_time_ms < 0 or args.zq_leaf_depth < 0:
        raise ValueError("search needs positive simulations, at least two ZQ nodes, and nonnegative time/depth")
    if not math.isfinite(args.claustro_cpuct) or args.claustro_cpuct <= 0:
        raise ValueError("claustro_cpuct must be finite and positive")
    positive = {
        "games": args.games,
        "max-plies": args.max_plies,
        "max-positions": args.max_positions,
        "workers": args.workers,
        "chunk-size": args.chunk_size,
        "batch-size": args.batch_size,
        "trajectory-movetime-ms": args.trajectory_movetime_ms,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if not 0 <= args.opening_plies <= 24:
        raise ValueError("opening-plies must be in [0,24]")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0,1)")
    for name in ("outcome_weight", "bootstrap_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    temper_policy([1.0], args.policy_temperature)
    temper_value(0.0, args.value_temperature)


def source_identity(
    positions_path: Path, source: str, checkpoint: Path, params: Mapping[str, object], tools=()
) -> CacheIdentity:
    return CacheIdentity(
        input_sha256=sha256_file(positions_path),
        source=source,
        checkpoint_sha256=sha256_file(checkpoint),
        params_sha256=hash_json(params),
        tool_sha256=hash_json({str(p): sha256_file(Path(p)) for p in tools}),
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run collection, cached relabeling, target merge, and dataset encoding."""
    _check_args(args)
    os.environ["ZQ_PYTHON"] = sys.executable
    import torch
    os.environ["QUORIDOR_DEVICE"] = "gpu" if args.device in ("cuda", "gpu") or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.positions is None:
        rows = generate_positions(args, out_dir)
        input_source = "fresh-zquoridor-trajectories"
    else:
        rows = load_positions(args.positions.resolve())
        input_source = str(args.positions.resolve())
    if args.outcome_weight > 0 and args.positions is None:
        rows = [row for row in rows if "game_outcome" in (row.get("metadata") or {})]
        if not rows:
            raise ValueError("no completed games have known outcomes for temporal supervision")
    rows = assign_group_splits(sample_balanced_groups(rows, args.max_positions, args.seed), args.val_fraction, args.seed)
    state_encoder = args.state_encoder or _tool("teacher_encode_state", "tools/teacher/encode_state.cpp")
    rows = independent_states(rows, state_encoder)
    positions_path = out_dir / "positions.jsonl"
    _write_positions(positions_path, rows)

    from bot_setup import ensure_bot

    selected = list(MODE_SOURCES[args.mode])
    if args.bootstrap_weight > 0.0 and "zq-search" not in selected:
        selected.append("zq-search")
    needs_claustro = any(name.startswith("claustro") for name in selected)
    claustro = ensure_bot("claustrophobia", root=ROOT, build=True) if needs_claustro else {}
    claustro_checkpoint = (args.claustro_checkpoint or claustro.get("checkpoint"))
    encode_bridge = args.claustro_encode_bridge or claustro.get("encode_bridge")
    search_bridge = args.claustro_search_bridge or claustro.get("search_bridge")
    state_encoder = args.state_encoder or _tool("teacher_encode_state", "tools/teacher/encode_state.cpp")
    zq_bridge = args.zq_search_bridge or _tool("zq_deep_relabel", "tools/teacher/zq_deep_relabel.cpp")

    source_specs: dict[str, tuple[Path, dict, Relabeler]] = {}
    if "old-direct" in selected:
        params = {"batch_size": args.batch_size, "device": args.device, "kind": "direct"}
        source_specs["old-direct"] = (
            args.old_nnue,
            params,
            old_direct_relabeler(args.old_nnue, state_encoder, args.batch_size, args.device),
        )
    if "claustro-direct" in selected:
        params = {"batch_size": args.batch_size, "device": args.device, "kind": "direct"}
        source_specs["claustro-direct"] = (
            claustro_checkpoint,
            params,
            claustro_direct_relabeler(
                claustro_checkpoint, encode_bridge, args.batch_size, args.device
            ),
        )
    if "claustro-search" in selected:
        params = {"sims": args.claustro_sims, "cpuct": args.claustro_cpuct, "kind": "search"}
        source_specs["claustro-search"] = (
            claustro_checkpoint,
            params,
            claustro_search_relabeler(
                claustro_checkpoint, search_bridge, args.claustro_sims, args.claustro_cpuct
            ),
        )
    if "zq-search" in selected:
        params = {
            "nodes": args.zq_nodes,
            "time_ms": args.zq_time_ms,
            "leaf_depth": args.zq_leaf_depth,
            "kind": "search",
        }
        source_specs["zq-search"] = (
            args.zq_nnue,
            params,
            zq_search_relabeler(
                args.zq_nnue, zq_bridge, args.zq_nodes, args.zq_time_ms, args.zq_leaf_depth
            ),
        )

    caches = {}
    arrays = {}
    for name, (checkpoint, params, relabeler) in source_specs.items():
        identity = source_identity(positions_path, name, Path(checkpoint), params, tools=[
            HERE / "teaching_pipeline.py", HERE / "targets.py", HERE / "value_targets.py",
            state_encoder, zq_bridge, *([encode_bridge, search_bridge,
                HERE / "claustrophobia_inference_worker.py", HERE / "claustrophobia_ipc_eval.rs"] if needs_claustro else [])])
        cache = relabel_in_chunks(rows, relabeler, out_dir / "cache", identity, args.chunk_size)
        caches[name] = str(cache)
        arrays[name] = _load_npz(cache)

    merge_names = MODE_SOURCES[args.mode]
    merge_sources = {name: arrays[name] for name in merge_names}
    policy_weights = _weights(args.policy_source_weight, CONFIG["policy_weights"])
    value_weights = _weights(args.value_source_weight, CONFIG["value_weights"])
    outcomes = outcome_targets(rows, args.gamma) if args.outcome_weight > 0.0 else None
    bootstrap = arrays["zq-search"]["value"] if args.bootstrap_weight > 0.0 else None
    targets = join_sources(
        rows,
        merge_sources,
        policy_weights,
        value_weights,
        policy_temperature=args.policy_temperature,
        value_temperature=args.value_temperature,
        outcomes=outcomes,
        outcome_weight=args.outcome_weight,
        bootstrap=bootstrap,
        bootstrap_weight=args.bootstrap_weight,
    )
    targets_path = args.targets_out or out_dir / "teacher_targets.npz"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(targets_path, **targets)

    import build_teacher_soft

    dataset_path = args.dataset_out or out_dir / "dataset.npz"
    build_teacher_soft.main([
        "--positions", str(positions_path),
        "--targets", str(targets_path),
        "--encoder", str(state_encoder),
        "--out", str(dataset_path),
    ])
    dataset = _load_npz(dataset_path)
    dataset["group_id"] = np.asarray([row["group_id"] for row in rows], dtype="S64")
    np.savez(dataset_path, **dataset)

    manifest = {
        "schema": TARGET_SCHEMA,
        "mode": args.mode,
        "input": input_source,
        "positions": str(positions_path),
        "targets": str(targets_path),
        "dataset": str(dataset_path),
        "samples": len(rows),
        "games": len({group_id(row) for row in rows}),
        "train_samples": sum(row["split"] == "train" for row in rows),
        "validation_samples": sum(row["split"] == "val" for row in rows),
        "sources": list(merge_names),
        "source_caches": caches,
        "policy_source_weights": {name: policy_weights.get(name, 1.0) for name in merge_names},
        "value_source_weights": {name: value_weights.get(name, 1.0) for name in merge_names},
        "policy_temperature": args.policy_temperature,
        "value_temperature": args.value_temperature,
        "gamma": args.gamma,
        "outcome_weight": args.outcome_weight,
        "bootstrap_weight": args.bootstrap_weight,
        "group_split": True,
        "frozen_benchmark_openings_used": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the teaching pipeline."""
    try:
        run_pipeline(parse_args(argv))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
