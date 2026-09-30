#!/usr/bin/env python3
"""Generate a large self-play corpus focused on weak central openings.

This runner exists for one specific training problem: strengthen the engine in
central/Center-Rush structures where paired external-opponent benchmarks show a
persistent score deficit, without turning the next network into a narrow
opening-book imitator.

The campaign deliberately separates *trajectory exploration* from *training
labels*:

1. Build a large opening bank around the classic central arrival
   ``e2 e8 e3 e7 e4 e6`` and around empirically weak catalog lines.  The bank
   is generated with legal-move geometry only; the current NNUE is NOT used to
   accept/reject seeds.  This avoids filtering the opening space through the
   same evaluator that the new network is supposed to improve.
2. Start self-play from those roughly 8-10 ply roots and use several MCAB
   visit-temperature / Dirichlet-noise profiles.  Temperature changes the
   trajectory, while V3 still stores the untempered top-eight root-visit
   distribution as the policy target.
3. Record aligned ``.meta`` sidecars.  They retain MCAB root values and exact
   plies-to-end, so the replay stage can train value from searched evaluation
   rather than blindly copying the raw NNUE evaluation stored in V3.
4. Prepare a stored-search replay dataset with an exact 50/50 value blend:

       value = 0.5 * signed_MCAB_root_value + 0.5 * terminal_game_result

   ``stored_gamma=1`` keeps that blend literal.  Duplicate canonical states are
   aggregated by ``training/prepare_replay.py`` and game IDs are globally
   namespaced per shard before admission, keeping train/validation grouping
   clean.

The defaults intentionally target sixteen million recorded full-search
positions and cap the deduplicated replay dataset at twelve million positions.
The runner is resumable shard-by-shard.  Set ``dry_run`` to ``False`` in
``CONFIG`` (or pass ``--no-dry-run``) only when the self-play executable and
weights are ready.

Outputs under ``data/selfplay/central-weakness-rollouts-16m``:

- ``opening_bank_8to10ply.jsonl``: generated start positions;
- ``opening_bank.manifest.json``: reproducible bank identity and statistics;
- ``central_weakness_XXXXX.bin/.meta``: V3 self-play + aligned search metadata;
- ``campaign_manifest.json``: resumable campaign provenance;
- ``training_recipe.json``: exact replay/training contract;
- ``replay-50-50/dataset.npz``: optional prepared training dataset.

This is a public campaign runner.  Edit ``CONFIG`` for normal use; CLI flags
provide one-run overrides for the operational fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the already-tested lightweight Python rules/replay implementation used
# by the canonical seed-schedule builder.  The leading underscore is deliberate:
# these helpers are implementation details, but sharing them prevents a second
# subtly different Quoridor legality implementation from drifting in this file.
from tools.teacher.build_selfplay_seed_schedule import (  # noqa: E402
    _ReplayState,
    _apply_move,
    _replay_history,
)
from tools.teacher.run_four_million_selfplay import OutputLock  # noqa: E402

SCHEMA = "zquoridor.central_weakness_rollouts.v1"
BANK_SCHEMA = "zquoridor.central_weakness_opening_bank.v1"
RECIPE_SCHEMA = "zquoridor.central_weakness_training_recipe.v1"
V3_RECORD_BYTES = 64
META_DTYPE = np.dtype([
    ("game", "<u8"), ("root", "<f4"), ("plies", "<u2"), ("length", "<u2"),
    ("source", "u1"), ("flags", "u1"), ("reserved", "<u2"),
])
META_RECORD_BYTES = META_DTYPE.itemsize
CLASSIC_PREFIX = ("e2", "e8", "e3", "e7", "e4", "e6")

# These catalog indices were among the weakest in the 2026-09 central-family
# diagnostic.  They are empirical seeds, not a claim that the listed move is a
# forced loss.  Keeping the list in CONFIG makes the campaign easy to retarget
# after a new benchmark.
DEFAULT_WEAK_OPENING_INDICES = (7, 10, 15, 18, 23, 24, 29, 30)


@dataclass(frozen=True)
class RolloutProfile:
    """One deterministic exploration regime used for a whole output shard."""

    name: str
    temp_opening: float
    temp_end: float
    temp_decay_plies: int
    root_noise_epsilon: float
    full_search_prob: float
    epsilon_midgame: float = 0.01


ROLLOUT_PROFILES = {
    # Wide: aggressively explores plausible alternatives from root visits.
    "wide": RolloutProfile("wide", 1.60, 0.35, 24, 0.35, 0.42, 0.015),
    # Balanced: broad enough to escape the current principal variation while
    # still spending more of the trajectory on full-search samples.
    "balanced": RolloutProfile("balanced", 1.15, 0.20, 22, 0.25, 0.55, 0.010),
    # Sharp: preserves a lower-temperature control population near the engine's
    # stronger choices; useful against learning only from noisy trajectories.
    "sharp": RolloutProfile("sharp", 0.78, 0.12, 18, 0.15, 0.70, 0.005),
}
PROFILE_CYCLE = ("wide", "balanced", "wide", "balanced", "sharp")

# Edit this block for normal use.  CLI options override the scalar/path fields.
CONFIG = {
    "opening_book": "tools/external/openings_center_rush_50pairs.jsonl",
    "out": "data/selfplay/central-weakness-rollouts-16m",
    "exe": "bin/selfplay",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "seed_bank_size": 50_000,
    # Roughly-eight-ply means most roots are 8 plies, with 9-10 ply tails to
    # provide parity balance and enough combinatorial room for a large bank.
    "seed_ply_weights": {8: 0.55, 9: 0.30, 10: 0.15},
    "weak_opening_indices": DEFAULT_WEAK_OPENING_INDICES,
    "source_mix": {"weak": 0.60, "catalog": 0.25, "classic": 0.15},
    "weak_exact_fraction": 0.30,
    "branch_ply_weights": {6: 0.25, 7: 0.35, 8: 0.30, 9: 0.10},
    "mirror_probability": 0.50,
    # Seed-bank proposal is geometry-only.  Separate pawn/wall mass avoids
    # uniform legal-move sampling being dominated by the much larger wall set.
    "seed_pawn_mass": 0.42,
    "seed_uniform_move_prob": 0.20,
    "seed_proposal_temperature": 1.35,
    "seed_wall_margin": 1,
    "seed_cache_limit": 10_000,
    "target_positions": 16_000_000,
    "replay_max_positions": 12_000_000,
    "games_per_shard": 512,
    "threads": 12,
    "depth": 50,
    "time_ms": 80,
    "cheap_time_ms": 20,
    "max_plies": 160,
    "seed": 20260930,
    "val_fraction": 0.15,
    # Exact requested 50/50 result/search-evaluation blend.  Gamma 1.0 keeps
    # the terminal half from being weakened as the sample moves earlier.
    "stored_outcome_weight": 0.50,
    "stored_gamma": 1.0,
    "prepare_replay": True,
    "rebuild_bank": False,
    "dry_run": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stable_fingerprint(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _weighted_choice(rng: random.Random, weights: Mapping[object, float]):
    items = [(item, float(weight)) for item, weight in weights.items() if float(weight) > 0.0]
    if not items:
        raise ValueError("weighted choice requires at least one positive weight")
    total = sum(weight for _, weight in items)
    needle = rng.random() * total
    acc = 0.0
    for item, weight in items:
        acc += weight
        if needle <= acc:
            return item
    return items[-1][0]


def _mirror_move_text(text: str) -> str:
    """Mirror one move left/right; applying twice returns the original move."""
    if len(text) not in (2, 3):
        raise ValueError(f"invalid move text: {text!r}")
    col = ord(text[0].lower()) - ord("a")
    row = text[1]
    if len(text) == 2:
        if not 0 <= col < 9:
            raise ValueError(f"invalid pawn move: {text!r}")
        return chr(ord("a") + (8 - col)) + row
    if not 0 <= col < 8 or text[2].lower() not in ("h", "v"):
        raise ValueError(f"invalid wall move: {text!r}")
    return chr(ord("a") + (7 - col)) + row + text[2].lower()


def _mirror_history(history: Sequence[str]) -> tuple[str, ...]:
    return tuple(_mirror_move_text(move) for move in history)


def _clone_state(state: _ReplayState) -> _ReplayState:
    return _ReplayState(
        list(state.pawns), set(state.walls_h), set(state.walls_v),
        list(state.walls_left), state.turn,
    )


def _wall_anchors(state: _ReplayState, margin: int) -> set[tuple[int, int]]:
    margin = max(0, min(3, int(margin)))
    anchors = {
        (row, col)
        for row in range(margin, 8 - margin)
        for col in range(margin, 8 - margin)
    }
    # Always retain local tactical wall anchors around either pawn, even when
    # a margin would otherwise remove them from the central proposal set.
    for pawn in state.pawns:
        prow, pcol = divmod(pawn, 9)
        for row in range(max(0, prow - 2), min(8, prow + 2)):
            for col in range(max(0, pcol - 2), min(8, pcol + 2)):
                anchors.add((row, col))
    return anchors


def _legal_proposals(state: _ReplayState, wall_margin: int) -> list[tuple[str, str, float]]:
    """Return legal bank-generation moves with network-independent scores."""
    result: list[tuple[str, str, float]] = []
    mover = state.turn
    current_row, current_col = divmod(state.pawns[mover], 9)

    # Pawn candidates are cheap to validate; brute-force all cells through the
    # shared rules helper so jumps/diagonals stay exactly aligned with the repo.
    for cell in range(81):
        row, col = divmod(cell, 9)
        text = f"{chr(ord('a') + col)}{row + 1}"
        trial = _clone_state(state)
        if not _apply_move(trial, text):
            continue
        direction = 1 if mover == 0 else -1
        progress = direction * (row - current_row)
        lateral = abs(col - current_col)
        center = abs(col - 4)
        score = 1.10 * progress - 0.12 * center - 0.08 * lateral
        result.append((text, "pawn", score))

    # Central/tactical wall anchors are enough for the opening-bank proposal.
    # The subsequent MCAB rollouts still see every legal move on the board.
    for row, col in sorted(_wall_anchors(state, wall_margin)):
        for orientation in ("h", "v"):
            text = f"{chr(ord('a') + col)}{row + 1}{orientation}"
            trial = _clone_state(state)
            if not _apply_move(trial, text):
                continue
            center_dist = abs(row - 3.5) + abs(col - 3.5)
            pawn_dist = min(
                abs(row + 0.5 - (pawn // 9)) + abs(col + 0.5 - (pawn % 9))
                for pawn in state.pawns
            )
            score = -0.10 * center_dist - 0.04 * pawn_dist
            result.append((text, "wall", score))
    return result


def _sample_proposal(
    rng: random.Random,
    candidates: Sequence[tuple[str, str, float]],
    *,
    pawn_mass: float,
    temperature: float,
    uniform_prob: float,
) -> str:
    if not candidates:
        raise ValueError("cannot sample from an empty legal-move proposal")
    if temperature <= 0.0:
        raise ValueError("seed proposal temperature must be positive")
    if not 0.0 <= uniform_prob <= 1.0:
        raise ValueError("seed uniform move probability must be in [0,1]")
    pawn = [item for item in candidates if item[1] == "pawn"]
    wall = [item for item in candidates if item[1] == "wall"]
    if pawn and wall:
        group = pawn if rng.random() < pawn_mass else wall
    else:
        group = pawn or wall

    scores = [item[2] for item in group]
    maximum = max(scores)
    soft = [math.exp((score - maximum) / temperature) for score in scores]
    total_soft = sum(soft)
    n = len(group)
    probs = [
        (1.0 - uniform_prob) * weight / total_soft + uniform_prob / n
        for weight in soft
    ]
    needle = rng.random()
    acc = 0.0
    for item, probability in zip(group, probs):
        acc += probability
        if needle <= acc:
            return item[0]
    return group[-1][0]


def _advance_history(
    history: Sequence[str],
    target_plies: int,
    rng: random.Random,
    *,
    pawn_mass: float,
    temperature: float,
    uniform_prob: float,
    wall_margin: int,
    cache: dict[tuple[str, ...], list[tuple[str, str, float]]],
    cache_limit: int,
) -> tuple[str, ...] | None:
    if len(history) > target_plies:
        history = history[:target_plies]
    state = _replay_history(history)
    if state is None:
        return None
    result = list(history)
    while len(result) < target_plies:
        key = tuple(result)
        candidates = cache.get(key)
        if candidates is None:
            candidates = _legal_proposals(state, wall_margin)
            if len(cache) < cache_limit:
                cache[key] = candidates
        if not candidates:
            return None
        move = _sample_proposal(
            rng, candidates, pawn_mass=pawn_mass,
            temperature=temperature, uniform_prob=uniform_prob,
        )
        if not _apply_move(state, move):
            # A cached proposal must stay legal for the same exact history.
            raise RuntimeError(f"cached legal proposal became illegal: {key!r} -> {move}")
        result.append(move)
    return tuple(result)


def _load_opening_book(path: Path) -> list[dict]:
    """Load legal catalog rows; the bank later selects classic-prefix rows."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"opening book is empty: {path}")
    for row in rows:
        moves = row.get("moves")
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise ValueError(f"opening row lacks a string 'moves' list: {row!r}")
        if _replay_history(moves) is None:
            raise ValueError(f"opening row contains an illegal history: {row.get('opening_index')}")
    return rows


def _generate_opening_bank(args: argparse.Namespace, bank_path: Path, manifest_path: Path) -> dict:
    book_path = args.opening_book.resolve()
    rows = _load_opening_book(book_path)
    classic_rows = [
        row for row in rows
        if tuple(row["moves"][: len(CLASSIC_PREFIX)]) == CLASSIC_PREFIX
    ]
    if not classic_rows:
        raise ValueError(f"opening book has no rows with classic prefix {CLASSIC_PREFIX}")
    by_index = {int(row.get("opening_index", -1)): row for row in classic_rows}
    missing = [index for index in args.weak_opening_indices if index not in by_index]
    if missing:
        raise ValueError(f"weak opening indices are absent from {book_path}: {missing}")

    identity = {
        "schema": BANK_SCHEMA,
        "book": str(book_path),
        "book_sha256": _sha256(book_path),
        "classic_prefix_catalog_rows": len(classic_rows),
        "size": args.seed_bank_size,
        "seed": args.seed,
        "seed_ply_weights": args.seed_ply_weights,
        "weak_opening_indices": list(args.weak_opening_indices),
        "source_mix": args.source_mix,
        "weak_exact_fraction": args.weak_exact_fraction,
        "branch_ply_weights": args.branch_ply_weights,
        "mirror_probability": args.mirror_probability,
        "pawn_mass": args.seed_pawn_mass,
        "uniform_move_prob": args.seed_uniform_move_prob,
        "proposal_temperature": args.seed_proposal_temperature,
        "wall_margin": args.seed_wall_margin,
    }
    fingerprint = _stable_fingerprint(identity)
    if bank_path.exists() and manifest_path.exists() and not args.rebuild_bank:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(
                "opening bank settings changed; use --rebuild-bank or a new output directory"
            )
        if previous.get("rows") != args.seed_bank_size:
            raise ValueError("existing opening bank has the wrong row count")
        return previous

    rng = random.Random(args.seed ^ 0xC3E71A9)
    weak_rows = [by_index[index] for index in args.weak_opening_indices]
    seen: set[tuple[str, ...]] = set()
    bank_rows: list[dict] = []
    cache: dict[tuple[str, ...], list[tuple[str, str, float]]] = {}
    source_counts = {"weak_exact": 0, "weak_branch": 0, "catalog": 0, "classic": 0}
    ply_counts: dict[int, int] = {}
    mirrored_count = 0

    def admit(history: Sequence[str], *, source: str, opening: dict | None,
              branch_ply: int, mirrored: bool) -> bool:
        nonlocal mirrored_count
        key = tuple(history)
        if key in seen or _replay_history(key) is None:
            return False
        seen.add(key)
        row = {
            "schema": "zquoridor.position.v1",
            "id": f"cwbank-{len(bank_rows):06d}",
            "history": list(key),
            "side_to_move": len(key) & 1,
            "source": source,
            "opening_index": None if opening is None else opening.get("opening_index"),
            "category": None if opening is None else opening.get("category"),
            "branch_ply": int(branch_ply),
            "target_plies": len(key),
            "mirrored": bool(mirrored),
        }
        bank_rows.append(row)
        source_counts[source] = source_counts.get(source, 0) + 1
        ply_counts[len(key)] = ply_counts.get(len(key), 0) + 1
        mirrored_count += int(mirrored)
        return True

    # Guarantee the exact measured weak motifs (plus their left/right mirrors)
    # are present before filling the stochastic bank around them.
    for opening in weak_rows:
        moves = tuple(opening["moves"])
        for depth in range(8, min(10, len(moves)) + 1):
            exact = moves[:depth]
            if len(bank_rows) < args.seed_bank_size:
                admit(exact, source="weak_exact", opening=opening, branch_ply=depth, mirrored=False)
            mirrored = _mirror_history(exact)
            if len(bank_rows) < args.seed_bank_size:
                admit(mirrored, source="weak_exact", opening=opening, branch_ply=depth, mirrored=True)

    max_attempts = max(args.seed_bank_size * 80, 10_000)
    attempts = 0
    while len(bank_rows) < args.seed_bank_size and attempts < max_attempts:
        attempts += 1
        target_plies = int(_weighted_choice(rng, args.seed_ply_weights))
        source_mode = str(_weighted_choice(rng, args.source_mix))
        opening: dict | None = None
        branch_ply = len(CLASSIC_PREFIX)
        source_label = source_mode

        if source_mode == "classic":
            history = CLASSIC_PREFIX
        else:
            opening = rng.choice(weak_rows if source_mode == "weak" else classic_rows)
            moves = tuple(opening["moves"])
            if source_mode == "weak" and rng.random() < args.weak_exact_fraction:
                branch_ply = min(target_plies, len(moves))
                history = moves[:branch_ply]
                source_label = "weak_exact"
            else:
                valid_branch = {
                    int(ply): float(weight)
                    for ply, weight in args.branch_ply_weights.items()
                    if len(CLASSIC_PREFIX) <= int(ply) <= min(target_plies, len(moves))
                }
                branch_ply = int(_weighted_choice(rng, valid_branch)) if valid_branch else len(CLASSIC_PREFIX)
                history = moves[:branch_ply]
                source_label = "weak_branch" if source_mode == "weak" else "catalog"

        candidate = _advance_history(
            history, target_plies, rng,
            pawn_mass=args.seed_pawn_mass,
            temperature=args.seed_proposal_temperature,
            uniform_prob=args.seed_uniform_move_prob,
            wall_margin=args.seed_wall_margin,
            cache=cache,
            cache_limit=args.seed_cache_limit,
        )
        if candidate is None:
            continue
        mirrored = rng.random() < args.mirror_probability
        if mirrored:
            candidate = _mirror_history(candidate)
        admit(candidate, source=source_label, opening=opening,
              branch_ply=branch_ply, mirrored=mirrored)

    if len(bank_rows) != args.seed_bank_size:
        raise RuntimeError(
            f"generated only {len(bank_rows):,}/{args.seed_bank_size:,} unique legal roots "
            f"after {attempts:,} attempts; widen the seed-depth distribution or proposal set"
        )

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bank_path.with_suffix(bank_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in bank_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(bank_path)
    manifest = {
        **identity,
        "fingerprint": fingerprint,
        "rows": len(bank_rows),
        "bank_sha256": _sha256(bank_path),
        "source_counts": source_counts,
        "ply_counts": {str(key): value for key, value in sorted(ply_counts.items())},
        "mirrored_rows": mirrored_count,
        "generation_attempts": attempts,
        "network_evaluator_used_for_seed_selection": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _resolve_executable(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        return path
    if path.suffix.lower() != ".exe":
        alternate = path.with_suffix(".exe")
        if alternate.exists():
            return alternate
    raise FileNotFoundError(path)


def _pair_record_count(v3_path: Path, meta_path: Path) -> int:
    if not v3_path.exists() or not meta_path.exists():
        raise ValueError(f"incomplete self-play pair: {v3_path} / {meta_path}")
    if v3_path.stat().st_size % V3_RECORD_BYTES:
        raise ValueError(f"V3 shard is not {V3_RECORD_BYTES}-byte aligned: {v3_path}")
    if meta_path.stat().st_size % META_RECORD_BYTES:
        raise ValueError(f"metadata shard is not {META_RECORD_BYTES}-byte aligned: {meta_path}")
    v3_count = v3_path.stat().st_size // V3_RECORD_BYTES
    meta_count = meta_path.stat().st_size // META_RECORD_BYTES
    if v3_count != meta_count:
        raise ValueError(f"V3/meta record-count mismatch: {v3_count} != {meta_count}")
    return int(v3_count)


def _namespace_metadata_game_ids(meta_path: Path, shard_index: int, games_per_shard: int) -> None:
    """Make metadata game IDs globally unique across independently run shards."""
    data = np.memmap(meta_path, dtype=META_DTYPE, mode="r+")
    try:
        if len(data) == 0:
            return
        max_local = int(data["game"].max())
        if max_local >= games_per_shard:
            raise ValueError(
                f"unexpected local game id {max_local} for {games_per_shard} games in {meta_path}"
            )
        data["game"] += np.uint64(shard_index * games_per_shard)
        data.flush()
    finally:
        del data


def _profile_for_shard(shard_index: int) -> RolloutProfile:
    return ROLLOUT_PROFILES[PROFILE_CYCLE[shard_index % len(PROFILE_CYCLE)]]


def _build_selfplay_command(
    args: argparse.Namespace,
    *,
    executable: Path,
    weights: Path,
    bank_path: Path,
    shard_index: int,
    out_bin: Path,
    out_meta: Path,
) -> tuple[list[str], RolloutProfile, int]:
    profile = _profile_for_shard(shard_index)
    shard_seed = int(args.seed + shard_index * 999_983)
    command = [
        str(executable),
        "--games", str(args.games_per_shard),
        "--chunk-games", str(args.games_per_shard),
        "--threads", str(args.threads),
        "--depth", str(args.depth),
        "--time-ms", str(args.time_ms),
        "--max-plies", str(args.max_plies),
        "--positions", str(bank_path),
        "--nnue-weights", str(weights),
        "--seed", str(shard_seed),
        "--out", str(out_bin),
        "--meta-out", str(out_meta),
        "--mcab",
        "--mcab-root-noise",
        "--mcab-root-noise-alpha", "0.30",
        "--mcab-root-noise-epsilon", str(profile.root_noise_epsilon),
        "--mcab-root-select", "visits",
        "--mc-mode",
        "--mc-obvious-plies", "0",
        "--mc-temp-opening", str(profile.temp_opening),
        "--mc-temp-end", str(profile.temp_end),
        "--mc-temp-decay-plies", str(profile.temp_decay_plies),
        "--epsilon-midgame", str(profile.epsilon_midgame),
        "--playout-cap",
        "--full-search-prob", str(profile.full_search_prob),
        "--cheap-time-ms", str(args.cheap_time_ms),
        "--policy-order",
    ]
    return command, profile, shard_seed


def _existing_shards(out_dir: Path) -> tuple[int, int, list[dict]]:
    total = 0
    next_index = 0
    shards: list[dict] = []
    for v3_path in sorted(out_dir.glob("central_weakness_*.bin")):
        try:
            index = int(v3_path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        meta_path = v3_path.with_suffix(".meta")
        count = _pair_record_count(v3_path, meta_path)
        total += count
        next_index = max(next_index, index + 1)
        shards.append({"index": index, "records": count, "bin": v3_path.name, "meta": meta_path.name})
    return total, next_index, shards


def _campaign_identity(args: argparse.Namespace, executable: Path, weights: Path, bank_manifest: dict) -> dict:
    return {
        "schema": SCHEMA,
        "engine": str(executable),
        "engine_sha256": _sha256(executable),
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "bank_fingerprint": bank_manifest["fingerprint"],
        "bank_sha256": bank_manifest["bank_sha256"],
        "target_positions": args.target_positions,
        "games_per_shard": args.games_per_shard,
        "threads": args.threads,
        "depth": args.depth,
        "time_ms": args.time_ms,
        "cheap_time_ms": args.cheap_time_ms,
        "max_plies": args.max_plies,
        "seed": args.seed,
        "profile_cycle": list(PROFILE_CYCLE),
        "profiles": {name: profile.__dict__ for name, profile in ROLLOUT_PROFILES.items()},
    }


def _write_training_recipe(args: argparse.Namespace, out_dir: Path) -> dict:
    replay_dir = out_dir / "replay-50-50"
    recipe = {
        "schema": RECIPE_SCHEMA,
        "corpus": str(out_dir.resolve()),
        "replay_dataset": str((replay_dir / "dataset.npz").resolve()),
        "value_target": {
            "formula": "0.5 * signed_mcab_root_value + 0.5 * terminal_game_result",
            "stored_outcome_weight": args.stored_outcome_weight,
            "stored_gamma": args.stored_gamma,
            "uses_raw_nnue_eval_as_teacher": False,
        },
        "policy_target": "stored untempered top-eight MCAB root-visit distribution",
        "anti_overfit": [
            "geometry-only opening-bank generation (no NNUE acceptance filter)",
            "wide/balanced/sharp visit-temperature profiles",
            "Dirichlet root-noise variation by profile",
            "playout-cap cheap/full trajectory randomization",
            "canonical-state deduplication and duplicate-target averaging in stored replay",
            "game-grouped train/validation split",
        ],
        "recommended_network_training": {
            "initialize_from": "current production/champion weights rather than random initialization",
            "retain_general_strength": "mix 20-30% frozen broad/general replay before final training",
            "promotion_gate": "paired normal-book and Center-Rush external-opponent benchmarks; reject normal-book regression",
        },
    }
    _atomic_json(out_dir / "training_recipe.json", recipe)
    return recipe


def _build_replay_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "training" / "prepare_replay.py"),
        "--source", str(out_dir),
        "--out-dir", str(out_dir / "replay-50-50"),
        "--max-positions", str(args.replay_max_positions),
        "--seed", str(args.seed),
        "--val-fraction", str(args.val_fraction),
        "--mode", "stored_search",
        "--stored-gamma", str(args.stored_gamma),
        "--stored-outcome-weight", str(args.stored_outcome_weight),
    ]


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in text.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _resolve_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-book", type=Path, default=Path(CONFIG["opening_book"]))
    parser.add_argument("--out", type=Path, default=Path(CONFIG["out"]))
    parser.add_argument("--exe", type=Path, default=Path(CONFIG["exe"]))
    parser.add_argument("--weights", type=Path, default=Path(CONFIG["weights"]))
    parser.add_argument("--seed-bank-size", type=int, default=CONFIG["seed_bank_size"])
    parser.add_argument("--weak-opening-indices", type=_parse_int_list,
                        default=CONFIG["weak_opening_indices"])
    parser.add_argument("--target-positions", type=int, default=CONFIG["target_positions"])
    parser.add_argument("--replay-max-positions", type=int, default=CONFIG["replay_max_positions"])
    parser.add_argument("--games-per-shard", type=int, default=CONFIG["games_per_shard"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--depth", type=int, default=CONFIG["depth"])
    parser.add_argument("--time-ms", type=int, default=CONFIG["time_ms"])
    parser.add_argument("--cheap-time-ms", type=int, default=CONFIG["cheap_time_ms"])
    parser.add_argument("--max-plies", type=int, default=CONFIG["max_plies"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--val-fraction", type=float, default=CONFIG["val_fraction"])
    parser.add_argument("--prepare-replay", action=argparse.BooleanOptionalAction,
                        default=CONFIG["prepare_replay"])
    parser.add_argument("--rebuild-bank", action=argparse.BooleanOptionalAction,
                        default=CONFIG["rebuild_bank"])
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction,
                        default=CONFIG["dry_run"])
    args = parser.parse_args(argv)

    # Complex proposal distributions intentionally live in CONFIG so they are
    # visible/editable together instead of becoming unreadable CLI JSON.
    args.seed_ply_weights = dict(CONFIG["seed_ply_weights"])
    args.source_mix = dict(CONFIG["source_mix"])
    args.weak_exact_fraction = float(CONFIG["weak_exact_fraction"])
    args.branch_ply_weights = dict(CONFIG["branch_ply_weights"])
    args.mirror_probability = float(CONFIG["mirror_probability"])
    args.seed_pawn_mass = float(CONFIG["seed_pawn_mass"])
    args.seed_uniform_move_prob = float(CONFIG["seed_uniform_move_prob"])
    args.seed_proposal_temperature = float(CONFIG["seed_proposal_temperature"])
    args.seed_wall_margin = int(CONFIG["seed_wall_margin"])
    args.seed_cache_limit = int(CONFIG["seed_cache_limit"])
    args.stored_outcome_weight = float(CONFIG["stored_outcome_weight"])
    args.stored_gamma = float(CONFIG["stored_gamma"])

    if args.seed_bank_size < 1 or args.target_positions < 1 or args.replay_max_positions < 1:
        parser.error("bank and position targets must be positive")
    if args.games_per_shard < 1 or args.threads < 1 or args.depth < 1 or args.max_plies < 1:
        parser.error("games/threads/depth/max-plies must be positive")
    if not (0 < args.cheap_time_ms <= args.time_ms):
        parser.error("cheap-time-ms must be positive and <= time-ms")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must be in (0,1)")
    if not math.isclose(args.stored_outcome_weight, 0.5, abs_tol=1e-12):
        parser.error("this dedicated campaign requires an exact 0.50 stored outcome weight")
    if not math.isclose(args.stored_gamma, 1.0, abs_tol=1e-12):
        parser.error("this dedicated campaign keeps stored-gamma=1.0 for the literal 50/50 blend")
    return args


def main(argv=None) -> int:
    args = _resolve_args(argv)
    out_dir = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    opening_book = (ROOT / args.opening_book).resolve() if not args.opening_book.is_absolute() else args.opening_book.resolve()
    args.opening_book = opening_book

    planned = {
        "schema": SCHEMA,
        "opening_book": str(opening_book),
        "out": str(out_dir),
        "seed_bank_size": args.seed_bank_size,
        "seed_ply_weights": args.seed_ply_weights,
        "weak_opening_indices": list(args.weak_opening_indices),
        "target_positions": args.target_positions,
        "replay_max_positions": args.replay_max_positions,
        "value_blend": "50% MCAB root search value / 50% terminal result",
        "profiles": {name: profile.__dict__ for name, profile in ROLLOUT_PROFILES.items()},
        "profile_cycle": list(PROFILE_CYCLE),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(planned, indent=2), flush=True)
        return 0

    if not opening_book.exists():
        raise FileNotFoundError(opening_book)
    executable = _resolve_executable((ROOT / args.exe) if not args.exe.is_absolute() else args.exe)
    weights = ((ROOT / args.weights) if not args.weights.is_absolute() else args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)

    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "opening_bank_8to10ply.jsonl"
    bank_manifest_path = out_dir / "opening_bank.manifest.json"

    with OutputLock(out_dir):
        bank_manifest = _generate_opening_bank(args, bank_path, bank_manifest_path)
        identity = _campaign_identity(args, executable, weights, bank_manifest)
        identity_fingerprint = _stable_fingerprint(identity)
        manifest_path = out_dir / "campaign_manifest.json"

        total, next_shard, existing = _existing_shards(out_dir)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("identity_fingerprint") != identity_fingerprint:
                raise ValueError("campaign identity changed; use a new output directory")
            shard_log = list(previous.get("shards", existing))
        else:
            shard_log = existing

        _write_training_recipe(args, out_dir)
        stop_path = out_dir / "stop.request"
        while total < args.target_positions:
            if stop_path.exists():
                _atomic_json(manifest_path, {
                    **identity,
                    "identity_fingerprint": identity_fingerprint,
                    "status": "stopped",
                    "positions": total,
                    "target_positions": args.target_positions,
                    "next_shard": next_shard,
                    "shards": shard_log,
                })
                print(f"[central-weakness] stop requested at {total:,} positions", flush=True)
                return 0

            staging = out_dir / "staging"
            staging.mkdir(exist_ok=True)
            stage_bin = staging / f"central_weakness_{next_shard:05d}.bin.tmp"
            stage_meta = staging / f"central_weakness_{next_shard:05d}.meta.tmp"
            for path in (stage_bin, stage_meta):
                if path.exists():
                    path.unlink()
            final_bin = out_dir / f"central_weakness_{next_shard:05d}.bin"
            final_meta = out_dir / f"central_weakness_{next_shard:05d}.meta"

            command, profile, shard_seed = _build_selfplay_command(
                args, executable=executable, weights=weights, bank_path=bank_path,
                shard_index=next_shard, out_bin=stage_bin, out_meta=stage_meta,
            )
            print(
                f"[central-weakness] shard={next_shard:05d} profile={profile.name} "
                f"positions={total:,}/{args.target_positions:,}",
                flush=True,
            )
            started = time.time()
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"self-play shard {next_shard} failed with exit code {completed.returncode}"
                )
            produced = _pair_record_count(stage_bin, stage_meta)
            if produced == 0:
                raise RuntimeError(f"self-play shard {next_shard} produced zero recorded positions")
            _namespace_metadata_game_ids(stage_meta, next_shard, args.games_per_shard)
            stage_bin.replace(final_bin)
            stage_meta.replace(final_meta)

            elapsed = time.time() - started
            total += produced
            shard_log.append({
                "index": next_shard,
                "profile": profile.name,
                "seed": shard_seed,
                "records": produced,
                "seconds": elapsed,
                "bin": final_bin.name,
                "meta": final_meta.name,
                "command": command,
            })
            next_shard += 1
            _atomic_json(manifest_path, {
                **identity,
                "identity_fingerprint": identity_fingerprint,
                "status": "running" if total < args.target_positions else "selfplay_complete",
                "positions": total,
                "target_positions": args.target_positions,
                "next_shard": next_shard,
                "shards": shard_log,
            })

        recipe = _write_training_recipe(args, out_dir)
        replay_command = _build_replay_command(args, out_dir)
        if args.prepare_replay:
            print("[central-weakness] preparing 50/50 stored-search replay dataset", flush=True)
            completed = subprocess.run(replay_command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"prepare_replay failed with exit code {completed.returncode}")
            recipe["replay_status"] = "complete"
            _atomic_json(out_dir / "training_recipe.json", recipe)
        else:
            recipe["replay_status"] = "not_run"
            recipe["replay_command"] = replay_command
            _atomic_json(out_dir / "training_recipe.json", recipe)

        _atomic_json(manifest_path, {
            **identity,
            "identity_fingerprint": identity_fingerprint,
            "status": "complete",
            "positions": total,
            "target_positions": args.target_positions,
            "next_shard": next_shard,
            "shards": shard_log,
            "replay_prepared": bool(args.prepare_replay),
        })
        print(
            f"[central-weakness] complete: {total:,} recorded positions; "
            f"bank={args.seed_bank_size:,}; replay={'prepared' if args.prepare_replay else 'deferred'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[central-weakness] ERROR: {exc}", file=sys.stderr, flush=True)
        raise