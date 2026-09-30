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

The design target is sixteen million positions: eight million production,
four million contact-bucketed, and four million explicitly unassigned. The
first sixteen rollout plies are always searched at 200 ms. Later plies may
steer cheaply, but only 200 ms full searches become training records.
The deduplicated replay dataset is capped at twelve million positions.
The runner is resumable shard-by-shard.  Set ``dry_run`` to ``False`` in
``CONFIG`` (or pass ``--no-dry-run``) only when the self-play executable and
weights are ready.

Outputs under ``data/selfplay/central-weakness-rollouts-16m``:

- ``opening_bank_8to10ply.jsonl``: generated start positions;
- ``opening_bank.manifest.json``: reproducible bank identity and statistics;
- ``<source>/central_weakness_XXXXX.bin/.meta``: V3 self-play + aligned search metadata;
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
import shutil
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
from tools.teacher.selfplay_unique_store import V3_DTYPE  # noqa: E402

SCHEMA = "zquoridor.central_weakness_rollouts.v2"
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

CAMPAIGN_TOTAL_TARGET = 16_000_000
SELFPLAY_SOURCES = (
    {"name": "production_bucketed", "architecture": "multipath_phase_bucketed",
     "features": 504, "target_positions": 8_000_000,
     "weights": "data/nnue/nnue_weights_int8.bin", "executable": "bin/selfplay"},
    {"name": "contact_bucketed", "architecture": "multipath_phase_contact_bucketed",
     "features": 858, "target_positions": 4_000_000,
     "weights": "results/experiments/multipath_contact_bucketed_unified/student_int8.bin",
     "architecture_manifest": "results/experiments/multipath_contact_bucketed_unified/student.architecture.json",
     "executable": "bin/selfplay_multipath_phase_contact_bucketed"},
)
RESERVE_UNASSIGNED = CAMPAIGN_TOTAL_TARGET - sum(s["target_positions"] for s in SELFPLAY_SOURCES)

# Edit this block for normal use.  CLI options override the scalar/path fields.
CONFIG = {
    "opening_book": "tools/external/openings_center_rush_50pairs.jsonl",
    "out": "data/selfplay/central-weakness-rollouts-16m",
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
    "replay_max_positions": 12_000_000,
    "games_per_shard": 512,
    "threads": 10,
    "depth": 50,
    "time_ms": 200,
    "cheap_time_ms": 20,
    "full_search_opening_plies": 16,
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
        if previous.get("bank_sha256") != _sha256(bank_path):
            raise ValueError("existing opening bank content changed")
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
        if len(bank_rows) % 1_000 == 0:
            print(f"[central-weakness] opening bank: {len(bank_rows):,}/{args.seed_bank_size:,} legal roots", flush=True)
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


def _retain_complete_search_records(v3_path: Path, meta_path: Path) -> int:
    """Keep only aligned V3 rows with usable searched root value and visits."""
    _pair_record_count(v3_path, meta_path)
    rows = np.fromfile(v3_path, dtype=V3_DTYPE)
    metadata = np.fromfile(meta_path, dtype=META_DTYPE)
    valid = ((metadata["flags"] & 2) != 0) & np.isfinite(metadata["root"])
    valid &= (metadata["root"] >= 0) & (metadata["root"] <= 1)
    valid &= rows["policy_top_prob"].sum(axis=1) > 0
    dropped = int(len(rows) - np.count_nonzero(valid))
    if dropped:
        rows[valid].tofile(v3_path)
        metadata[valid].tofile(meta_path)
    return dropped


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
    source_index: int = 0,
) -> tuple[list[str], RolloutProfile, int]:
    profile = _profile_for_shard(shard_index)
    # Every source sees the same profile and opening-bank schedule at a given
    # local shard index. The source offset keeps RNG streams independent.
    shard_seed = int(args.seed + shard_index * 999_983 + source_index * 104_729)
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
        "--full-search-opening-plies", str(args.full_search_opening_plies),
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
    for meta_path in out_dir.glob("central_weakness_*.meta"):
        if not meta_path.with_suffix(".bin").exists():
            raise ValueError(f"orphan metadata shard: {meta_path}")
    return total, next_index, shards


def _recover_partial_shards(source_dir: Path) -> None:
    """Preserve incomplete admissions outside the accepted shard namespace."""
    candidates = {p.stem for p in source_dir.glob("central_weakness_*.bin")}
    candidates.update(p.stem for p in source_dir.glob("central_weakness_*.meta"))
    for stem in sorted(candidates):
        bin_path = source_dir / f"{stem}.bin"
        meta_path = source_dir / f"{stem}.meta"
        sidecar = source_dir / f"{stem}.shard.json"
        if bin_path.is_file() and meta_path.is_file() and sidecar.is_file():
            _pair_record_count(bin_path, meta_path)
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            if (info.get("bin_sha256") != _sha256(bin_path)
                    or info.get("meta_sha256") != _sha256(meta_path)):
                raise ValueError(f"accepted shard SHA mismatch: {stem}")
            continue
        orphan_dir = source_dir / "staging" / f"interrupted_{stem}_{time.time_ns()}"
        orphan_dir.mkdir(parents=True)
        for path in (bin_path, meta_path, sidecar):
            if path.exists():
                path.replace(orphan_dir / path.name)


def _source_path(value: str) -> Path:
    path = ROOT / value
    if os.name == "nt" and path.suffix.lower() != ".exe" and value.startswith("bin/"):
        path = path.with_suffix(".exe")
    return path.resolve()


def _source_artifacts(source: Mapping) -> dict:
    weights = _source_path(source["weights"])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    manifest_path = _source_path(source["architecture_manifest"]) if source.get("architecture_manifest") else None
    architecture = None
    flags = []
    if manifest_path:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        architecture = json.loads(manifest_path.read_text(encoding="utf-8"))
        if architecture["architecture"] != source["architecture"] or architecture["features"] != source["features"]:
            raise ValueError(f"architecture manifest disagrees with source {source['name']}")
        flags = architecture["cpp_flags"]
        if not isinstance(flags, list) or not flags or not all(isinstance(flag, str) and flag.startswith("-DZQ_NNUE_") for flag in flags):
            raise ValueError(f"invalid cpp_flags in {manifest_path}")
        expected = architecture.get("int8_sha256")
        if expected and _sha256(weights) != expected:
            raise ValueError(f"weights SHA disagrees with {manifest_path}")
    # Quantized layouts have a fixed byte count per feature for this 512-wide
    # architecture. This rejects a swapped 504/858-input file before search.
    expected_bytes = {504: 731_172, 858: 1_093_668}.get(source["features"])
    if expected_bytes and weights.stat().st_size != expected_bytes:
        raise ValueError(f"weights layout/feature count mismatch: {weights}")
    return {"weights": weights, "weights_sha256": _sha256(weights),
            "architecture_manifest": manifest_path,
            "architecture_manifest_sha256": _sha256(manifest_path) if manifest_path else None,
            "cpp_flags": flags}


def _ensure_executable(source: Mapping, artifacts: dict) -> tuple[Path, dict]:
    compiler = shutil.which("g++") or ("C:/mingw64/bin/g++.exe" if Path("C:/mingw64/bin/g++.exe").is_file() else None)
    if not compiler:
        raise RuntimeError("g++ is required to build architecture-specific self-play")
    compiler = str(Path(compiler).resolve())
    executable = _source_path(source["executable"])
    build_manifest_path = executable.with_name(executable.name + ".build.json")
    inputs = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "compiler": compiler, "compiler_sha256": _sha256(Path(compiler)),
              "architecture": source["architecture"], "features": source["features"],
              "cpp_flags": artifacts["cpp_flags"], "weights_sha256": artifacts["weights_sha256"],
              "architecture_manifest_sha256": artifacts["architecture_manifest_sha256"],
              "source_sha256": _sha256(ROOT / "tools/selfplay/selfplay_main.cpp"),
              "headers_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in
                                 sorted([*(ROOT / "src").glob("*.hpp"),
                                         *(ROOT / "tools/selfplay").glob("*.hpp")])}}
    if executable.is_file() and build_manifest_path.is_file():
        previous = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        if previous.get("inputs") == inputs and previous.get("executable_sha256") == _sha256(executable):
            return executable, previous
    executable.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-std=c++17", "-pthread", "-march=native"]
    if os.name == "nt" or (os.uname().machine.lower() in ("x86_64", "amd64")):
        command += ["-mavx2", "-mfma"]
    command += artifacts["cpp_flags"] + ["-I" + str(ROOT / "src"),
               "-I" + str(ROOT / "tools/selfplay"), "-o", str(executable),
               str(ROOT / "tools/selfplay/selfplay_main.cpp")]
    subprocess.run(command, cwd=ROOT, check=True)
    build = {"inputs": inputs, "command": command, "build_timestamp": time.time(),
             "executable_sha256": _sha256(executable)}
    _atomic_json(build_manifest_path, build)
    return executable, build


def _campaign_identity(args: argparse.Namespace, sources: list[dict], bank_manifest: dict) -> dict:
    return {
        "schema": SCHEMA,
        "sources": sources,
        "campaign_total_target": CAMPAIGN_TOTAL_TARGET,
        "assigned_target": sum(s["target_positions"] for s in SELFPLAY_SOURCES),
        "source_plan": [dict(s) for s in SELFPLAY_SOURCES],
        "reserve_unassigned": RESERVE_UNASSIGNED,
        "bank_fingerprint": bank_manifest["fingerprint"],
        "bank_sha256": bank_manifest["bank_sha256"],
        "games_per_shard": args.games_per_shard,
        "threads": args.threads,
        "depth": args.depth,
        "time_ms": args.time_ms,
        "cheap_time_ms": args.cheap_time_ms,
        "full_search_opening_plies": args.full_search_opening_plies,
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
        "mode": "stored_search",
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
            f"200 ms full search for first {args.full_search_opening_plies} plies after seed; later cheap trajectory plies are not recorded",
            "canonical-state deduplication and duplicate-target averaging in stored replay",
            "game-grouped train/validation split",
        ],
        "recommended_network_training": {
            "initialize_from": "current production/champion weights rather than random initialization",
            "central_fraction": "70-80% of training replay",
            "frozen_general_background_fraction": "20-30% of training replay",
            "retain_general_strength": "include normal book, general midgame, endgame, wall fights and noncentral Claustrophobia",
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
    parser.add_argument("--source", choices=[s["name"] for s in SELFPLAY_SOURCES],
                        help="run just one assigned source (useful for smoke tests)")
    parser.add_argument("--seed-bank-size", type=int, default=CONFIG["seed_bank_size"])
    parser.add_argument("--weak-opening-indices", type=_parse_int_list,
                        default=CONFIG["weak_opening_indices"])
    parser.add_argument("--target-positions", type=int,
                        help="override selected source target for an isolated test corpus")
    parser.add_argument("--replay-max-positions", type=int, default=CONFIG["replay_max_positions"])
    parser.add_argument("--games-per-shard", type=int, default=CONFIG["games_per_shard"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--depth", type=int, default=CONFIG["depth"])
    parser.add_argument("--time-ms", type=int, default=CONFIG["time_ms"])
    parser.add_argument("--cheap-time-ms", type=int, default=CONFIG["cheap_time_ms"])
    parser.add_argument("--full-search-opening-plies", type=int, default=CONFIG["full_search_opening_plies"])
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

    if args.seed_bank_size < 1 or (args.target_positions is not None and args.target_positions < 1) or args.replay_max_positions < 1:
        parser.error("bank and position targets must be positive")
    if args.games_per_shard < 1 or args.threads < 1 or args.depth < 1 or args.max_plies < 1:
        parser.error("games/threads/depth/max-plies must be positive")
    if args.games_per_shard >= 1 << 32:
        parser.error("games-per-shard exceeds the uint64 source namespace")
    if args.time_ms != 200:
        parser.error("this campaign requires --time-ms 200 for full-search plies")
    if not 0 < args.cheap_time_ms <= args.time_ms or args.full_search_opening_plies < 1:
        parser.error("invalid cheap time or opening full-search window")
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
    selected = [dict(s) for s in SELFPLAY_SOURCES if not args.source or s["name"] == args.source]
    if args.target_positions is not None:
        for source in selected:
            source["target_positions"] = args.target_positions
    if not args.source and not args.target_positions and sum(s["target_positions"] for s in selected) + RESERVE_UNASSIGNED != CAMPAIGN_TOTAL_TARGET:
        raise ValueError("source quotas and reserve do not sum to campaign target")
    if args.dry_run:
        completed = {s["name"]: _existing_shards(out_dir / s["name"])[0] for s in selected}
        print("CENTRAL WEAKNESS MULTI-NETWORK CAMPAIGN")
        print(f"Total design target: {CAMPAIGN_TOTAL_TARGET:,} positions")
        print("Assigned:")
        for s in selected:
            print(f"  {s['name']:<25} {s['target_positions']:>10,}  {s['target_positions']/CAMPAIGN_TOTAL_TARGET:5.1%}")
        print(f"Reserve: unassigned {RESERVE_UNASSIGNED:,} ({RESERVE_UNASSIGNED/CAMPAIGN_TOTAL_TARGET:.1%})")
        print(f"Search: 200 ms for first {args.full_search_opening_plies} plies after seed; "
              f"then {args.cheap_time_ms} ms cheap trajectory plies; only 200 ms searches recorded")
        print(f"Opening bank: {args.seed_bank_size:,} roots; 8-10 plies; classic central prefix; weak-line oversampling; mirrored")
        print("Profiles: wide / balanced / sharp (same local cycle for every source)")
        print("Value replay: 50% MCAB root search value + 50% terminal result")
        for s in selected:
            print(f"{s['name']}: architecture={s['architecture']} features={s['features']} "
                  f"weights={_source_path(s['weights'])} executable={_source_path(s['executable'])} "
                  f"target={s['target_positions']:,} completed={completed[s['name']]:,} "
                  f"remaining={max(0,s['target_positions']-completed[s['name']]):,}")
        return 0

    if not opening_book.exists():
        raise FileNotFoundError(opening_book)
    runtime = {}
    for s in selected:
        artifacts = _source_artifacts(s)
        executable, build = _ensure_executable(s, artifacts)
        runtime[s["name"]] = (artifacts, executable, build)

    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "opening_bank_8to10ply.jsonl"
    bank_manifest_path = out_dir / "opening_bank.manifest.json"

    with OutputLock(out_dir):
        print(f"[central-weakness] preparing shared opening bank ({args.seed_bank_size:,} roots)", flush=True)
        bank_manifest = _generate_opening_bank(args, bank_path, bank_manifest_path)
        source_manifest = []
        for s in selected:
            artifacts, executable, build = runtime[s["name"]]
            source_manifest.append({**s, "weights": str(artifacts["weights"]),
                                    "weights_sha256": artifacts["weights_sha256"],
                                    "architecture_manifest": str(artifacts["architecture_manifest"]) if artifacts["architecture_manifest"] else None,
                                    "architecture_manifest_sha256": artifacts["architecture_manifest_sha256"],
                                    "executable": str(executable), "executable_sha256": build["executable_sha256"],
                                    "build_manifest": str(executable) + ".build.json",
                                    "cpp_flags": artifacts["cpp_flags"], "build": build})
        identity = _campaign_identity(args, source_manifest, bank_manifest)
        identity_fingerprint = _stable_fingerprint(identity)
        manifest_path = out_dir / "campaign_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("identity_fingerprint") != identity_fingerprint:
                raise ValueError("campaign identity changed; use a new output directory")
        _write_training_recipe(args, out_dir)
        stop_path = out_dir / "stop.request"
        for source in selected:
            source_dir = out_dir / source["name"]
            source_dir.mkdir(exist_ok=True)
            _recover_partial_shards(source_dir)
        def snapshot(status):
            entries = []
            progress = {}
            for source in selected:
                count, next_index, shards = _existing_shards(out_dir / source["name"])
                progress[source["name"]] = {"completed": count, "target": source["target_positions"],
                                            "remaining": max(0, source["target_positions"] - count),
                                            "next_shard": next_index}
                for shard in shards:
                    sidecar = out_dir / source["name"] / shard["bin"].replace(".bin", ".shard.json")
                    provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                    entries.append({"source": source["name"], "architecture": source["architecture"],
                                    "v3": f"{source['name']}/{shard['bin']}",
                                    "meta": f"{source['name']}/{shard['meta']}", **shard,
                                    **provenance})
            _atomic_json(manifest_path, {**identity, "identity_fingerprint": identity_fingerprint,
                                         "status": status, "progress": progress, "shards": entries,
                                         "positions": sum(v["completed"] for v in progress.values())})

        snapshot("running")
        for source in selected:
            source_index = next(i for i, item in enumerate(SELFPLAY_SOURCES) if item["name"] == source["name"])
            source_dir = out_dir / source["name"]
            artifacts, executable, _build = runtime[source["name"]]
            total, next_shard, _ = _existing_shards(source_dir)
            while total < source["target_positions"]:
                if next_shard >= 1 << 32:
                    raise OverflowError("source shard namespace exhausted")
                if stop_path.exists():
                    snapshot("stopped")
                    print(f"[central-weakness] stop requested at {total:,} {source['name']} positions", flush=True)
                    return 0
                staging = source_dir / "staging"
                staging.mkdir(exist_ok=True)
                stage_bin = staging / f"central_weakness_{next_shard:05d}.bin.tmp"
                stage_meta = staging / f"central_weakness_{next_shard:05d}.meta.tmp"
                for path in (stage_bin, stage_meta):
                    if path.exists():
                        interrupted = staging / f"interrupted_{next_shard}_{time.time_ns()}"
                        interrupted.mkdir(exist_ok=True)
                        path.replace(interrupted / path.name)
                final_bin = source_dir / f"central_weakness_{next_shard:05d}.bin"
                final_meta = source_dir / f"central_weakness_{next_shard:05d}.meta"
                command, profile, shard_seed = _build_selfplay_command(
                    args, executable=executable, weights=artifacts["weights"], bank_path=bank_path,
                    shard_index=next_shard, source_index=source_index, out_bin=stage_bin, out_meta=stage_meta)
                print(f"[central-weakness] source={source['name']} shard={next_shard:05d} "
                      f"profile={profile.name} positions={total:,}/{source['target_positions']:,}", flush=True)
                started = time.time()
                subprocess.run(command, cwd=ROOT, check=True)
                dropped = _retain_complete_search_records(stage_bin, stage_meta)
                produced = _pair_record_count(stage_bin, stage_meta)
                if produced == 0:
                    raise RuntimeError(f"self-play shard {next_shard} produced zero positions")
                _namespace_metadata_game_ids(stage_meta, source_index * (1 << 32) + next_shard,
                                             args.games_per_shard)
                stage_bin.replace(final_bin)
                stage_meta.replace(final_meta)
                shard_info = {"source": source["name"], "architecture": source["architecture"],
                              "profile": profile.name, "seed": shard_seed, "records": produced,
                              "incomplete_search_records_dropped": dropped,
                              "seconds": time.time() - started, "time_ms": args.time_ms,
                              "bin_sha256": _sha256(final_bin), "meta_sha256": _sha256(final_meta),
                              "weights_sha256": artifacts["weights_sha256"],
                              "executable_sha256": _sha256(executable),
                              "opening_bank_sha256": bank_manifest["bank_sha256"]}
                _atomic_json(final_bin.with_suffix(".shard.json"), shard_info)
                total += produced
                next_shard += 1
                snapshot("running")

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

        all_sources_selected = len(selected) == len(SELFPLAY_SOURCES)
        campaign_assigned_complete = all_sources_selected and args.target_positions is None
        snapshot("assigned_sources_complete_reserve_unassigned" if campaign_assigned_complete and RESERVE_UNASSIGNED else
                 "complete" if campaign_assigned_complete else "test_targets_complete" if args.target_positions else "selected_sources_complete")
        print(f"{'Assigned' if campaign_assigned_complete else 'Test/selected'} sources complete: "
              f"{sum(_existing_shards(out_dir / s['name'])[0] for s in selected):,}", flush=True)
        print(f"Campaign reserve still unassigned: {RESERVE_UNASSIGNED:,}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[central-weakness] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
