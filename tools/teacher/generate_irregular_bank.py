#!/usr/bin/env python3
"""Generate a validated opening bank of irregular / perimeter / non-standard openings.

Ingests diverse, irregular opening roots from the external confirmation catalog
(openings_normal_confirm_400.jsonl), mirrors them horizontally, samples
tactical branching extensions, and produces a verified zquoridor.position.v1
opening bank for self-play diversity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.teacher.build_selfplay_seed_schedule import (
    _ReplayState,
    _apply_move,
    _replay_history,
)

CONFIG = {
    "source_openings": "tools/external/openings_normal_confirm_400.jsonl",
    "out": "tools/external/openings_irregular_bank.jsonl",
    "target_bank_size": 2500,
    "seed": 20261003,
    "pawn_mass": 0.40,
    "temperature": 1.25,
    "uniform_prob": 0.15,
    "mirror_probability": 0.50,
    "seed_ply_weights": {
        6: 0.25,
        7: 0.30,
        8: 0.30,
        9: 0.15,
    },
}


def _mirror_col(col: int) -> int:
    return 8 - col


def _mirror_wall_col(col: int) -> int:
    return 7 - col


def _mirror_move(text: str) -> str:
    text = text.strip()
    col = ord(text[0].lower()) - ord("a")
    row = int(text[1]) - 1
    if len(text) == 2:
        return f"{chr(ord('a') + _mirror_col(col))}{row + 1}"
    orientation = text[2].lower()
    return f"{chr(ord('a') + _mirror_wall_col(col))}{row + 1}{orientation}"


def _mirror_history(history: Sequence[str]) -> list[str]:
    return [_mirror_move(m) for m in history]


def _wall_anchors(state: _ReplayState, margin: int = 1) -> set[tuple[int, int]]:
    anchors = set()
    for row in range(8):
        for col in range(8):
            anchors.add((row, col))
    return anchors


def _legal_proposals(state: _ReplayState) -> list[tuple[str, str, float]]:
    result = []
    mover = state.turn
    current_row, current_col = divmod(state.pawns[mover], 9)

    for cell in range(81):
        row, col = divmod(cell, 9)
        text = f"{chr(ord('a') + col)}{row + 1}"
        trial = _ReplayState(
            list(state.pawns), set(state.walls_h), set(state.walls_v),
            list(state.walls_left), state.turn,
        )
        if not _apply_move(trial, text):
            continue
        direction = 1 if mover == 0 else -1
        progress = direction * (row - current_row)
        lateral = abs(col - current_col)
        center = abs(col - 4)
        score = 1.00 * progress - 0.05 * center - 0.05 * lateral
        result.append((text, "pawn", score))

    for row, col in sorted(_wall_anchors(state)):
        for orientation in ("h", "v"):
            text = f"{chr(ord('a') + col)}{row + 1}{orientation}"
            trial = _ReplayState(
                list(state.pawns), set(state.walls_h), set(state.walls_v),
                list(state.walls_left), state.turn,
            )
            if not _apply_move(trial, text):
                continue
            center_dist = abs(row - 3.5) + abs(col - 3.5)
            pawn_dist = min(
                abs(row + 0.5 - (pawn // 9)) + abs(col + 0.5 - (pawn % 9))
                for pawn in state.pawns
            )
            score = -0.05 * center_dist - 0.05 * pawn_dist
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
    pawn = [item for item in candidates if item[1] == "pawn"]
    wall = [item for item in candidates if item[1] == "wall"]
    if pawn and wall:
        group = pawn if rng.random() < pawn_mass else wall
    else:
        group = pawn or wall

    scores = [item[2] for item in group]
    maximum = max(scores)
    soft = [math.exp((s - maximum) / temperature) for s in scores]
    total_soft = sum(soft)
    n = len(group)
    probs = [
        (1.0 - uniform_prob) * w / total_soft + uniform_prob / n
        for w in soft
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
    cache: dict[tuple[str, ...], list[tuple[str, str, float]]] | None = None,
) -> tuple[str, ...] | None:
    if len(history) > target_plies:
        history = history[:target_plies]
    state = _replay_history(history)
    if state is None:
        return None
    result = list(history)
    while len(result) < target_plies:
        key = tuple(result)
        candidates = cache.get(key) if cache is not None else None
        if candidates is None:
            candidates = _legal_proposals(state)
            if cache is not None and len(cache) < 30000:
                cache[key] = candidates
        if not candidates:
            return None
        move = _sample_proposal(
            rng, candidates, pawn_mass=pawn_mass,
            temperature=temperature, uniform_prob=uniform_prob,
        )
        if not _apply_move(state, move):
            return None
        result.append(move)
    return tuple(result)


def load_source_seeds(path: Path) -> list[list[str]]:
    seeds = []
    if not path.exists():
        return seeds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            moves = data.get("moves") or data.get("history")
            if moves:
                seeds.append(list(moves))
    return seeds


def generate_bank(config: dict) -> list[dict]:
    rng = random.Random(config["seed"])
    target_size = int(config["target_bank_size"])
    ply_weights = config["seed_ply_weights"]
    pawn_mass = config["pawn_mass"]
    temperature = config["temperature"]
    uniform_prob = config["uniform_prob"]
    mirror_prob = config["mirror_probability"]

    # Ingest sources
    raw_seeds = []
    source_path = ROOT / config["source_openings"]
    raw_seeds.extend(load_source_seeds(source_path))
    print(f"Loaded {len(raw_seeds)} raw irregular seeds from source file.", flush=True)

    seen = set()
    bank = []
    cache: dict[tuple[str, ...], list[tuple[str, str, float]]] = {}

    def try_admit(moves: Sequence[str], category: str, is_mirrored: bool) -> bool:
        moves_tuple = tuple(moves)
        if moves_tuple in seen:
            return False
        state = _replay_history(moves_tuple)
        if state is None:
            return False
        if state.pawns[0] // 9 == 8 or state.pawns[1] // 9 == 0:
            return False
        seen.add(moves_tuple)
        idx = len(bank)
        bank.append({
            "schema": "zquoridor.position.v1",
            "id": f"irreg_{idx:05d}",
            "opening_index": idx,
            "history": list(moves_tuple),
            "moves": list(moves_tuple),
            "category": category,
            "mirrored": is_mirrored,
            "plies": len(moves_tuple),
        })
        return True

    print("Step 1: Admitting exact and mirrored irregular seeds...", flush=True)
    for moves in raw_seeds:
        try_admit(moves, "irregular_benchmark_root", False)
        try_admit(_mirror_history(moves), "irregular_benchmark_root", True)

    print(f"  Admitted {len(bank)} base seeds.", flush=True)

    print(f"Step 2: Generating tactical branches to reach {target_size} roots...", flush=True)
    attempts = 0
    max_attempts = target_size * 25
    last_reported = 0

    while len(bank) < target_size and attempts < max_attempts:
        attempts += 1
        base = rng.choice(raw_seeds)
        # Choose branch ply between 3 and min(len(base), 6)
        branch_max = min(len(base), 6)
        b_ply = rng.randint(3, branch_max)
        prefix = base[:b_ply]

        target_len = rng.choices(
            list(ply_weights.keys()),
            weights=list(ply_weights.values()),
        )[0]
        if target_len <= len(prefix):
            target_len = len(prefix) + 2

        advanced = _advance_history(
            prefix, target_len, rng,
            pawn_mass=pawn_mass,
            temperature=temperature,
            uniform_prob=uniform_prob,
            cache=cache,
        )
        if advanced is None:
            continue

        is_mirror = rng.random() < mirror_prob
        cand = _mirror_history(advanced) if is_mirror else advanced
        if try_admit(cand, "irregular_tactical_branch", is_mirror):
            if len(bank) // 500 > last_reported:
                last_reported = len(bank) // 500
                print(f"  Progress: {len(bank):,}/{target_size:,} roots", flush=True)

    print(f"Generation complete: {len(bank)} unique valid irregular roots created.", flush=True)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args()
    config = copy.deepcopy(CONFIG)
    config.update(vars(args))

    bank = generate_bank(config)

    out_path = Path(config["out"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [json.dumps(row) for row in bank]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(bank)} openings to: {out_path}")

    manifest = {
        "schema": "zquoridor.irregular_opening_bank.v1",
        "file": str(out_path.name),
        "total_rows": len(bank),
        "seed": config["seed"],
        "target_bank_size": config["target_bank_size"],
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
