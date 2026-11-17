#!/usr/bin/env python3
"""Generate a targeted opening bank focused on empirical engine weaknesses.

This generator takes the exact catalog openings and center-rush lines where
Zquoridor 3.0 showed score deficits against external benchmarks, generates
tactical branching variations around the key inflection plies (plies 6 to 9),
applies horizontal mirroring, and outputs a validated JSONL opening bank ready
for selfplay rollouts.
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

from tools.external.local_arena import Referee
from tools.teacher.build_selfplay_seed_schedule import (
    _ReplayState,
    _apply_move,
    _replay_history,
)

# Canonical empirically weak seeds identified from recent benchmark losses
EMPIRICAL_WEAK_SEEDS = [
    # 1. Pure root collision lines from empty-board match
    {
        "category": "pure_center_rush_clash",
        "name": "clash_g1_f3h_e4v",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "f3h", "e4v"],
        "branch_plies": [6, 7],
    },
    {
        "category": "pure_center_rush_clash",
        "name": "clash_g2_h3h_c6h",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "h3h", "c6h", "d5v"],
        "branch_plies": [6, 7, 8],
    },
    # 2. Subfamily early_dev_4 (30.0% score vs Claustrophobia)
    {
        "category": "early_dev_4_flank",
        "name": "sound5k_4539",
        "moves": ["e2", "e8", "d4v", "e7", "e3", "d7h", "f3", "d7", "d6h", "e7"],
        "branch_plies": [4, 6, 8],
    },
    {
        "category": "early_dev_4_flank",
        "name": "sound5k_3018",
        "moves": ["e2", "e8", "d4v", "e7", "d2", "c2v", "e6h", "d2v", "e1v", "c3h"],
        "branch_plies": [4, 6, 8],
    },
    {
        "category": "early_dev_4_flank",
        "name": "sound5k_4070",
        "moves": ["e2", "e8", "d4v", "e7", "e3", "d7h", "d2h", "f7h", "d3", "e6"],
        "branch_plies": [4, 6, 8],
    },
    # 3. Subfamily early_dev_0 (33.3% score)
    {
        "category": "early_dev_0_pawn_d2",
        "name": "sound5k_1023",
        "moves": ["e2", "e8", "d2", "c5h", "e2h", "e7", "d1v", "c2h"],
        "branch_plies": [3, 5, 7],
    },
    {
        "category": "early_dev_0_pawn_d2",
        "name": "cr_3688",
        "moves": ["e2", "e8", "d2", "c6h", "e6h", "f8", "f7v", "f7"],
        "branch_plies": [3, 5, 7],
    },
    # 4. Subfamily early_dev_1 (40.0% score)
    {
        "category": "early_dev_1_f2h",
        "name": "sound5k_2109",
        "moves": ["e2", "e8", "e3", "e7", "f2h", "e6", "d5v", "e6h", "e5h", "f6"],
        "branch_plies": [4, 6, 8],
    },
    {
        "category": "early_dev_1_f2h",
        "name": "sound5k_2566",
        "moves": ["e2", "e8", "e3", "e7", "e6h", "d3h", "d7v", "d2v", "e7v", "f7h"],
        "branch_plies": [4, 6, 8],
    },
    {
        "category": "early_dev_1_f2h",
        "name": "cr_394",
        "moves": ["e2", "e8", "e3", "e7", "e4", "d7h", "d6h", "e8v"],
        "branch_plies": [5, 6, 7],
    },
    # 5. Classic rush tactical 0-2 lost pairs
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_3123",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "d7h", "d4", "c6h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_1605",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e5", "f4v", "c5h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_2529",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6h", "d3h", "e6v"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_2908",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "d7h", "d3v", "f6h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_3237",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "d6h", "d3v", "f6"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_3104",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "c3h", "e6h", "d4", "d6v"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_41",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e7h", "c4h", "d6h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_75",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e5", "f4v", "d6h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_297",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "c3h", "e7h", "f4", "f4v"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_3823",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "d7h", "f4", "d6"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_4748",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6h", "d4h", "f5"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_3301",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "c3h", "d7h", "f3h", "f6h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_4559",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "d6h", "f4", "d6"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_4373",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e5", "d4h", "c5h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_2042",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e6h", "d3v", "f7h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_3867",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6h", "e3h", "f7h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "sound5k_1853",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "d3h", "e5", "f3h", "e6h"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "cr_3152",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "d6h", "d3v", "d6"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "cr_688",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e7h", "d3v", "f6"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "cr_1217",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6h", "e3h", "e5v"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "cr_3250",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "c3h", "e6h", "e3h", "d6v"],
        "branch_plies": [6, 7, 8],
    },
    {
        "category": "classic_rush_tactical",
        "name": "cr_67",
        "moves": ["e2", "e8", "e3", "d6h", "e7h", "e3h", "f3", "d8"],
        "branch_plies": [4, 6, 7],
    },
    {
        "category": "classic_rush_tactical",
        "name": "cr_3090",
        "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "f3h", "d7h", "d4", "c4v"],
        "branch_plies": [6, 7, 8],
    },
    # 6. Normal book 0-2 lost pairs (flank/perimeter wall development)
    {
        "category": "perimeter_wall_transition",
        "name": "normal_6",
        "moves": ["c4h", "c7h", "g6h", "g2v", "f2v", "d2v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_12",
        "moves": ["d3v", "b1v", "a7h", "g8h", "d1v", "d4h"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_15",
        "moves": ["b5h", "f6h", "d5h", "b2h", "b7h", "h2h"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_16",
        "moves": ["d4h", "b4h", "f2v", "c7v", "e6v", "a7v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_17",
        "moves": ["e6v", "b1v", "f4v", "a7v", "a6h", "d1v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_23",
        "moves": ["g6h", "a4h", "b8v", "h4h", "f2v", "c8v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_25",
        "moves": ["c6h", "h8v", "f1h", "g4h", "c4v", "d2v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_51",
        "moves": ["h6h", "a1h", "b7v", "c3v", "g7v", "a5h"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_55",
        "moves": ["h4v", "g3h", "b8v", "b2h", "g5v", "g8v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_88",
        "moves": ["b6h", "g8h", "a5h", "a1v", "a3h", "c1v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_91",
        "moves": ["b8h", "g2v", "d1", "g1h", "f7h", "b5v"],
        "branch_plies": [4, 5, 6],
    },
    {
        "category": "perimeter_wall_transition",
        "name": "normal_94",
        "moves": ["b5v", "g4v", "f6v", "e5h", "d8v", "b8h"],
        "branch_plies": [4, 5, 6],
    },
]

CONFIG = {
    "out": "tools/external/openings_targeted_weakness_bank.jsonl",
    "target_bank_size": 2500,
    "seed_ply_weights": {8: 0.50, 9: 0.35, 10: 0.15},
    "mirror_probability": 0.50,
    "pawn_mass": 0.42,
    "temperature": 1.35,
    "uniform_prob": 0.20,
    "seed": 20261003,
}


def _mirror_move_text(text: str) -> str:
    col = ord(text[0].lower()) - ord("a")
    row = text[1]
    if len(text) == 2:
        return chr(ord("a") + (8 - col)) + row
    return chr(ord("a") + (7 - col)) + row + text[2].lower()


def _mirror_history(history: Sequence[str]) -> tuple[str, ...]:
    return tuple(_mirror_move_text(m) for m in history)


def _wall_anchors(state: _ReplayState, margin: int = 1) -> set[tuple[int, int]]:
    anchors = {
        (row, col)
        for row in range(margin, 8 - margin)
        for col in range(margin, 8 - margin)
    }
    for pawn in state.pawns:
        prow, pcol = divmod(pawn, 9)
        for row in range(max(0, prow - 2), min(8, prow + 2)):
            for col in range(max(0, pcol - 2), min(8, pcol + 2)):
                anchors.add((row, col))
    return anchors


def _legal_proposals(state: _ReplayState, wall_margin: int = 1) -> list[tuple[str, str, float]]:
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
        score = 1.10 * progress - 0.12 * center - 0.08 * lateral
        result.append((text, "pawn", score))

    for row, col in sorted(_wall_anchors(state, wall_margin)):
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


def generate_bank(config: dict) -> list[dict]:
    rng = random.Random(config["seed"])
    target_size = int(config["target_bank_size"])
    ply_weights = config["seed_ply_weights"]
    pawn_mass = config["pawn_mass"]
    temperature = config["temperature"]
    uniform_prob = config["uniform_prob"]
    mirror_prob = config["mirror_probability"]

    seen = set()
    bank = []
    cache: dict[tuple[str, ...], list[tuple[str, str, float]]] = {}

    def try_admit(moves: Sequence[str], category: str, seed_name: str, is_mirrored: bool) -> bool:
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
            "id": f"tw_{idx:05d}",
            "opening_index": idx,
            "history": list(moves_tuple),
            "moves": list(moves_tuple),
            "category": category,
            "source_seed": seed_name,
            "mirrored": is_mirrored,
            "plies": len(moves_tuple),
        })
        return True

    print(f"Step 1: Ingesting exact weak seeds ({len(EMPIRICAL_WEAK_SEEDS)} sources)...", flush=True)
    for item in EMPIRICAL_WEAK_SEEDS:
        cat = item["category"]
        name = item["name"]
        raw_moves = item["moves"]
        # Admit exact
        try_admit(raw_moves, cat, name, False)
        # Admit mirror
        try_admit(_mirror_history(raw_moves), cat, name, True)

    print(f"  Admitted {len(bank)} exact and mirrored seeds.", flush=True)

    print(f"Step 2: Generating tactical branches to reach {target_size} roots...", flush=True)
    seeds_list = list(EMPIRICAL_WEAK_SEEDS)

    attempts = 0
    max_attempts = target_size * 20
    last_reported = 0

    while len(bank) < target_size and attempts < max_attempts:
        attempts += 1
        item = rng.choice(seeds_list)
        cat = item["category"]
        name = item["name"]
        base_moves = item["moves"]
        branch_plies = item["branch_plies"]

        # Pick branch ply
        b_ply = rng.choice(branch_plies)
        prefix = base_moves[:b_ply]

        # Target length (8, 9, 10 plies)
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
        if try_admit(cand, cat, name, is_mirror):
            if len(bank) // 500 > last_reported:
                last_reported = len(bank) // 500
                print(f"  Progress: {len(bank):,}/{target_size:,} roots (cached positions: {len(cache):,})", flush=True)

    print(f"Generation finished: {len(bank)} unique valid roots created.", flush=True)
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

    # Generate companion manifest
    categories_count = {}
    for row in bank:
        cat = row["category"]
        categories_count[cat] = categories_count.get(cat, 0) + 1

    manifest = {
        "schema": "zquoridor.targeted_weakness_opening_bank.v1",
        "file": str(out_path.name),
        "total_rows": len(bank),
        "seed": config["seed"],
        "target_bank_size": config["target_bank_size"],
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "categories_distribution": categories_count,
    }
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
