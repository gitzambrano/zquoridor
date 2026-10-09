#!/usr/bin/env python3
"""Generate diverse Quoridor positions and dense search targets via random wall placements."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external.local_arena import Referee
from training.teachers.targets import (
    POLICY_DIM,
    dumps,
    make_label,
    make_position,
    normalize_policy,
    zq_move_to_policy_index,
)

# Edit these defaults for a normal run. CLI options override one invocation.
CONFIG = {
    "count": 50,
    "min_walls": 2,
    "max_walls": 14,
    "pawn_walk_plies": 4,
    "move_time_ms": 20,
    "zq_executable": str(ROOT / "bin" / "zquoridor.exe"),
    "output": str(ROOT / "data/teaching/wall_drops"),
    "seed": 20261009,
    "export_npz": True,
    "split": "train",
}


def compute_shortest_paths(referee: Referee, side_to_move: int) -> tuple[int, int]:
    """Compute shortest path distances for own and opponent pawns using breadth-first search."""
    def bfs_distance(player: int) -> int:
        start = referee.pawns[player]
        goal_row = 8 if player == 0 else 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, distance = queue.popleft()
            if current[0] == goal_row:
                return distance
            for neighbor in referee._neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return 99

    own_dist = bfs_distance(side_to_move)
    opp_dist = bfs_distance(1 - side_to_move)
    return own_dist, opp_dist


def generate_wall_drop_history(
    min_walls: int,
    max_walls: int,
    pawn_walk_plies: int,
    rng: random.Random,
) -> list[str]:
    """Generate a valid game history containing random legal pawn moves and wall placements."""
    referee = Referee()
    history: list[str] = []

    # 1. Optional initial pawn movements
    walk_target = rng.randint(0, pawn_walk_plies)
    for _ in range(walk_target):
        pawn_moves = [m for m in referee.legal_moves() if len(m) == 2]
        if not pawn_moves:
            break
        chosen_move = rng.choice(pawn_moves)
        referee.apply(chosen_move)
        history.append(chosen_move)
        if referee.winner is not None:
            return []

    # 2. Sequential random legal wall placements
    wall_target = rng.randint(min_walls, max_walls)
    placed_walls = 0
    attempts = 0
    max_attempts = wall_target * 40

    while placed_walls < wall_target and attempts < max_attempts:
        attempts += 1
        side = referee.side_to_move
        if referee.walls_left[side] <= 0:
            pawn_moves = [m for m in referee.legal_moves() if len(m) == 2]
            if not pawn_moves:
                break
            chosen_move = rng.choice(pawn_moves)
            referee.apply(chosen_move)
            history.append(chosen_move)
            if referee.winner is not None:
                return []
            continue

        row = rng.randint(0, 7)
        col = rng.randint(0, 7)
        orientation = rng.choice(["h", "v"])
        col_char = chr(ord("a") + col)
        wall_text = f"{col_char}{row + 1}{orientation}"

        if referee.is_legal_move(wall_text):
            referee.apply(wall_text)
            history.append(wall_text)
            placed_walls += 1

    # Verify that the generated state is active and non-terminal
    if referee.winner is not None:
        return []
    if placed_walls < min_walls:
        return []

    return history


def query_engine_root(
    executable: str,
    history: Sequence[str],
    move_time_ms: int,
) -> dict | None:
    """Send position and search command to Zquoridor UCI process and parse the root JSON."""
    moves_clause = " moves " + " ".join(history) if history else ""
    commands = (
        "isready\n"
        f"position startpos{moves_clause}\n"
        f"go movetime {move_time_ms}\n"
        "quit\n"
    )

    try:
        proc = subprocess.run(
            [executable, "--dump-root"],
            input=commands,
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("info string root_json "):
            json_text = line[len("info string root_json "):].strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return None

    return None


def run_generation(config: dict) -> dict:
    """Execute the wall drop generation loop and write output artifacts."""
    rng = random.Random(config["seed"])
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    exe_path = config["zq_executable"]
    if not Path(exe_path).is_file():
        raise FileNotFoundError(f"Zquoridor executable not found: {exe_path}")

    positions: list[dict] = []
    labels: list[dict] = []
    policy_matrix: list[list[float]] = []
    value_list: list[float] = []
    aux_features: list[list[int]] = []
    sample_ids: list[str] = []

    generated = 0
    attempts = 0
    max_total_attempts = config["count"] * 10

    print(f"Generating {config['count']} wall-drop positions with seed {config['seed']}...", flush=True)

    while generated < config["count"] and attempts < max_total_attempts:
        attempts += 1
        history = generate_wall_drop_history(
            min_walls=config["min_walls"],
            max_walls=config["max_walls"],
            pawn_walk_plies=config["pawn_walk_plies"],
            rng=rng,
        )
        if not history:
            continue

        root_data = query_engine_root(
            executable=exe_path,
            history=history,
            move_time_ms=config["move_time_ms"],
        )
        if not root_data or root_data.get("target_status") != "ok":
            continue

        raw_visits = root_data.get("visit_counts", [])
        if len(raw_visits) != POLICY_DIM:
            continue

        total_visits = sum(raw_visits)
        if total_visits > 0:
            policy = [float(v) / float(total_visits) for v in raw_visits]
        else:
            priors = root_data.get("priors", [])
            if len(priors) != POLICY_DIM or sum(priors) <= 0:
                continue
            policy = normalize_policy(priors)

        root_value = float(root_data.get("root_value", 0.0))

        # Reconstruct referee to calculate path distances and wall balance
        ref = Referee()
        for mv in history:
            ref.apply(mv)

        own_d, opp_d = compute_shortest_paths(ref, ref.side_to_move)
        aux_row = [
            own_d,
            opp_d,
            ref.walls_left[ref.side_to_move],
            ref.walls_left[1 - ref.side_to_move],
            len(history),
            ref.side_to_move,
        ]

        pos_record = make_position(
            history=history,
            split=config["split"],
            source="wall_drops",
            opening_index=-1,
            ply=len(history),
            metadata={"generator": "generate_wall_drops.py", "placed_walls": len(history)},
        )

        label_record = make_label(
            position=pos_record,
            teacher="zquoridor-4.0",
            mode="search-root",
            policy=policy,
            value=root_value,
            bestmove=root_data.get("bestmove"),
            budget={"move_time_ms": config["move_time_ms"], "simulations": root_data.get("simulations", 0)},
            metadata={"aux_shortest_paths": [own_d, opp_d]},
        )

        positions.append(pos_record)
        labels.append(label_record)
        policy_matrix.append(policy)
        value_list.append(root_value)
        aux_features.append(aux_row)
        sample_ids.append(pos_record["id"])

        generated += 1
        if generated % 10 == 0 or generated == config["count"]:
            print(f"  Generated {generated}/{config['count']} positions (attempts: {attempts})", flush=True)

    # Write JSONL position corpus
    positions_file = output_dir / "positions.jsonl"
    with positions_file.open("w", encoding="utf-8") as f_pos:
        for p in positions:
            f_pos.write(dumps(p) + "\n")

    # Write JSONL teacher targets
    targets_file = output_dir / "targets.jsonl"
    with targets_file.open("w", encoding="utf-8") as f_tgt:
        for lbl in labels:
            f_tgt.write(dumps(lbl) + "\n")

    # Export compact NumPy archive if requested
    if config["export_npz"] and positions:
        npz_file = output_dir / "targets.npz"
        np.savez_compressed(
            npz_file,
            policy=np.asarray(policy_matrix, dtype=np.float32),
            value=np.asarray(value_list, dtype=np.float32),
            aux=np.asarray(aux_features, dtype=np.int32),
            sample_ids=np.asarray(sample_ids),
        )
        print(f"Exported dense NumPy targets to: {npz_file}", flush=True)

    summary = {
        "generated": len(positions),
        "requested": config["count"],
        "positions_file": str(positions_file),
        "targets_file": str(targets_file),
        "seed": config["seed"],
    }

    manifest_file = output_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f_man:
        json.dump(summary, f_man, indent=2)

    print(f"Wall-drop collection complete: {len(positions)} records written to {output_dir}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build command line interface for wall drop generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, help="Number of positions to generate")
    parser.add_argument("--min-walls", type=int, help="Minimum number of placed walls")
    parser.add_argument("--max-walls", type=int, help="Maximum number of placed walls")
    parser.add_argument("--pawn-walk-plies", type=int, help="Maximum plies of initial pawn walks")
    parser.add_argument("--move-time-ms", type=int, help="Engine search time in milliseconds")
    parser.add_argument("--zq-executable", type=str, help="Path to Zquoridor UCI executable")
    parser.add_argument("--output", type=str, help="Output directory for generated files")
    parser.add_argument("--seed", type=int, help="Random number generator seed")
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), help="Dataset split name")
    parser.add_argument("--no-export-npz", dest="export_npz", action="store_false", help="Disable NPZ export")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point with dictionary configuration and command line overrides."""
    parser = build_parser()
    args = parser.parse_args(argv)

    active_config = dict(CONFIG)
    for key, val in vars(args).items():
        if val is not None:
            active_config[key] = val

    run_generation(active_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
