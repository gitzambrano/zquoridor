#!/usr/bin/env python3
"""Run Gumbel AlphaZero planning and Sequential Halving to generate high-density policy targets."""
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
    "count": 30,
    "top_k": 8,
    "c_scale": 2.0,
    "move_time_ms": 15,
    "zq_executable": str(ROOT / "bin" / "zquoridor.exe"),
    "source_positions": None,
    "output": str(ROOT / "data/teaching/gumbel_sims"),
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


def sample_gumbel(rng: random.Random) -> float:
    """Sample one standard Gumbel random variable using the inverse transform method."""
    uniform = rng.uniform(1e-10, 1.0 - 1e-10)
    return -math.log(-math.log(uniform))


def evaluate_action_value(
    executable: str,
    history: Sequence[str],
    action_move: str,
    move_time_ms: int,
) -> float:
    """Evaluate candidate action by querying the child root value from the opponent perspective."""
    child_history = list(history) + [action_move]
    child_root = query_engine_root(
        executable=executable,
        history=child_history,
        move_time_ms=move_time_ms,
    )
    if child_root is None or child_root.get("target_status") != "ok":
        return 0.0
    child_v = float(child_root.get("root_value", 0.0))
    # From parent perspective, Q = -V(child)
    return -child_v


def run_sequential_halving(
    executable: str,
    history: Sequence[str],
    candidates: Sequence[tuple[str, int, float]],
    move_time_ms: int,
) -> dict[int, float]:
    """Execute Sequential Halving across candidate actions to focus simulation time on top moves."""
    active: list[tuple[str, int, float]] = list(candidates)
    num_candidates = len(active)
    if num_candidates <= 1:
        rounds = 1
    else:
        rounds = max(1, math.ceil(math.log2(num_candidates)))

    action_q: dict[int, float] = {}

    for round_index in range(rounds):
        round_time = max(5, int(move_time_ms * (round_index + 1) / rounds))
        scores: list[tuple[float, tuple[str, int, float]]] = []

        for move_str, policy_idx, _ in active:
            q_val = evaluate_action_value(
                executable=executable,
                history=history,
                action_move=move_str,
                move_time_ms=round_time,
            )
            action_q[policy_idx] = q_val
            scores.append((q_val, (move_str, policy_idx, q_val)))

        # Sort descending by Q-value
        scores.sort(key=lambda item: item[0], reverse=True)

        if len(scores) <= 1:
            break

        # Halve the active candidates for the subsequent round
        keep_count = max(1, math.ceil(len(scores) / 2))
        active = [item[1] for item in scores[:keep_count]]

    return action_q


def generate_candidate_positions(count: int, rng: random.Random) -> list[list[str]]:
    """Generate diverse move sequences for Gumbel simulation benchmarking."""
    positions: list[list[str]] = []
    attempts = 0
    max_attempts = count * 20

    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        ref = Referee()
        hist: list[str] = []
        plies = rng.randint(4, 20)

        for _ in range(plies):
            legal = ref.legal_moves()
            if not legal:
                break
            # Bias toward a balanced mix of pawn steps and wall drops
            pawn_moves = [m for m in legal if len(m) == 2]
            wall_moves = [m for m in legal if len(m) == 3]
            if pawn_moves and (not wall_moves or rng.random() < 0.6):
                choice = rng.choice(pawn_moves)
            elif wall_moves:
                choice = rng.choice(wall_moves)
            else:
                choice = rng.choice(legal)

            ref.apply(choice)
            hist.append(choice)
            if ref.winner is not None:
                break

        if ref.winner is None and len(hist) >= 4:
            positions.append(hist)

    return positions


def run_gumbel_generation(config: dict) -> dict:
    """Execute Gumbel AlphaZero simulation pipeline and output refined targets."""
    rng = random.Random(config["seed"])
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    exe_path = config["zq_executable"]
    if not Path(exe_path).is_file():
        raise FileNotFoundError(f"Zquoridor executable not found: {exe_path}")

    # Load source positions or generate diverse game states
    if config["source_positions"]:
        src_path = Path(config["source_positions"])
        input_histories: list[list[str]] = []
        with src_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    item = json.loads(line)
                    input_histories.append(item.get("history", []))
    else:
        input_histories = generate_candidate_positions(config["count"], rng)

    positions: list[dict] = []
    labels: list[dict] = []
    policy_matrix: list[list[float]] = []
    value_list: list[float] = []
    aux_features: list[list[int]] = []
    sample_ids: list[str] = []

    print(f"Executing Gumbel Sequential Halving across {len(input_histories)} positions...", flush=True)

    for idx, history in enumerate(input_histories):
        # 1. Query root prior distribution and initial search evaluation
        root_data = query_engine_root(
            executable=exe_path,
            history=history,
            move_time_ms=config["move_time_ms"],
        )
        if not root_data or root_data.get("target_status") != "ok":
            continue

        raw_priors = root_data.get("priors", [])
        if len(raw_priors) != POLICY_DIM:
            continue

        # 2. Enumerate legal actions from referee
        ref = Referee()
        try:
            for mv in history:
                ref.apply(mv)
        except Exception:
            continue

        if ref.winner is not None:
            continue

        legal_moves = ref.legal_moves()
        if not legal_moves:
            continue

        legal_map: dict[str, int] = {}
        for m in legal_moves:
            legal_map[m] = zq_move_to_policy_index(m, ref.side_to_move)

        # 3. Add Gumbel noise to legal move prior logits
        candidates: list[tuple[str, int, float]] = []
        for move_str, policy_idx in legal_map.items():
            prior_prob = max(1e-7, float(raw_priors[policy_idx]))
            prior_logit = math.log(prior_prob)
            gumbel_val = sample_gumbel(rng)
            perturbed_score = prior_logit + gumbel_val
            candidates.append((move_str, policy_idx, perturbed_score))

        # Select top-K perturbed candidates
        candidates.sort(key=lambda item: item[2], reverse=True)
        top_candidates = candidates[:min(config["top_k"], len(candidates))]

        # 4. Sequential Halving across top candidates
        action_q = run_sequential_halving(
            executable=exe_path,
            history=history,
            candidates=top_candidates,
            move_time_ms=config["move_time_ms"],
        )

        # 5. Compute Gumbel improved policy target
        min_eval_q = min(action_q.values()) if action_q else -0.5
        c_scale = float(config["c_scale"])

        target_logits = [-1e9] * POLICY_DIM
        for move_str, policy_idx in legal_map.items():
            prior_prob = max(1e-7, float(raw_priors[policy_idx]))
            prior_logit = math.log(prior_prob)
            q_val = action_q.get(policy_idx, min_eval_q - 0.2)
            target_logits[policy_idx] = prior_logit + c_scale * q_val

        # Softmax over legal actions
        max_logit = max(target_logits[p_idx] for p_idx in legal_map.values())
        exp_sum = 0.0
        policy_target = [0.0] * POLICY_DIM
        for p_idx in legal_map.values():
            exp_val = math.exp(target_logits[p_idx] - max_logit)
            policy_target[p_idx] = exp_val
            exp_sum += exp_val

        if exp_sum > 0.0:
            policy_target = [val / exp_sum for val in policy_target]
        else:
            continue

        # Expected value from improved policy
        expected_value = 0.0
        for p_idx in legal_map.values():
            q_val = action_q.get(p_idx, min_eval_q - 0.2)
            expected_value += policy_target[p_idx] * q_val

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
            source="gumbel_sims",
            opening_index=-1,
            ply=len(history),
            metadata={"generator": "generate_gumbel_sims.py", "top_k": config["top_k"]},
        )

        best_action_idx = max(action_q.keys(), key=lambda k: action_q[k]) if action_q else None
        best_move_str = None
        if best_action_idx is not None:
            for mv, p_idx in legal_map.items():
                if p_idx == best_action_idx:
                    best_move_str = mv
                    break

        label_record = make_label(
            position=pos_record,
            teacher="zquoridor-4.0-gumbel",
            mode="gumbel-sequential-halving",
            policy=policy_target,
            value=expected_value,
            bestmove=best_move_str,
            budget={"top_k": config["top_k"], "move_time_ms": config["move_time_ms"]},
            metadata={"aux_shortest_paths": [own_d, opp_d]},
        )

        positions.append(pos_record)
        labels.append(label_record)
        policy_matrix.append(policy_target)
        value_list.append(expected_value)
        aux_features.append(aux_row)
        sample_ids.append(pos_record["id"])

        if (len(positions) % 10 == 0) or len(positions) == config["count"]:
            print(f"  Processed {len(positions)}/{config['count']} Gumbel targets", flush=True)

        if len(positions) >= config["count"]:
            break

    # Write output artifacts
    positions_file = output_dir / "positions.jsonl"
    with positions_file.open("w", encoding="utf-8") as f_pos:
        for p in positions:
            f_pos.write(dumps(p) + "\n")

    targets_file = output_dir / "targets.jsonl"
    with targets_file.open("w", encoding="utf-8") as f_tgt:
        for lbl in labels:
            f_tgt.write(dumps(lbl) + "\n")

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

    print(f"Gumbel planning complete: {len(positions)} targets written to {output_dir}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build command line interface for Gumbel simulation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, help="Number of positions to evaluate")
    parser.add_argument("--top-k", type=int, help="Number of candidate moves for Sequential Halving")
    parser.add_argument("--c-scale", type=float, help="Scale multiplier for action value in policy completion")
    parser.add_argument("--move-time-ms", type=int, help="Engine search time in milliseconds per candidate")
    parser.add_argument("--zq-executable", type=str, help="Path to Zquoridor UCI executable")
    parser.add_argument("--source-positions", type=str, help="Path to existing JSONL position corpus")
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

    run_gumbel_generation(active_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
