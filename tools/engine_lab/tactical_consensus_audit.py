#!/usr/bin/env python3
"""Audit wall-choice stability with multiple search budgets and configurations.

This is a diagnostic, NOT an independent game-theoretic oracle or training label.
Never use frozen evaluation books as training examples.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.external.local_arena import Referee

CONFIG = {
    "engine": "bin/zq_tactical_consensus",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "book_paths": {
        "normal": "tools/external/openings_normal_confirm_400.jsonl",
        "center": "tools/external/openings_center_rush_sound_5k.jsonl",
        "targeted": "tools/external/openings_targeted_weakness_bank.jsonl",
    },
    "book": "targeted",
    "samples": 14,
    "seed": 20261106,
    "budget_fast_ms": 200,
    "budget_mid_ms": 1200,
    "budget_deep_ms": 3000,
    "output": "results/tactical-consensus",
}


def load_book(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()]


def detect_benchmark_overlap(moves: list[str], benchmarks: list[tuple[str, ...]]) -> bool:
    seq = tuple(moves)
    # Reject exact examples and either-direction full-move-history prefix overlap.
    return any(seq[: min(len(seq), len(example))] ==
               example[: min(len(seq), len(example))] for example in benchmarks)


def ask_engine(executable: str, weights: str, moves: list[str],
               milliseconds: int, extra: tuple[str, ...] = ()) -> dict:
    script = "uci\nposition startpos" + (
        (" moves " + " ".join(moves)) if moves else ""
    ) + f"\ngo movetime {milliseconds}\nquit\n"
    result = subprocess.run(
        [executable, "--nnue", weights, "--dump-root", *extra],
        input=script, text=True, capture_output=True, timeout=60, check=True,
    )
    if "info string error" in result.stdout:
        raise ValueError("Engine refused an opening: " + result.stdout[-1000:])
    roots = [json.loads(line.split("root_json ", 1)[1])
             for line in result.stdout.splitlines() if "root_json " in line]
    if len(roots) != 1 or roots[0].get("target_status") != "ok":
        raise ValueError("Missing root search statistics: " + result.stdout[-1300:])
    root = roots[0]
    return {
        "move": root["bestmove"],
        "best_action": root["best_action"],
        "simulations": root["simulations"],
        "root_visits": root["root_visits"],
        "root_value": root.get("root_value"),
        "prior_of_best": root["priors"][root["best_action"]],
        "root_visits_of_best": root["visit_counts"][root["best_action"]],
    }


def shortest_path_lengths(referee: Referee) -> tuple[int, int]:
    distances = []
    for player in (0, 1):
        src = referee.pawns[player]
        goal = 8 if player == 0 else 0
        seen = {src}
        queue = deque([(src, 0)])
        found = None
        while queue:
            cell, steps = queue.popleft()
            if cell[0] == goal:
                found = steps
                break
            for neighbor in referee._neighbors(cell):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, steps + 1))
        if found is None:
            raise ValueError("Position is not connected")
        distances.append(found)
    return tuple(distances)


def wall_properties(history: list[str], selected: str) -> dict:
    if len(selected) != 3:
        return {"is_wall": False, "path_distance_neutral": None}
    referee = Referee()
    for move in history:
        referee.apply(move)
    before = shortest_path_lengths(referee)
    if not referee.is_legal_move(selected):
        raise ValueError("Engine recommended an illegal wall: " + selected)
    referee.apply(selected)
    after = shortest_path_lengths(referee)
    return {
        "is_wall": True,
        "path_distance_neutral": before == after,
        "path_distance_before": before,
        "path_distance_after": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", choices=tuple(CONFIG["book_paths"]), default=CONFIG["book"])
    parser.add_argument("--samples", type=int, default=CONFIG["samples"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--engine", default=CONFIG["engine"])
    parser.add_argument("--weights", default=CONFIG["weights"])
    parser.add_argument("--budget-fast-ms", type=int, default=CONFIG["budget_fast_ms"])
    parser.add_argument("--budget-mid-ms", type=int, default=CONFIG["budget_mid_ms"])
    parser.add_argument("--budget-deep-ms", type=int, default=CONFIG["budget_deep_ms"])
    parser.add_argument("--output", default=CONFIG["output"])
    args = parser.parse_args()
    cfg = {**CONFIG, **vars(args)}
    if cfg["samples"] < 1 or min(cfg["budget_fast_ms"], cfg["budget_mid_ms"]) < 1:
        parser.error("sample count and search budgets must be positive")
    if cfg["budget_deep_ms"] <= cfg["budget_mid_ms"]:
        parser.error("deep budget must exceed middle budget")
    book = load_book(ROOT / CONFIG["book_paths"][cfg["book"]])
    benchmarks = [
        tuple(row["moves"])
        for category in ("normal", "center")
        for row in load_book(ROOT / CONFIG["book_paths"][category])
    ]
    chosen = list(range(len(book)))
    random.Random(cfg["seed"] + {"normal": 11, "center": 13, "targeted": 17}[cfg["book"]]).shuffle(chosen)
    if len(chosen) < cfg["samples"]:
        parser.error("sample count exceeds source book")

    output = ROOT / cfg["output"] / cfg["book"]
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index in chosen[:cfg["samples"]]:
        history = book[index]["moves"]
        fast = ask_engine(cfg["engine"], cfg["weights"], history, cfg["budget_fast_ms"])
        mid = ask_engine(cfg["engine"], cfg["weights"], history, cfg["budget_mid_ms"])
        deep = ask_engine(cfg["engine"], cfg["weights"], history, cfg["budget_deep_ms"])
        # The alternate graph mix is NOT an independent trained evaluator.
        # It provides a weak cross-configuration robustness check only.
        alt = ask_engine(cfg["engine"], cfg["weights"], history, cfg["budget_deep_ms"],
                         ("--graph-leaf-mix", "0.55"))
        move = deep["move"]
        wall = wall_properties(history, move)
        overlap = detect_benchmark_overlap(history, benchmarks)
        record = {
            "book": cfg["book"],
            "source_index": index,
            "history": history,
            "benchmark_overlap": overlap,
            "diagnostic_only": True,
            "fast": fast,
            "mid": mid,
            "deep": deep,
            "deep_alt_graphmix": alt,
            "fast_vs_mid_change": fast["move"] != mid["move"],
            "fast_vs_deep_change": fast["move"] != deep["move"],
            "mid_vs_deep_change": mid["move"] != deep["move"],
            "deep_config_agreement": deep["move"] == alt["move"],
            "stable_deep_move": mid["move"] == deep["move"] == alt["move"],
            "deep_choice": wall,
        }
        rows.append(record)
        print("POSITION " + json.dumps({
            "index": index,
            "moves": [fast["move"], mid["move"], deep["move"], alt["move"]],
            "stable_deep": record["stable_deep_move"],
            "neutral_wall": wall.get("path_distance_neutral"),
            "benchmark_overlap": overlap,
        }), flush=True)
        (output / "positions.json").write_text(json.dumps(rows, indent=2) + "\n")

    count = len(rows)
    summary = {
        "book": cfg["book"],
        "positions": count,
        "fast_to_deep_changes": sum(x["fast_vs_deep_change"] for x in rows),
        "mid_to_deep_changes": sum(x["mid_vs_deep_change"] for x in rows),
        "deep_configuration_agreements": sum(x["deep_config_agreement"] for x in rows),
        "stable_deep_choices": sum(x["stable_deep_move"] for x in rows),
        "deep_wall_choices": sum(x["deep_choice"]["is_wall"] for x in rows),
        "deep_distance_neutral_walls": sum(x["deep_choice"].get("path_distance_neutral") is True for x in rows),
        "overlap_with_frozen_evaluation": sum(x["benchmark_overlap"] for x in rows),
        "potential_nonbenchmark_cases": sum(not x["benchmark_overlap"] for x in rows),
        "caveat": "Deep search and altered graph mix share a network and are not independent truth labels.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
