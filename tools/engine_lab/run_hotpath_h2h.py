#!/usr/bin/env python3
"""Compare MCGS build variants in paired, legal games with fixed CPU time."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external import local_arena

CONFIG = {
    "candidate": "bin/zq_candidate",
    "baseline": "bin/zq_baseline",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "normal_book": "tools/external/openings_normal_screen_100.jsonl",
    "center_book": "tools/external/openings_center_rush_sound_5k.jsonl",
    "pairs_per_book": 4,
    "move_time_ms": 200,
    "clock_initial_ms": 0,
    "clock_increment_ms": 0,
    "max_plies": 140,
    "timeout_s": 8.0,
    "seed": 20261104,
    "output": "results/hotpath-strength/h2h",
}


def run(cfg: dict) -> dict:
    weights = str((ROOT / cfg["weights"]).resolve())
    candidate = str((ROOT / cfg["candidate"]).resolve())
    baseline = str((ROOT / cfg["baseline"]).resolve())

    def candidate_factory():
        return local_arena.UciPlayer([candidate, "--nnue", weights], "candidate")

    def baseline_factory():
        return local_arena.UciPlayer([baseline, "--nnue", weights], "baseline")

    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    games = []
    run_id = "hotpath-strength-20261104-" + str(cfg["seed"])
    for book_idx, key in enumerate(("normal_book", "center_book")):
        book = ROOT / cfg[key]
        positions = [json.loads(row)["moves"] for row in book.read_text().splitlines() if row.strip()]
        order = list(range(len(positions)))
        random.Random(cfg["seed"] + book_idx * 1049).shuffle(order)
        for i in range(cfg["pairs_per_book"]):
            source_idx = order[i]
            opening = positions[source_idx]
            for side in (0, 1):
                result = local_arena.play_game(
                    opponent="frozen-main",
                    opening_index=book_idx * 100000 + source_idx,
                    opening=opening,
                    zq_player=side,
                    zq_factory=candidate_factory,
                    opponent_factory=baseline_factory,
                    zq_budget=cfg["move_time_ms"],
                    opponent_budget=cfg["move_time_ms"],
                    move_timeout_s=cfg["timeout_s"],
                    max_plies=cfg["max_plies"],
                    run_id=run_id,
                    clock_initial_ms=cfg["clock_initial_ms"],
                    clock_increment_ms=cfg["clock_increment_ms"],
                    zq_ponder=False,
                )
                result["book"] = "normal" if book_idx == 0 else "center_rush"
                result["source_index"] = source_idx
                games.append(result)
                print(
                    f"GAME book={result['book']} index={source_idx} side={side} "
                    f"status={result['status']} result={result.get('result')} "
                    f"plies={result.get('plies')} termination={result['termination']}",
                    flush=True,
                )
                with (output / "games.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, separators=(",", ":")) + "\n")

    report = local_arena.summarize_pairs(games, bootstrap=2000, seed=cfg["seed"])
    report["configuration"] = cfg
    report["book_games"] = {book: sum(row["book"] == book for row in games)
                            for book in ("normal", "center_rush")}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps({
        k: report.get(k) for k in ("complete_pairs", "failed_games", "score_pct",
                                  "elo", "paired_bootstrap_95", "strength_claim_ready")
    }), flush=True)
    if report["failed_games"] or report["complete_pairs"] != 2 * cfg["pairs_per_book"]:
        raise RuntimeError("Paired match had failed or missing games")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=CONFIG["candidate"])
    parser.add_argument("--baseline", default=CONFIG["baseline"])
    parser.add_argument("--pairs-per-book", type=int, default=CONFIG["pairs_per_book"])
    parser.add_argument("--move-time-ms", type=int, default=CONFIG["move_time_ms"])
    parser.add_argument("--clock-initial-ms", type=int, default=CONFIG["clock_initial_ms"])
    parser.add_argument("--clock-increment-ms", type=int, default=CONFIG["clock_increment_ms"])
    parser.add_argument("--output", default=CONFIG["output"])
    args = parser.parse_args()
    cfg = {**CONFIG, **vars(args)}
    if cfg["pairs_per_book"] <= 0 or cfg["move_time_ms"] <= 0:
        parser.error("pairs-per-book and move-time-ms must be positive")
    if cfg["clock_initial_ms"] < 0 or cfg["clock_increment_ms"] < 0:
        parser.error("clocks must be nonnegative")
    run(cfg)


if __name__ == "__main__":
    main()
