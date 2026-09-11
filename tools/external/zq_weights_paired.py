#!/usr/bin/env python3
"""Pair-aware fixed-movetime arena between two ZQuoridor weight files.

Both sides use the same UCI adapter binary and therefore the same search code;
only command arguments (normally ``--nnue`` weights) differ. Each opening is
played twice with colors reversed. Output JSONL is compatible with
``analyze_paired_arena.py``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
from typing import Sequence

import titanium_arena_fixed  # noqa: F401; patches robust UCI I/O
import titanium_arena as arena


def read_openings(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        moves = obj.get("moves")
        if not isinstance(moves, list) or not all(isinstance(x, str) for x in moves):
            raise ValueError(f"{path}:{lineno}: expected string-list 'moves'")
        rows.append(moves)
    if not rows:
        raise ValueError(f"no openings in {path}")
    return rows


def play_one(opening_index: int, opening: Sequence[str], candidate_player: int,
             candidate_cmd: Sequence[str], baseline_cmd: Sequence[str],
             movetime_ms: int, max_plies: int) -> dict:
    candidate = arena.UCIEngine(candidate_cmd, "candidate")
    baseline = arena.UCIEngine(baseline_cmd, "baseline")
    history = list(opening)
    winner = -1
    termination = "max_plies"
    candidate_think = baseline_think = 0.0
    try:
        while len(history) < max_plies:
            player = len(history) & 1
            engine = candidate if player == candidate_player else baseline
            move, think_s, _info = engine.bestmove(history, movetime_ms)
            if move == "(none)":
                termination = "no_move"
                break
            if not arena.syntax_ok(move):
                raise RuntimeError(f"{engine.name}: invalid move syntax {move!r}")
            history.append(move)
            if engine is candidate:
                candidate_think += think_s
            else:
                baseline_think += think_s
            if arena.reached_goal(player, move):
                winner = player
                termination = "goal"
                break
    finally:
        candidate.close()
        baseline.close()

    result = 0.5 if winner < 0 else (1.0 if winner == candidate_player else 0.0)
    return {
        "opening_index": opening_index,
        "zq_player": candidate_player,
        "result": result,
        "winner": winner,
        "plies": len(history),
        "candidate_think_s": candidate_think,
        "baseline_think_s": baseline_think,
        "termination": termination,
        "moves": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--candidate-arg", action="append", default=[])
    parser.add_argument("--baseline-arg", action="append", default=[])
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument("--limit-openings", type=int, default=20)
    parser.add_argument("--movetime", type=int, default=200)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--games-out", required=True, type=Path)
    args = parser.parse_args()
    if args.limit_openings <= 0 or args.movetime <= 0 or args.max_plies <= 0 or args.workers <= 0:
        raise SystemExit("limits, movetime, max-plies, and workers must be positive")

    openings = read_openings(args.openings)[: args.limit_openings]
    if len(openings) < args.limit_openings:
        raise SystemExit(f"requested {args.limit_openings} openings but only {len(openings)} available")
    candidate_cmd = [args.engine, *args.candidate_arg]
    baseline_cmd = [args.engine, *args.baseline_arg]
    jobs = [
        (idx, opening, player)
        for idx, opening in enumerate(openings)
        for player in (0, 1)
    ]

    results: list[dict] = []
    if args.workers == 1:
        for job in jobs:
            results.append(play_one(*job, candidate_cmd, baseline_cmd, args.movetime, args.max_plies))
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(play_one, *job, candidate_cmd, baseline_cmd, args.movetime, args.max_plies)
                for job in jobs
            ]
            for future in cf.as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda row: (row["opening_index"], row["zq_player"]))
    args.games_out.parent.mkdir(parents=True, exist_ok=True)
    with args.games_out.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    wins = sum(row["result"] == 1.0 for row in results)
    losses = sum(row["result"] == 0.0 for row in results)
    draws = len(results) - wins - losses
    score = sum(float(row["result"]) for row in results) / len(results)
    summary = {
        "schema": "zquoridor.weights_paired.v1",
        "paired_openings": len(openings),
        "games": len(results),
        "movetime_ms": args.movetime,
        "workers": args.workers,
        "wld": [wins, losses, draws],
        "score_pct": 100.0 * score,
        "candidate_cmd": candidate_cmd,
        "baseline_cmd": baseline_cmd,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
