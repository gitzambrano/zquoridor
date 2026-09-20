#!/usr/bin/env python3
"""Run a small paired game-clock smoke test between two UCI engines."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external import local_arena  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-exe", required=True)
    parser.add_argument("--candidate-nnue", required=True)
    parser.add_argument("--baseline-exe", required=True)
    parser.add_argument("--baseline-nnue", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--base-ms", type=int, default=180_000)
    parser.add_argument("--increment-ms", type=int, default=2_000)
    parser.add_argument("--move-overhead-ms", type=int, default=20)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--move-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-plies", type=int, default=240)
    return parser


def read_openings(path: Path, pairs: int, seed: int) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        moves = value.get("moves")
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise ValueError(f"{path}:{index + 1}: expected a string list in 'moves'")
        referee = local_arena.Referee()
        for move in moves:
            referee.apply(move)
        if referee.winner is not None:
            raise ValueError(f"{path}:{index + 1}: the opening is terminal")
        rows.append((index, moves))
    if len(rows) < pairs:
        raise ValueError(f"the opening file has {len(rows)} rows but the run needs {pairs}")
    random.Random(seed).shuffle(rows)
    return rows[:pairs]


def run(args: argparse.Namespace) -> dict:
    if min(args.pairs, args.base_ms, args.increment_ms + 1, args.max_plies) <= 0:
        raise ValueError("pairs, base-ms, max-plies, and increment-ms must be valid")
    paths = {
        "candidate_exe": Path(args.candidate_exe).resolve(),
        "candidate_nnue": Path(args.candidate_nnue).resolve(),
        "baseline_exe": Path(args.baseline_exe).resolve(),
        "baseline_nnue": Path(args.baseline_nnue).resolve(),
        "openings": Path(args.openings).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("the output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)

    config = {
        "schema": "zquoridor.clock_smoke.config.v1",
        "pairs": args.pairs,
        "seed": args.seed,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "move_overhead_ms": args.move_overhead_ms,
        "move_timeout_s": args.move_timeout_s,
        "max_plies": args.max_plies,
    }
    manifest = local_arena.make_manifest(config, paths)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    openings = read_openings(paths["openings"], args.pairs, args.seed)
    candidate_command = [str(paths["candidate_exe"]), "--nnue", str(paths["candidate_nnue"]),
                         "--move-overhead", str(args.move_overhead_ms)]
    baseline_command = [str(paths["baseline_exe"]), "--nnue", str(paths["baseline_nnue"]),
                        "--move-overhead", str(args.move_overhead_ms)]

    rows = []
    games_path = output / "games.jsonl"
    with games_path.open("w", encoding="utf-8", buffering=1) as stream:
        for opening_index, opening in openings:
            for candidate_player in (0, 1):
                row = local_arena.play_game(
                    opponent="baseline",
                    opening_index=opening_index,
                    opening=opening,
                    zq_player=candidate_player,
                    zq_factory=lambda: local_arena.UciPlayer(
                        candidate_command, "candidate",
                        startup_timeout_s=args.startup_timeout_s,
                    ),
                    opponent_factory=lambda: local_arena.UciPlayer(
                        baseline_command, "baseline",
                        startup_timeout_s=args.startup_timeout_s,
                    ),
                    zq_budget=0,
                    opponent_budget=0,
                    move_timeout_s=args.move_timeout_s,
                    max_plies=args.max_plies,
                    run_id=manifest["run_id"],
                    clock_initial_ms=args.base_ms,
                    clock_increment_ms=args.increment_ms,
                )
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                print(f"clock smoke {len(rows)}/{args.pairs * 2}: {row['status']}", flush=True)

    failed = [row for row in rows if row["status"] != "ok"]
    elapsed = [move["elapsed_ms"] for row in rows for move in row["move_times"]]
    report = {
        "schema": "zquoridor.clock_smoke.report.v1",
        "run_id": manifest["run_id"],
        "games": len(rows),
        "failed_games": len(failed),
        "healthy": not failed,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "max_move_elapsed_ms": max(elapsed, default=0.0),
        "failures": [{"opening_index": row["opening_index"],
                      "candidate_player": row["zq_player"],
                      "error": row.get("error", "unknown error")} for row in failed],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(build_parser().parse_args(argv))
        return 0 if report["healthy"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"clock smoke error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
