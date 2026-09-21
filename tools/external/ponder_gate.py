#!/usr/bin/env python3
"""Paired strength gate for opponent-root pondering versus the same baseline."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external import local_arena  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True)
    p.add_argument("--nnue", required=True)
    p.add_argument("--openings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--mode", choices=("fixed", "clock"), default="fixed")
    p.add_argument("--movetime-ms", type=int, default=200)
    p.add_argument("--base-ms", type=int, default=180_000)
    p.add_argument("--increment-ms", type=int, default=2_000)
    p.add_argument("--move-overhead-ms", type=int, default=20)
    p.add_argument("--startup-timeout-s", type=float, default=60.0)
    p.add_argument("--move-timeout-s", type=float, default=45.0)
    p.add_argument("--max-plies", type=int, default=240)
    return p


def load_openings(path: Path, pairs: int, seed: int) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        moves = obj.get("moves")
        if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
            raise ValueError(f"{path}:{index + 1}: expected string list 'moves'")
        ref = local_arena.Referee()
        for move in moves:
            ref.apply(move)
        if ref.winner is not None:
            raise ValueError(f"{path}:{index + 1}: terminal opening")
        rows.append((index, list(moves)))
    if len(rows) < pairs:
        raise ValueError(f"requested {pairs} pairs but only {len(rows)} openings exist")
    random.Random(seed).shuffle(rows)
    return rows[:pairs]


def run(args: argparse.Namespace) -> dict:
    engine = Path(args.engine).resolve()
    nnue = Path(args.nnue).resolve()
    openings_path = Path(args.openings).resolve()
    for path in (engine, nnue, openings_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pairs <= 0 or args.workers <= 0 or args.max_plies <= 0:
        raise ValueError("pairs, workers and max-plies must be positive")
    if args.mode == "fixed" and args.movetime_ms <= 0:
        raise ValueError("movetime-ms must be positive")
    if args.mode == "clock" and (args.base_ms <= 0 or args.increment_ms < 0):
        raise ValueError("invalid game clock")

    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    selected = load_openings(openings_path, args.pairs, args.seed)
    command = [
        str(engine), "--nnue", str(nnue),
        "--move-overhead", str(args.move_overhead_ms),
    ]
    config = {
        "schema": "zquoridor.ponder_gate.config.v1",
        "mode": args.mode,
        "pairs": args.pairs,
        "workers": args.workers,
        "seed": args.seed,
        "movetime_ms": args.movetime_ms,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "move_overhead_ms": args.move_overhead_ms,
        "max_plies": args.max_plies,
        "candidate": "same engine + opponent-root pondering",
        "baseline": "same engine without pondering",
    }
    manifest = local_arena.make_manifest(
        config, {"engine": engine, "nnue": nnue, "openings": openings_path}
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    def one(opening_index: int, opening: list[str], candidate_side: int) -> dict:
        common = dict(
            opponent="baseline",
            opening_index=opening_index,
            opening=opening,
            zq_player=candidate_side,
            zq_factory=lambda: local_arena.UciPlayer(
                command, "ponder-candidate", startup_timeout_s=args.startup_timeout_s
            ),
            opponent_factory=lambda: local_arena.UciPlayer(
                command, "baseline", startup_timeout_s=args.startup_timeout_s
            ),
            zq_budget=args.movetime_ms if args.mode == "fixed" else 0,
            opponent_budget=args.movetime_ms if args.mode == "fixed" else 0,
            move_timeout_s=args.move_timeout_s,
            max_plies=args.max_plies,
            run_id=manifest["run_id"],
            zq_ponder=True,
        )
        if args.mode == "clock":
            common["clock_initial_ms"] = args.base_ms
            common["clock_increment_ms"] = args.increment_ms
        return local_arena.play_game(**common)

    jobs = [
        (opening_index, opening, side)
        for opening_index, opening in selected
        for side in (0, 1)
    ]
    rows: list[dict] = []
    lock = threading.Lock()
    games_path = out / "games.jsonl"
    with games_path.open("w", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(one, opening_index, opening, side):
                    (opening_index, side)
                for opening_index, opening, side in jobs
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                row = future.result()
                with lock:
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    completed += 1
                    print(
                        f"{args.mode} ponder gate {completed}/{len(jobs)} "
                        f"opening={row['opening_index']} side={row['zq_player']} "
                        f"status={row['status']} result={row.get('result')}",
                        flush=True,
                    )

    paired = local_arena.summarize_pairs(rows, bootstrap=10_000, seed=args.seed)
    candidate_moves = [
        move
        for row in rows if row.get("status") == "ok"
        for move in row.get("move_times", [])
        if move.get("player") == "zquoridor"
    ]
    pondered_opponent_moves = [
        move
        for row in rows if row.get("status") == "ok"
        for move in row.get("move_times", [])
        if "zq_ponder_budget_ms" in move
    ]
    tree_hit_moves = sum(
        "treeHit=1" in str(move.get("search_last", ""))
        for move in candidate_moves
    )
    ponder_reuse_calls = sum(
        "treeHit=1" in str(move.get("zq_ponder_last", ""))
        for move in pondered_opponent_moves
    )
    ponder_budget_ms = sum(
        int(move.get("zq_ponder_budget_ms", 0))
        for move in pondered_opponent_moves
    )
    ponder_elapsed_ms = sum(
        float(move.get("zq_ponder_elapsed_ms", 0.0))
        for move in pondered_opponent_moves
    )
    summary = {
        "schema": "zquoridor.ponder_gate.summary.v1",
        "run_id": manifest["run_id"],
        "mode": args.mode,
        **paired,
        "candidate_searches": len(candidate_moves),
        "candidate_tree_hits": tree_hit_moves,
        "candidate_tree_hit_pct": (
            100.0 * tree_hit_moves / len(candidate_moves) if candidate_moves else 0.0
        ),
        "ponder_calls": len(pondered_opponent_moves),
        "ponder_calls_reusing_prior_tree": ponder_reuse_calls,
        "ponder_budget_ms": ponder_budget_ms,
        "ponder_elapsed_ms": ponder_elapsed_ms,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parser().parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ponder gate error: {exc}", file=sys.stderr)
        return 2
    return 0 if summary.get("failed_games", 1) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
