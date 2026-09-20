#!/usr/bin/env python3
"""Queue runner: wait for the active confirmation benchmark, then run H2H vs race512-search10-ft and 400 games vs Claustrophobia."""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def wait_for_file(path: Path, poll_interval_s: int = 15) -> None:
    log(f"Waiting for prerequisite benchmark to finish: {path}")
    last_heartbeat = 0.0
    while True:
        if path.is_file() and path.stat().st_size > 50:
            try:
                # Verify valid JSON
                data = json.loads(path.read_text(encoding="utf-8"))
                if "main" in data or "schema" in data:
                    log(f"Prerequisite benchmark finished successfully! Detected: {path}")
                    return
            except Exception:
                pass
        now = time.time()
        if now - last_heartbeat >= 60.0:
            last_heartbeat = now
            # Check how many games have been written to partial files
            parent = path.parent
            games_counts = []
            for gpath in parent.rglob("games.jsonl"):
                try:
                    count = len(gpath.read_text(encoding="utf-8").strip().splitlines())
                    games_counts.append(f"{gpath.parent.name}: {count} games")
                except Exception:
                    pass
            status_str = ", ".join(games_counts) if games_counts else "initializing"
            log(f"Queue status: waiting for {path.name}... Current progress: [{status_str}]")
        time.sleep(poll_interval_s)


def run_h2h(args: argparse.Namespace) -> None:
    log("=================================================================")
    log("Starting Job 1: H2H race512-multitier-champion vs race512-search10-ft")
    log("=================================================================")

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "match_finalists.py"),
        "--engine1-name", "race512-multitier-champion",
        "--engine1-executable", str(ROOT / "results" / "experiments" / "race512-multitier-champion" / "zquoridor.exe"),
        "--engine1-nnue", str(ROOT / "results" / "experiments" / "race512-multitier-champion" / "student_int8.bin"),
        "--engine2-name", "race512-search10-ft",
        "--engine2-executable", str(ROOT / "results" / "experiments" / "race512-search10-ft-s20260917" / "zquoridor.exe"),
        "--engine2-nnue", str(ROOT / "results" / "experiments" / "race512-search10-ft-s20260917" / "student_int8.bin"),
        "--pairs", str(args.h2h_pairs),
        "--move-time-ms", str(args.move_time_ms),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--openings", str(Path(args.openings).resolve()),
        "--output", str(ROOT / "benchmark_results" / "h2h-multitier-vs-race512-search10-ft-200ms"),
    ]
    log(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log("Job 1 (H2H vs race512-search10-ft) completed successfully!")


def run_claustrophobia_400(args: argparse.Namespace) -> None:
    log("=================================================================")
    log("Starting Job 2: 400 games (200 pairs) vs Claustrophobia on GPU")
    log("=================================================================")

    claustro_openings = ROOT / "tools" / "external" / "openings_claustro_followup_200pairs.jsonl"
    if not claustro_openings.exists():
        claustro_openings = Path(args.openings).resolve()

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_benchmark.py"),
        "--opponents", "claustrophobia",
        "--pairs", str(args.claustro_pairs),
        "--workers", str(args.workers),
        "--seed", str(args.seed + 1),
        "--openings", str(claustro_openings),
        "--zq-executable", str(ROOT / "results" / "experiments" / "race512-multitier-champion" / "zquoridor.exe"),
        "--nnue", str(ROOT / "results" / "experiments" / "race512-multitier-champion" / "student_int8.bin"),
        "--zq-move-time-ms", str(args.move_time_ms),
        "--claustrophobia-move-time-ms", str(args.move_time_ms),
        "--claustrophobia-device", "gpu",
        "--output", str(ROOT / "benchmark_results" / "race512-multitier-champion-claustrophobia-400g"),
    ]
    log(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log("Job 2 (400 games vs Claustrophobia) completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue post-confirmation benchmark runs")
    parser.add_argument("--wait-for-file", default=str(ROOT / "benchmark_results" / "race512-multitier-champion-confirm-200ms" / "summary.json"))
    parser.add_argument("--h2h-pairs", type=int, default=100)
    parser.add_argument("--claustro-pairs", type=int, default=200)
    parser.add_argument("--move-time-ms", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--openings", default=str(ROOT / "tools" / "external" / "openings_confirmation_v1.jsonl"))
    parser.add_argument("--skip-wait", action="store_true", help="Do not wait for prerequisite file")

    args = parser.parse_args()

    if not args.skip_wait and args.wait_for_file:
        wait_for_file(Path(args.wait_for_file).resolve())

    # Step 1: Head-to-Head vs race512-search10-ft
    run_h2h(args)

    # Step 2: 400 games vs Claustrophobia
    run_claustrophobia_400(args)

    log("ALL QUEUED BENCHMARKS HAVE FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
