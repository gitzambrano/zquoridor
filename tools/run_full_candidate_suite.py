#!/usr/bin/env python3
"""Run complete evaluation battery for a candidate:
1. 600 games vs Claustrophobia (GPU) on openings_600g_300pairs.jsonl
2. 100 games vs Titanium on openings_screen_v1.jsonl
3. 100 games vs Titanium on openings_center_rush_50pairs.jsonl
4. Comprehensive statistical breakdown of central vs wall openings and White vs Black performance.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Edit this block for the normal full gate. CLI options override these values.
CONFIG = {
    "candidate_exe": None,
    "candidate_nnue": None,
    "suite_name": "candidate-suite",
    "workers": 6,
    "skip_claustro": False,
    "skip_titanium": False,
    "dry_run": True,
}


def run_cmd(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run complete 600g Claustro + 200g Titanium battery.")
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif value is None:
            parser.add_argument(flag, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=None)
    parsed = parser.parse_args(argv)
    values = dict(CONFIG)
    for key, value in vars(parsed).items():
        if value is not None:
            values[key] = value
    if values["dry_run"]:
        print(json.dumps(values, indent=2, default=str), flush=True)
        return 0
    if not values["candidate_exe"] or not values["candidate_nnue"]:
        raise ValueError("candidate_exe and candidate_nnue are required in CONFIG or CLI")
    args = argparse.Namespace(**values)

    exe = Path(args.candidate_exe).resolve()
    nnue = Path(args.candidate_nnue).resolve()
    base_out = ROOT / "results" / "benchmarks" / args.suite_name

    claustro_out = base_out / "claustro_600g"
    titanium_norm_out = base_out / "titanium_100g_normal"
    titanium_cr_out = base_out / "titanium_100g_centerrush"

    # 1. 600 games vs Claustrophobia (GPU)
    if not args.skip_claustro:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 1: 600 GAMES VS CLAUSTROPHOBIA (GPU)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "claustrophobia",
            "--openings", str(ROOT / "tools" / "external" / "openings_600g_300pairs.jsonl"),
            "--pairs", "300",
            "--workers", str(args.workers),
            "--claustrophobia-device", "gpu",
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(claustro_out)
        ])

    # 2. 100 games vs Titanium (Normal)
    if not args.skip_titanium:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 2: 100 GAMES VS TITANIUM (NORMAL OPENINGS)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "titanium",
            "--openings", str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
            "--pairs", "50",
            "--workers", str(args.workers),
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(titanium_norm_out)
        ])

    # 3. 100 games vs Titanium (Center Rush)
    if not args.skip_titanium:
        print("\n" + "=" * 70, flush=True)
        print("PHASE 3: 100 GAMES VS TITANIUM (CENTER RUSH OPENINGS)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "titanium",
            "--openings", str(ROOT / "tools" / "external" / "openings_center_rush_50pairs.jsonl"),
            "--pairs", "50",
            "--workers", str(args.workers),
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(titanium_cr_out)
        ])

    # 4. Statistical breakdown
    print("\n" + "=" * 70)
    print("PHASE 4: STATISTICAL BREAKDOWN & ANALYSIS")
    print("=" * 70)

    for folder in [claustro_out, titanium_norm_out, titanium_cr_out]:
        games_file = folder / "games.jsonl"
        if games_file.exists():
            run_cmd([
                sys.executable, str(ROOT / "tools" / "analyze_center_openings.py"),
                str(games_file)
            ])


if __name__ == "__main__":
    raise SystemExit(main())
