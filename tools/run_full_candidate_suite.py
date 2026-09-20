#!/usr/bin/env python3
"""Run complete evaluation battery for a candidate:
1. 600 games vs Claustrophobia (GPU) on openings_600g_300pairs.jsonl
2. 100 games vs Titanium on openings_screen_v1.jsonl
3. 100 games vs Titanium on openings_center_rush_50pairs.jsonl
4. Comprehensive statistical breakdown of central vs wall openings and White vs Black performance.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run complete 600g Claustro + 200g Titanium battery.")
    parser.add_argument("--candidate-exe", required=True, help="Path to candidate executable")
    parser.add_argument("--candidate-nnue", required=True, help="Path to candidate quantized weights (.bin)")
    parser.add_argument("--suite-name", required=True, help="Folder name for benchmark outputs")
    parser.add_argument("--skip-claustro", action="store_true", help="Skip Claustrophobia 600g")
    parser.add_argument("--skip-titanium", action="store_true", help="Skip Titanium 200g")
    args = parser.parse_args()

    exe = Path(args.candidate_exe).resolve()
    nnue = Path(args.candidate_nnue).resolve()
    base_out = ROOT / "benchmark_results" / args.suite_name

    claustro_out = base_out / "claustro_600g"
    titanium_norm_out = base_out / "titanium_100g_normal"
    titanium_cr_out = base_out / "titanium_100g_centerrush"

    # 1. 600 games vs Claustrophobia (GPU)
    if not args.skip_claustro:
        print("\n" + "=" * 70)
        print("PHASE 1: 600 GAMES VS CLAUSTROPHOBIA (GPU)")
        print("=" * 70)
        run_cmd([
            sys.executable, str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "claustrophobia",
            "--openings", str(ROOT / "tools" / "external" / "openings_600g_300pairs.jsonl"),
            "--pairs", "300",
            "--claustrophobia-device", "gpu",
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(claustro_out)
        ])

    # 2. 100 games vs Titanium (Normal)
    if not args.skip_titanium:
        print("\n" + "=" * 70)
        print("PHASE 2: 100 GAMES VS TITANIUM (NORMAL OPENINGS)")
        print("=" * 70)
        run_cmd([
            sys.executable, str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "titanium",
            "--openings", str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
            "--pairs", "50",
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(titanium_norm_out)
        ])

        # 3. 100 games vs Titanium (Center Rush)
        print("\n" + "=" * 70)
        print("PHASE 3: 100 GAMES VS TITANIUM (CENTER RUSH OPENINGS)")
        print("=" * 70)
        run_cmd([
            sys.executable, str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "titanium",
            "--openings", str(ROOT / "tools" / "external" / "openings_center_rush_50pairs.jsonl"),
            "--pairs", "50",
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
    main()
