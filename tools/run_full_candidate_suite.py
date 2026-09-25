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
    "candidate_exe": str(ROOT / "results" / "experiments" / "multipath_unified_champion" / "zquoridor.exe"),
    "candidate_nnue": str(ROOT / "results" / "experiments" / "multipath_unified_champion" / "student_int8.bin"),
    "suite_name": "champion_external_battery",
    "workers": 4,
    "claustro_normal_pairs": 50,
    "claustro_centerrush_pairs": 50,
    "titanium_normal_pairs": 50,
    "titanium_centerrush_pairs": 50,
    "skip_claustro": False,
    "skip_titanium": False,
    "dry_run": False,
}


def run_cmd(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def summarize_battery(base_out: Path):
    """Compile comparative summary against historical baseline targets."""
    print("\n" + "=" * 80)
    print("BATTERY COMPARATIVE REPORT (Unified Champion vs Historical Baselines)")
    print("=" * 80)

    baselines = {
        "claustro_100g_normal": {"name": "Claustrophobia (Normal Book)", "base_score": 50.83, "base_elo": 5.8},
        "claustro_100g_centerrush": {"name": "Claustrophobia (Center Rush)", "base_score": 37.50, "base_elo": -88.7},
        "titanium_100g_normal": {"name": "Titanium (Normal Book)", "base_score": 60.00, "base_elo": 70.4},
        "titanium_100g_centerrush": {"name": "Titanium (Center Rush)", "base_score": 46.00, "base_elo": -27.9},
    }

    report_data = {}
    for key, info in baselines.items():
        summary_path = base_out / key / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            opp_key = "claustrophobia" if "claustro" in key else "titanium"
            s = data["summaries"][opp_key]
            score = s["score_pct"]
            elo = s["elo"]
            games = s.get("included_games", s.get("recorded_games", 0))
            pairs = s.get("complete_pairs", 0)
            boot = s.get("paired_bootstrap_95", {})
            b_low = boot.get("score_low_pct")
            b_high = boot.get("score_high_pct")
            delta_score = score - info["base_score"]
            delta_elo = elo - info["base_elo"]

            report_data[key] = {
                "name": info["name"],
                "games": games,
                "pairs": pairs,
                "score_pct": score,
                "elo": elo,
                "bootstrap_95": [b_low, b_high],
                "base_score": info["base_score"],
                "delta_score": delta_score,
                "delta_elo": delta_elo,
            }
        except Exception as exc:
            print(f"Error reading {summary_path}: {exc}")

    header = f"{'Sub-suite':<32} | {'Games':<6} | {'Candidate':<11} | {'Baseline':<10} | {'Delta Score':<12} | {'Delta Elo':<10}"
    print(header)
    print("-" * len(header))
    for key, d in report_data.items():
        cand_str = f"{d['score_pct']:.1f}% ({d['elo']:+.1f})"
        base_str = f"{d['base_score']:.1f}%"
        delta_str = f"{d['delta_score']:+.1f}%"
        elo_str = f"{d['delta_elo']:+.1f}"
        print(f"{d['name']:<32} | {d['games']:<6} | {cand_str:<11} | {base_str:<10} | {delta_str:<12} | {elo_str:<10}")

    out_file = base_out / "battery_summary.json"
    out_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nSaved battery summary to {out_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run complete candidate external battery vs Claustrophobia and Titanium.")
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

    claustro_norm_out = base_out / "claustro_100g_normal"
    claustro_cr_out = base_out / "claustro_100g_centerrush"
    titanium_norm_out = base_out / "titanium_100g_normal"
    titanium_cr_out = base_out / "titanium_100g_centerrush"

    # 1. Claustrophobia (Normal Openings)
    if not args.skip_claustro and args.claustro_normal_pairs > 0:
        print("\n" + "=" * 70, flush=True)
        print(f"PHASE 1: {args.claustro_normal_pairs * 2} GAMES VS CLAUSTROPHOBIA (NORMAL OPENINGS, GPU)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "claustrophobia",
            "--openings", str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
            "--pairs", str(args.claustro_normal_pairs),
            "--workers", str(args.workers),
            "--claustrophobia-device", "gpu",
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(claustro_norm_out)
        ])

    # 2. Claustrophobia (Center Rush Openings)
    if not args.skip_claustro and args.claustro_centerrush_pairs > 0:
        print("\n" + "=" * 70, flush=True)
        print(f"PHASE 2: {args.claustro_centerrush_pairs * 2} GAMES VS CLAUSTROPHOBIA (CENTER RUSH OPENINGS, GPU)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "claustrophobia",
            "--openings", str(ROOT / "tools" / "external" / "openings_center_rush_50pairs.jsonl"),
            "--pairs", str(args.claustro_centerrush_pairs),
            "--workers", str(args.workers),
            "--claustrophobia-device", "gpu",
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(claustro_cr_out)
        ])

    # 3. Titanium (Normal Openings)
    if not args.skip_titanium and args.titanium_normal_pairs > 0:
        print("\n" + "=" * 70, flush=True)
        print(f"PHASE 3: {args.titanium_normal_pairs * 2} GAMES VS TITANIUM (NORMAL OPENINGS)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "titanium",
            "--openings", str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
            "--pairs", str(args.titanium_normal_pairs),
            "--workers", str(args.workers),
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(titanium_norm_out)
        ])

    # 4. Titanium (Center Rush Openings)
    if not args.skip_titanium and args.titanium_centerrush_pairs > 0:
        print("\n" + "=" * 70, flush=True)
        print(f"PHASE 4: {args.titanium_centerrush_pairs * 2} GAMES VS TITANIUM (CENTER RUSH OPENINGS)", flush=True)
        print("=" * 70, flush=True)
        run_cmd([
            sys.executable, "-u", str(ROOT / "tools" / "run_benchmark.py"),
            "--opponents", "titanium",
            "--openings", str(ROOT / "tools" / "external" / "openings_center_rush_50pairs.jsonl"),
            "--pairs", str(args.titanium_centerrush_pairs),
            "--workers", str(args.workers),
            "--zq-executable", str(exe),
            "--nnue", str(nnue),
            "--output", str(titanium_cr_out)
        ])

    # 5. Statistical breakdown and comparative evaluation
    print("\n" + "=" * 70)
    print("PHASE 5: STATISTICAL BREAKDOWN & COMPARISON")
    print("=" * 70)

    for folder in [claustro_norm_out, claustro_cr_out, titanium_norm_out, titanium_cr_out]:
        games_file = folder / "games.jsonl"
        if games_file.exists():
            run_cmd([
                sys.executable, str(ROOT / "tools" / "analyze_center_openings.py"),
                str(games_file)
            ])

    summarize_battery(base_out)


if __name__ == "__main__":
    raise SystemExit(main())
