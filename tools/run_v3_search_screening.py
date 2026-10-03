#!/usr/bin/env python3
"""Screen aggressive V3 search configurations against V3 main and Claustrophobia.

The NNUE is held fixed at the current production V3 weights. Candidate-vs-main
therefore measures search changes only. Each non-main candidate also plays the
same paired opening sample against Claustrophobia. The "main" entry is included
only to calibrate V3-main vs Claustrophobia on the exact same sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import match_finalists, run_benchmark

SCREEN_ID = "v3-search-aggressive-20261003"

# Keep individual mechanisms separable before testing the full combinations.
# All candidates use the exact same V3 NNUE weights as production.
CANDIDATES: dict[str, list[str]] = {
    "main": [],
    "lmr210": [
        "--lmr-divisor", "2.10",
    ],
    "lmr200": [
        "--lmr-divisor", "2.00",
    ],
    "policy_lmr": [
        "--policy-lmr",
    ],
    "policy_lmr_aggr": [
        "--policy-lmr",
        "--policy-lmr-hot", "2.0",
        "--policy-lmr-cold", "4.0",
        "--lmr-divisor", "2.00",
    ],
    "lmp15": [
        "--policy-lmp",
        "--policy-lmp-base", "0.15",
        "--policy-lmp-min-count", "12",
    ],
    "lmr_lmp": [
        "--policy-lmr",
        "--policy-lmr-hot", "2.0",
        "--policy-lmr-cold", "4.0",
        "--lmr-divisor", "2.00",
        "--policy-lmp",
        "--policy-lmp-base", "0.15",
        "--policy-lmp-min-count", "12",
    ],
    "cpuct072": [
        "--cpuct", "0.72",
    ],
    "full_aggr": [
        "--cpuct", "0.72",
        "--policy-lmr",
        "--policy-lmr-hot", "2.0",
        "--policy-lmr-cold", "4.0",
        "--lmr-divisor", "2.00",
        "--policy-lmp",
        "--policy-lmp-base", "0.15",
        "--policy-lmp-min-count", "12",
    ],
    "pw16": [
        "--progressive-widening",
        "--widening-initial", "16",
        "--widening-coeff", "2.0",
        "--widening-exp", "0.5",
    ],
    "pw12_full": [
        "--cpuct", "0.75",
        "--progressive-widening",
        "--widening-initial", "12",
        "--widening-coeff", "2.0",
        "--widening-exp", "0.5",
        "--policy-lmr",
        "--policy-lmr-hot", "2.0",
        "--policy-lmr-cold", "4.0",
        "--lmr-divisor", "2.00",
        "--policy-lmp",
        "--policy-lmp-base", "0.15",
        "--policy-lmp-min-count", "12",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--move-time-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument(
        "--openings",
        default=str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "benchmarks" / SCREEN_ID),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--skip-v3", action="store_true")
    parser.add_argument("--skip-claustrophobia", action="store_true")
    return parser


def run_one(args: argparse.Namespace) -> dict:
    if args.pairs <= 0 or args.workers <= 0 or args.move_time_ms <= 0:
        raise ValueError("pairs, workers and move-time-ms must be positive")

    name = args.candidate
    candidate_args = list(CANDIDATES[name])
    weights = (ROOT / "data" / "nnue" / "nnue_weights_int8.bin").resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"V3 production weights not found: {weights}")

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = run_benchmark._build_zq(
        ROOT / "bin" / "v3_search_screen" / f"zquoridor_uci{suffix}"
    )
    openings = Path(args.openings).resolve()
    output = Path(args.output).resolve() / name
    output.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "schema": "zquoridor.v3_search_screen.v1",
        "screen_id": SCREEN_ID,
        "candidate": name,
        "candidate_args": candidate_args,
        "weights": str(weights),
        "pairs": int(args.pairs),
        "move_time_ms": int(args.move_time_ms),
        "seed": int(args.seed),
        "openings": str(openings),
        "vs_v3_main": None,
        "vs_claustrophobia": None,
    }

    # Same executable + same V3 weights on both sides. Only engine1_args differ.
    if name != "main" and not args.skip_v3:
        h2h = dict(match_finalists.CONFIG)
        h2h.update(
            engine1_name=f"v3_{name}",
            engine1_executable=str(exe),
            engine1_nnue=str(weights),
            engine1_args=candidate_args,
            engine2_name="v3_main",
            engine2_executable=str(exe),
            engine2_nnue=str(weights),
            engine2_args=[],
            pairs=int(args.pairs),
            move_time_ms=int(args.move_time_ms),
            workers=int(args.workers),
            seed=int(args.seed),
            openings=str(openings),
            output=str(output / "vs_v3_main"),
            bootstrap=int(args.bootstrap),
        )
        report["vs_v3_main"] = match_finalists.run(h2h)["summary"]

    if not args.skip_claustrophobia:
        bench = dict(run_benchmark.CONFIG)
        bench.update(
            opponents=["claustrophobia"],
            pairs=int(args.pairs),
            workers=int(args.workers),
            seed=int(args.seed),
            openings=str(openings),
            output=str(output / "vs_claustrophobia"),
            resume=True,
            retry_failed=True,
            auto_setup=True,
            zq_executable=str(exe),
            nnue=str(weights),
            zq_args=candidate_args,
            zq_move_time_ms=int(args.move_time_ms),
            claustrophobia_move_time_ms=int(args.move_time_ms),
            claustrophobia_device="cpu",
            bootstrap=int(args.bootstrap),
        )
        external = run_benchmark.run(bench)
        report["vs_claustrophobia"] = external["summaries"]["claustrophobia"]

    summary_path = output / "screen_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== V3 SEARCH SCREEN SUMMARY ===")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run_one(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"screening error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
