#!/usr/bin/env python3
"""Confirmation gates for the strongest Zquoridor V3 search candidates.

All Zquoridor-vs-Zquoridor gates keep the production V3 NNUE and executable
identical on both sides; only runtime search arguments differ. External gates
use the same candidate configuration against pinned Claustrophobia/Titanium.
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
from tools.run_v3_search_screening import CANDIDATES

FINALISTS = ("cpuct072", "lmr210", "cpuct072_lmr210")
ALL_EXTERNAL = ("main", *FINALISTS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    p.add_argument("--gate", required=True, choices=("main-fixed", "main-clock", "external"))
    p.add_argument("--pairs", required=True, type=int)
    p.add_argument("--openings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--move-time-ms", type=int, default=200)
    p.add_argument("--base-ms", type=int, default=180000)
    p.add_argument("--increment-ms", type=int, default=2000)
    p.add_argument("--move-overhead-ms", type=int, default=20)
    p.add_argument("--opponents", default="claustrophobia,titanium")
    p.add_argument("--bootstrap", type=int, default=20000)
    return p


def run(args: argparse.Namespace) -> dict:
    if args.pairs <= 0 or args.workers <= 0:
        raise ValueError("pairs and workers must be positive")
    if args.gate != "external" and args.candidate not in FINALISTS:
        raise ValueError("main H2H confirmation is only for the three finalists")

    weights = (ROOT / "data" / "nnue" / "nnue_weights_int8.bin").resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"production V3 NNUE not found: {weights}")
    openings = Path(args.openings).resolve()
    if not openings.is_file():
        raise FileNotFoundError(f"opening book not found: {openings}")

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = run_benchmark._build_zq(
        ROOT / "bin" / "v3_search_confirmation" / f"zquoridor_uci{suffix}"
    )
    candidate_args = list(CANDIDATES[args.candidate])
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "zquoridor.v3_search_confirmation.v1",
        "candidate": args.candidate,
        "candidate_args": candidate_args,
        "gate": args.gate,
        "pairs": int(args.pairs),
        "seed": int(args.seed),
        "openings": str(openings),
    }

    if args.gate in ("main-fixed", "main-clock"):
        cfg = dict(match_finalists.CONFIG)
        cfg.update(
            engine1_name=f"v3_{args.candidate}",
            engine1_executable=str(exe),
            engine1_nnue=str(weights),
            engine1_args=candidate_args,
            engine2_name="v3_main",
            engine2_executable=str(exe),
            engine2_nnue=str(weights),
            engine2_args=[],
            pairs=int(args.pairs),
            workers=int(args.workers),
            seed=int(args.seed),
            openings=str(openings),
            output=str(out),
            bootstrap=int(args.bootstrap),
        )
        if args.gate == "main-clock":
            cfg.update(
                base_ms=int(args.base_ms),
                increment_ms=int(args.increment_ms),
                move_overhead_ms=int(args.move_overhead_ms),
            )
        else:
            cfg.update(
                move_time_ms=int(args.move_time_ms),
                base_ms=0,
                increment_ms=0,
            )
        h2h = match_finalists.run(cfg)
        report["summary"] = h2h["summary"]
        report["clock"] = {
            "base_ms": int(cfg.get("base_ms", 0)),
            "increment_ms": int(cfg.get("increment_ms", 0)),
            "move_time_ms": None if int(cfg.get("base_ms", 0)) > 0 else int(cfg["move_time_ms"]),
        }
    else:
        opponents = [x.strip().lower() for x in args.opponents.split(",") if x.strip()]
        if not opponents or any(x not in ("claustrophobia", "titanium") for x in opponents):
            raise ValueError("external opponents must be claustrophobia and/or titanium")
        cfg = dict(run_benchmark.CONFIG)
        cfg.update(
            opponents=opponents,
            pairs=int(args.pairs),
            workers=int(args.workers),
            seed=int(args.seed),
            openings=str(openings),
            output=str(out),
            resume=True,
            retry_failed=True,
            auto_setup=True,
            zq_executable=str(exe),
            nnue=str(weights),
            zq_args=candidate_args,
            zq_move_time_ms=int(args.move_time_ms),
            titanium_move_time_ms=int(args.move_time_ms),
            claustrophobia_move_time_ms=int(args.move_time_ms),
            claustrophobia_device="cpu",
            bootstrap=int(args.bootstrap),
        )
        external = run_benchmark.run(cfg)
        report["summaries"] = external["summaries"]

    (out / "confirmation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== V3 SEARCH CONFIRMATION ===")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"confirmation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
