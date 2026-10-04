#!/usr/bin/env python3
"""Heavy confirmation harness for transformative search experiments.

Runs either:
  * candidate executable vs frozen V3 main (fixed 200 ms or 3+2), or
  * one selected executable vs Claustrophobia/Titanium.

The workflow compiles experimental macros separately and supplies executables
explicitly, so the production NNUE and frozen main remain unchanged.
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", required=True, choices=("main-fixed", "main-clock", "external"))
    p.add_argument("--name", required=True)
    p.add_argument("--engine-exe", required=True)
    p.add_argument("--main-exe")
    p.add_argument("--nnue", required=True)
    p.add_argument("--main-nnue")
    p.add_argument("--pairs", required=True, type=int)
    p.add_argument("--openings", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--move-time-ms", type=int, default=200)
    p.add_argument("--base-ms", type=int, default=180000)
    p.add_argument("--increment-ms", type=int, default=2000)
    p.add_argument("--move-overhead-ms", type=int, default=20)
    p.add_argument("--opponents", default="claustrophobia,titanium")
    p.add_argument("--bootstrap", type=int, default=20000)
    return p


def run(a: argparse.Namespace) -> dict:
    if a.pairs <= 0 or a.workers <= 0:
        raise ValueError("pairs and workers must be positive")

    engine_exe = Path(a.engine_exe).resolve()
    nnue = Path(a.nnue).resolve()
    openings = Path(a.openings).resolve()
    for label, path in (("engine-exe", engine_exe), ("nnue", nnue), ("openings", openings)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "zquoridor.transform_search_confirmation.v1",
        "name": a.name,
        "gate": a.gate,
        "pairs": a.pairs,
        "seed": a.seed,
        "openings": str(openings),
    }

    if a.gate in ("main-fixed", "main-clock"):
        if not a.main_exe or not a.main_nnue:
            raise ValueError("main-fixed/main-clock require --main-exe and --main-nnue")
        main_exe = Path(a.main_exe).resolve()
        main_nnue = Path(a.main_nnue).resolve()
        if not main_exe.is_file() or not main_nnue.is_file():
            raise FileNotFoundError("frozen main executable/NNUE missing")

        cfg = dict(match_finalists.CONFIG)
        cfg.update(
            engine1_name=a.name,
            engine1_executable=str(engine_exe),
            engine1_nnue=str(nnue),
            engine1_args=[],
            engine2_name="v3_main_c57e96e",
            engine2_executable=str(main_exe),
            engine2_nnue=str(main_nnue),
            engine2_args=[],
            pairs=a.pairs,
            workers=a.workers,
            seed=a.seed,
            openings=str(openings),
            output=str(out),
            bootstrap=a.bootstrap,
        )
        if a.gate == "main-clock":
            cfg.update(
                base_ms=a.base_ms,
                increment_ms=a.increment_ms,
                move_overhead_ms=a.move_overhead_ms,
            )
        else:
            cfg.update(move_time_ms=a.move_time_ms, base_ms=0, increment_ms=0)
        result = match_finalists.run(cfg)
        report["summary"] = result["summary"]
        report["clock"] = {
            "move_time_ms": None if a.gate == "main-clock" else a.move_time_ms,
            "base_ms": a.base_ms if a.gate == "main-clock" else 0,
            "increment_ms": a.increment_ms if a.gate == "main-clock" else 0,
        }
    else:
        opponents = [x.strip().lower() for x in a.opponents.split(",") if x.strip()]
        if not opponents or any(x not in ("claustrophobia", "titanium") for x in opponents):
            raise ValueError("unsupported external opponent")
        cfg = dict(run_benchmark.CONFIG)
        cfg.update(
            opponents=opponents,
            pairs=a.pairs,
            workers=a.workers,
            seed=a.seed,
            openings=str(openings),
            output=str(out),
            resume=True,
            retry_failed=True,
            auto_setup=True,
            zq_executable=str(engine_exe),
            nnue=str(nnue),
            zq_args=[],
            zq_move_time_ms=a.move_time_ms,
            titanium_move_time_ms=a.move_time_ms,
            claustrophobia_move_time_ms=a.move_time_ms,
            claustrophobia_device="cpu",
            bootstrap=a.bootstrap,
        )
        result = run_benchmark.run(cfg)
        report["summaries"] = result["summaries"]

    (out / "confirmation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"transform confirmation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
