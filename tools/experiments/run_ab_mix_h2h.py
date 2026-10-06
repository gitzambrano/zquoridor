#!/usr/bin/env python3
"""Controlled H2H matrix for selective alpha-beta inside production MCGS.

The experiment keeps code, NNUE weights, time control, opening subsets, colors,
and referee fixed. Only the candidate's alpha-beta knobs change.

Default workload per arm:
  - 133 Center Rush pairs = 266 games
  -  67 Normal pairs      = 134 games
  - 400 games total, color-swapped by opening

Arms:
  remove_final : disable the production endgame AB leaf rule.
  deeper_final : keep the production gate (mover has 0 walls) but use depth 4.
  root_verify  : production endgame rule + shallow AB root prefilter.
  hybrid       : deeper endgame AB + shallow AB root prefilter.

Global MCAB leaf depth remains 0 in every arm because global leaf AB was
previously measured as strongly regressive at 200 ms/move.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import match_finalists
from tools.external import local_arena

CONFIG = {
    "move_time_ms": 200,
    "workers": 14,
    "bootstrap": 20000,
    "central_pairs": 133,
    "normal_pairs": 67,
    "central_seed": 2026100601,
    "normal_seed": 2026100602,
    "central_openings": str(ROOT / "tools" / "external" / "openings_center_rush_sound_5k.jsonl"),
    "normal_openings": str(ROOT / "tools" / "external" / "openings_normal_confirm_400.jsonl"),
    "nnue": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "output": str(ROOT / "results" / "experiments" / "ab_mix_h2h_20261006"),
}

BASELINE_ARGS = (
    "--leaf-depth 0 "
    "--endgame-mover-walls 0 --endgame-leaf-depth 2 "
    "--ab-prefilter-depth 0 --ab-prefilter-topk 0 --ab-prefilter-timefrac 0.25"
)

ARMS = {
    "remove_final": (
        "--leaf-depth 0 "
        "--endgame-mover-walls -1 "
        "--ab-prefilter-depth 0 --ab-prefilter-topk 0 --ab-prefilter-timefrac 0.25"
    ),
    "deeper_final": (
        "--leaf-depth 0 "
        "--endgame-mover-walls 0 --endgame-leaf-depth 4 "
        "--ab-prefilter-depth 0 --ab-prefilter-topk 0 --ab-prefilter-timefrac 0.25"
    ),
    "root_verify": (
        "--leaf-depth 0 "
        "--endgame-mover-walls 0 --endgame-leaf-depth 2 "
        "--ab-prefilter-depth 2 --ab-prefilter-topk 12 --ab-prefilter-timefrac 0.10"
    ),
    "hybrid": (
        "--leaf-depth 0 "
        "--endgame-mover-walls 0 --endgame-leaf-depth 4 "
        "--ab-prefilter-depth 2 --ab-prefilter-topk 12 --ab-prefilter-timefrac 0.10"
    ),
}


def _build_engine() -> Path:
    if os.name == "nt":
        subprocess.run([str(ROOT / "build" / "build_uci.bat")], cwd=ROOT, check=True, shell=True)
        exe = ROOT / "bin" / "zquoridor.exe"
    else:
        subprocess.run([str(ROOT / "build" / "build_uci.sh")], cwd=ROOT, check=True)
        exe = ROOT / "bin" / "zquoridor"
    if not exe.is_file():
        raise FileNotFoundError(f"UCI engine not produced: {exe}")
    return exe


def _read_games(path: Path, suite_tag: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Opening indices overlap between the two books. Give each suite a
        # distinct opponent key so pair-aware aggregation cannot collide.
        row = dict(row)
        row["opponent"] = f"{row.get('opponent', 'baseline')}:{suite_tag}"
        rows.append(row)
    return rows


def _run_suite(*, arm: str, candidate_args: str, suite: str, openings: str,
               pairs: int, seed: int, exe: Path, output_root: Path,
               move_time_ms: int, workers: int, bootstrap: int, nnue: Path) -> dict:
    out = output_root / arm / suite
    config = dict(match_finalists.CONFIG)
    config.update({
        "engine1_name": f"{arm}",
        "engine1_executable": str(exe),
        "engine1_nnue": str(nnue),
        "engine1_args": candidate_args,
        "engine2_name": "main_control",
        "engine2_executable": str(exe),
        "engine2_nnue": str(nnue),
        "engine2_args": BASELINE_ARGS,
        "pairs": pairs,
        "move_time_ms": move_time_ms,
        "base_ms": 0,
        "increment_ms": 0,
        "workers": workers,
        "seed": seed,
        "openings": openings,
        "output": str(out),
        "bootstrap": bootstrap,
    })
    report = match_finalists.run(config)
    return {
        "suite": suite,
        "pairs": pairs,
        "games": 2 * pairs,
        "seed": seed,
        "summary": report["summary"],
        "games_path": str(out / "games.jsonl"),
    }


def run(args: argparse.Namespace) -> dict:
    exe = Path(args.executable).resolve() if args.executable else _build_engine()
    nnue = Path(args.nnue).resolve()
    if not exe.is_file():
        raise FileNotFoundError(exe)
    if not nnue.is_file():
        raise FileNotFoundError(nnue)

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = args.arms or list(ARMS)
    unknown = [name for name in selected if name not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {', '.join(unknown)}")

    full_report = {
        "schema": "zquoridor.ab_mix_h2h.v1",
        "control_args": BASELINE_ARGS,
        "move_time_ms": args.move_time_ms,
        "nnue": str(nnue),
        "executable": str(exe),
        "design": {
            "paired_colors": True,
            "central_pairs": args.central_pairs,
            "normal_pairs": args.normal_pairs,
            "games_per_arm": 2 * (args.central_pairs + args.normal_pairs),
            "global_leaf_depth": 0,
        },
        "arms": {},
    }

    for offset, arm in enumerate(selected):
        print(f"\n=== ARM {arm}: {ARMS[arm]} ===", flush=True)
        central = _run_suite(
            arm=arm, candidate_args=ARMS[arm], suite="center_rush",
            openings=args.central_openings, pairs=args.central_pairs,
            seed=args.central_seed, exe=exe, output_root=output_root,
            move_time_ms=args.move_time_ms, workers=args.workers,
            bootstrap=args.bootstrap, nnue=nnue,
        )
        normal = _run_suite(
            arm=arm, candidate_args=ARMS[arm], suite="normal",
            openings=args.normal_openings, pairs=args.normal_pairs,
            seed=args.normal_seed, exe=exe, output_root=output_root,
            move_time_ms=args.move_time_ms, workers=args.workers,
            bootstrap=args.bootstrap, nnue=nnue,
        )

        rows = []
        rows += _read_games(Path(central["games_path"]), "center_rush")
        rows += _read_games(Path(normal["games_path"]), "normal")
        combined = local_arena.summarize_pairs(
            rows, bootstrap=args.bootstrap, seed=args.central_seed + args.normal_seed + offset
        )
        full_report["arms"][arm] = {
            "candidate_args": ARMS[arm],
            "center_rush": central,
            "normal": normal,
            "combined": combined,
        }

        print(
            f"{arm}: combined score={combined.get('score_pct')}% "
            f"Elo={combined.get('elo')} "
            f"95%={combined.get('paired_bootstrap_95')}",
            flush=True,
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")
    return full_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", choices=sorted(ARMS))
    p.add_argument("--executable", default="", help="Use an existing UCI binary instead of rebuilding.")
    p.add_argument("--nnue", default=CONFIG["nnue"])
    p.add_argument("--move-time-ms", type=int, default=CONFIG["move_time_ms"])
    p.add_argument("--workers", type=int, default=CONFIG["workers"])
    p.add_argument("--bootstrap", type=int, default=CONFIG["bootstrap"])
    p.add_argument("--central-pairs", type=int, default=CONFIG["central_pairs"])
    p.add_argument("--normal-pairs", type=int, default=CONFIG["normal_pairs"])
    p.add_argument("--central-seed", type=int, default=CONFIG["central_seed"])
    p.add_argument("--normal-seed", type=int, default=CONFIG["normal_seed"])
    p.add_argument("--central-openings", default=CONFIG["central_openings"])
    p.add_argument("--normal-openings", default=CONFIG["normal_openings"])
    p.add_argument("--output", default=CONFIG["output"])
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ab-mix h2h error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
