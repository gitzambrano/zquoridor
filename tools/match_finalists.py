#!/usr/bin/env python3
"""Run a direct head-to-head match between two Zquoridor NNUE candidates."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_benchmark
from tools.external import local_arena

CONFIG = {
    "engine1_name": "race512-search10-ft",
    "engine1_executable": str(ROOT / "results" / "experiments" / "race512-search10-ft-s20260917" / "zquoridor.exe"),
    "engine1_nnue": str(ROOT / "results" / "experiments" / "race512-search10-ft-s20260917" / "student_int8.bin"),
    "engine2_name": "base512-search10-ft",
    "engine2_executable": str(ROOT / "results" / "experiments" / "base512-search10-ft-s20260917" / "zquoridor.exe"),
    "engine2_nnue": str(ROOT / "results" / "experiments" / "base512-search10-ft-s20260917" / "student_int8.bin"),
    "pairs": 100,
    "move_time_ms": 200,
    "workers": 2,
    "seed": 20260920,
    "openings": str(ROOT / "tools" / "external" / "openings_confirmation_v1.jsonl"),
    "output": str(ROOT / "results" / "benchmarks" / "finalists-h2h-race512-vs-base512-200ms"),
    "bootstrap": 20000,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs: dict[str, object] = {"default": None}
        if isinstance(value, int):
            kwargs["type"] = int
        else:
            kwargs["type"] = str
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    for key in ("engine1_executable", "engine1_nnue", "engine2_executable", "engine2_nnue", "openings"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} does not exist: {config[key]}")
    for key in ("pairs", "move_time_ms", "workers", "bootstrap"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def run(config: dict) -> dict:
    openings = run_benchmark._read_openings(Path(config["openings"]), int(config["pairs"]), int(config["seed"]))
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    e1_exe = Path(config["engine1_executable"]).resolve()
    e1_nnue = Path(config["engine1_nnue"]).resolve()
    e2_exe = Path(config["engine2_executable"]).resolve()
    e2_nnue = Path(config["engine2_nnue"]).resolve()

    manifest = local_arena.make_manifest(
        {
            "protocol": "finalists-h2h-fixed-clock-v1",
            "pairs": int(config["pairs"]),
            "move_time_ms": int(config["move_time_ms"]),
            "seed": int(config["seed"]),
            "engine1_name": config["engine1_name"],
            "engine2_name": config["engine2_name"],
            "openings": [index for index, _ in openings],
        },
        {
            "engine1_executable": e1_exe,
            "engine1_nnue": e1_nnue,
            "engine2_executable": e2_exe,
            "engine2_nnue": e2_nnue,
            "referee": ROOT / "tools" / "external" / "local_arena.py",
        },
    )

    games_path, old_rows = local_arena.prepare_resume(output, manifest)
    latest = {(int(row["opening_index"]), int(row["zq_player"])): row for row in old_rows}

    def play(index: int, opening: list[str], side: int) -> dict:
        return local_arena.play_game(
            opponent=config["engine2_name"],
            opening_index=index,
            opening=opening,
            zq_player=side,
            zq_factory=lambda: local_arena.UciPlayer([str(e1_exe), "--nnue", str(e1_nnue)], config["engine1_name"]),
            opponent_factory=lambda: local_arena.UciPlayer([str(e2_exe), "--nnue", str(e2_nnue)], config["engine2_name"]),
            zq_budget=int(config["move_time_ms"]),
            opponent_budget=int(config["move_time_ms"]),
            move_timeout_s=30.0,
            max_plies=240,
            run_id=manifest["run_id"],
        )

    pending = [
        (index, opening, side)
        for index, opening in openings
        for side in (0, 1)
        if latest.get((index, side), {}).get("status") != "ok"
    ]

    print(f"Total pairs: {config['pairs']} ({2 * int(config['pairs'])} games), pending: {len(pending)}")
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["workers"])) as pool:
            futures = [pool.submit(play, *task) for task in pending]
            done_count = len(latest)
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[int(row["opening_index"]), int(row["zq_player"])] = row
                done_count += 1
                if done_count % 10 == 0 or done_count == 2 * int(config["pairs"]):
                    print(f"[{done_count}/{2 * int(config['pairs'])}] opening={row['opening_index']} side={row['zq_player']} result={row.get('result')}", flush=True)

    summary = local_arena.summarize_pairs(list(latest.values()), bootstrap=int(config["bootstrap"]), seed=int(config["seed"]))
    report = {
        "schema": "zquoridor.finalists_h2h.v1",
        "engine1": config["engine1_name"],
        "engine2": config["engine2_name"],
        "move_time_ms": int(config["move_time_ms"]),
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n--- FINAL H2H RESULT ---")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run(resolve_config(build_parser().parse_args(argv)))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"h2h error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
