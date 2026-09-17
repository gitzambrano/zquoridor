#!/usr/bin/env python3
"""Compare one NNUE candidate against main and the external reference bots."""
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

# Edit this block for normal local use. Command-line options override these values.
CONFIG = {
    "candidate_executable": None,
    "candidate_nnue": None,
    "baseline_executable": None,
    "baseline_nnue": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "pairs": 20,
    "move_time_ms": 200,
    "workers": 1,
    "seed": 20260916,
    "openings": str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
    "output": str(ROOT / "benchmark_results" / "candidate"),
    "claustrophobia_device": "cpu",
    "claustrophobia_max_sims": 4096,
    "benchmark_main_external": True,
    "resume": True,
    "auto_setup": True,
    "bootstrap": 20000,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs: dict[str, object] = {"default": None}
        if isinstance(value, bool):
            kwargs = {"action": argparse.BooleanOptionalAction, "default": None}
        elif isinstance(value, int):
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
    for key in ("candidate_executable", "candidate_nnue"):
        if not config[key]:
            raise ValueError(f"{key} is required")
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} does not exist: {config[key]}")
    if config["baseline_executable"]:
        if not Path(config["baseline_executable"]).is_file():
            raise FileNotFoundError(f"baseline_executable does not exist: {config['baseline_executable']}")
    for key in ("baseline_nnue", "openings"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} does not exist: {config[key]}")
    for key in ("pairs", "move_time_ms", "workers", "claustrophobia_max_sims", "bootstrap"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def _baseline_executable(config: dict) -> Path:
    if config["baseline_executable"]:
        return Path(config["baseline_executable"]).resolve()
    suffix = ".exe" if sys.platform == "win32" else ""
    return run_benchmark._build_zq(ROOT / "bin" / "local_benchmark" / f"zquoridor_uci{suffix}")


def _baseline_matrix(config: dict, openings: list[tuple[int, list[str]]], output: Path) -> dict:
    candidate = (Path(config["candidate_executable"]).resolve(), Path(config["candidate_nnue"]).resolve())
    baseline = (_baseline_executable(config), Path(config["baseline_nnue"]).resolve())
    manifest = local_arena.make_manifest(
        {"protocol": "candidate-main-fixed-clock-v1", "pairs": config["pairs"],
         "move_time_ms": config["move_time_ms"], "seed": config["seed"],
         "openings": [index for index, _ in openings]},
        {"candidate_executable": candidate[0], "candidate_nnue": candidate[1],
         "baseline_executable": baseline[0], "baseline_nnue": baseline[1],
         "referee": ROOT / "tools" / "external" / "local_arena.py"},
    )
    games_path, old_rows = local_arena.prepare_resume(output, manifest)
    latest = {(int(row["opening_index"]), int(row["zq_player"])): row for row in old_rows}

    def play(index: int, opening: list[str], side: int) -> dict:
        return local_arena.play_game(
            opponent="main", opening_index=index, opening=opening, zq_player=side,
            zq_factory=lambda: local_arena.UciPlayer([str(candidate[0]), "--nnue", str(candidate[1])], "candidate"),
            opponent_factory=lambda: local_arena.UciPlayer([str(baseline[0]), "--nnue", str(baseline[1])], "main"),
            zq_budget=int(config["move_time_ms"]), opponent_budget=int(config["move_time_ms"]),
            move_timeout_s=30.0, max_plies=240, run_id=manifest["run_id"],
        )

    pending = [(index, opening, side) for index, opening in openings for side in (0, 1)
               if latest.get((index, side), {}).get("status") != "ok"]
    output.mkdir(parents=True, exist_ok=True)
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["workers"])) as pool:
            futures = [pool.submit(play, *task) for task in pending]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[int(row["opening_index"]), int(row["zq_player"])] = row
    return local_arena.summarize_pairs(list(latest.values()), bootstrap=int(config["bootstrap"]), seed=int(config["seed"]))


def run(config: dict) -> dict:
    openings = run_benchmark._read_openings(Path(config["openings"]), int(config["pairs"]), int(config["seed"]))
    output = Path(config["output"]).resolve()
    main_report = _baseline_matrix(config, openings, output / "vs-main")
    external_config = dict(run_benchmark.CONFIG,
        opponents=["titanium", "claustrophobia"], pairs=int(config["pairs"]), workers=int(config["workers"]),
        seed=int(config["seed"]), openings=str(Path(config["openings"]).resolve()),
        output=str(output / "vs-external"), resume=bool(config["resume"]), auto_setup=bool(config["auto_setup"]),
        zq_executable=str(Path(config["candidate_executable"]).resolve()), nnue=str(Path(config["candidate_nnue"]).resolve()),
        zq_move_time_ms=int(config["move_time_ms"]), titanium_move_time_ms=int(config["move_time_ms"]),
        claustrophobia_move_time_ms=int(config["move_time_ms"]),
        claustrophobia_max_sims=int(config["claustrophobia_max_sims"]),
        claustrophobia_device=str(config["claustrophobia_device"]), bootstrap=int(config["bootstrap"]),
    )
    external_report = run_benchmark.run(external_config)
    main_external = None
    if config["benchmark_main_external"]:
        main_external_config = dict(external_config,
            zq_executable=str(_baseline_executable(config)),
            nnue=str(Path(config["baseline_nnue"]).resolve()),
            output=str(output / "main-vs-external"),
        )
        main_external = run_benchmark.run(main_external_config)
    report = {"schema": "zquoridor.candidate_matrix.v1", "move_time_ms": config["move_time_ms"],
              "main": main_report, "external": external_report["summaries"],
              "main_external": None if main_external is None else main_external["summaries"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run(resolve_config(build_parser().parse_args(argv)))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"candidate benchmark error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
