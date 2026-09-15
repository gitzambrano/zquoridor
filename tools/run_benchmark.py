#!/usr/bin/env python3
"""Build dependencies and run reproducible local strength benchmarks."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Edit this block for normal local use. Command-line options override these values.
CONFIG = {
    "opponents": ["titanium", "claustrophobia"],
    "pairs": 20,
    "workers": 1,
    "seed": 20260914,
    "openings": str(ROOT / "tools" / "external" / "openings_titanium.jsonl"),
    "output": str(ROOT / "benchmark_results" / "local"),
    "resume": True,
    "retry_failed": True,
    "auto_setup": True,
    "zq_executable": None,
    "nnue": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "zq_args": [],
    "zq_move_time_ms": 200,
    "titanium_move_time_ms": 200,
    "claustrophobia_sims": 512,
    "claustrophobia_cpuct": 1.5,
    "claustrophobia_device": "cpu",
    "startup_timeout_s": 120.0,
    "move_timeout_s": 30.0,
    "max_plies": 180,
    "bootstrap": 20000,
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external import local_arena  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", help="Use a comma-separated list of opponents.")
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--openings")
    parser.add_argument("--output")
    parser.add_argument("--zq-executable")
    parser.add_argument("--nnue")
    parser.add_argument("--zq-arg", dest="zq_args", action="append")
    parser.add_argument("--zq-move-time-ms", type=int)
    parser.add_argument("--titanium-move-time-ms", type=int)
    parser.add_argument("--claustrophobia-sims", type=int)
    parser.add_argument("--claustrophobia-cpuct", type=float)
    parser.add_argument("--claustrophobia-device", choices=("cpu", "gpu"))
    parser.add_argument("--startup-timeout-s", type=float)
    parser.add_argument("--move-timeout-s", type=float)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-setup", action=argparse.BooleanOptionalAction, default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    if isinstance(config["opponents"], str):
        config["opponents"] = [item.strip().lower() for item in config["opponents"].split(",")
                               if item.strip()]
    config["zq_args"] = list(config.get("zq_args") or [])
    supported = {"titanium", "claustrophobia"}
    unknown = set(config["opponents"]) - supported
    if unknown:
        raise ValueError(f"unknown opponent: {', '.join(sorted(unknown))}")
    for key in ("pairs", "workers", "zq_move_time_ms", "titanium_move_time_ms",
                "claustrophobia_sims", "max_plies", "bootstrap"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("startup_timeout_s", "move_timeout_s", "claustrophobia_cpuct"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def _read_openings(path: Path, pairs: int, seed: int) -> list[tuple[int, list[str]]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        moves = value.get("moves")
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise ValueError(f"{path}:{index + 1}: expected a string list in 'moves'")
        referee = local_arena.Referee()
        for move in moves:
            referee.apply(move)
        if referee.winner is not None:
            raise ValueError(f"{path}:{index + 1}: the opening is terminal")
        rows.append((index, moves))
    if len(rows) < pairs:
        raise ValueError(f"the opening file has {len(rows)} rows but the run needs {pairs}")
    random.Random(seed).shuffle(rows)
    return rows[:pairs]


def _build_zq(output: Path) -> Path:
    source = ROOT / "tools" / "external" / "zquoridor_uci.cpp"
    headers = list((ROOT / "src").glob("*.hpp"))
    if output.is_file() and output.stat().st_mtime_ns >= max(
            path.stat().st_mtime_ns for path in [source, *headers]):
        return output.resolve()
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is required to build the Zquoridor benchmark adapter")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-std=c++17", "-march=native"]
    if platform.machine().lower() in ("amd64", "x86_64"):
        command.extend(["-mavx2", "-mfma"])
    command.extend(["-I", str(ROOT / "src"), str(source), "-o", str(output)])
    print("Build Zquoridor:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return output.resolve()


def _bot_info(name: str, auto_setup: bool) -> dict:
    try:
        from tools.external.bot_setup import ensure_bot
    except ImportError as exc:
        raise RuntimeError("tools.external.bot_setup is not available") from exc
    return ensure_bot(name, root=ROOT, build=auto_setup)


def _task_key(row: dict) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["opening_index"]), int(row["zq_player"])


def run(config: dict) -> dict:
    openings_path = Path(config["openings"]).resolve()
    nnue = Path(config["nnue"]).resolve()
    if not nnue.is_file():
        raise FileNotFoundError(f"NNUE weights not found: {nnue}")
    if config["zq_executable"]:
        zq_executable = Path(config["zq_executable"]).resolve()
        if not zq_executable.is_file():
            raise FileNotFoundError(f"Zquoridor executable not found: {zq_executable}")
    else:
        suffix = ".exe" if sys.platform == "win32" else ""
        zq_executable = _build_zq(ROOT / "bin" / "local_benchmark" / f"zquoridor_uci{suffix}")

    bot_info = {name: _bot_info(name, bool(config["auto_setup"]))
                for name in config["opponents"]}
    if "claustrophobia" in bot_info and "benchmark_bridge" not in bot_info["claustrophobia"]:
        raise RuntimeError(
            "the Claustrophobia setup did not provide benchmark_bridge; "
            "the benchmark requires a persistent model process"
        )
    openings = _read_openings(openings_path, int(config["pairs"]), int(config["seed"]))

    identity_config = {key: value for key, value in config.items()
                       if key not in ("resume", "retry_failed", "auto_setup", "output")}
    identity_config["openings"] = str(openings_path)
    identity_config["nnue"] = str(nnue)
    identity_config["zq_executable"] = str(zq_executable)
    identity_config["selected_opening_indices"] = [index for index, _ in openings]
    identity_config["compute"] = {
        "zquoridor": "cpu fixed move time",
        "titanium": "cpu fixed move time" if "titanium" in bot_info else None,
        "claustrophobia": (
            f"{config['claustrophobia_device']} exact MCTS simulations"
            if "claustrophobia" in bot_info else None
        ),
    }
    identity_config["claustrophobia_backend"] = "upstream-mcts-python-torch-ipc"
    artifacts = {"zq_executable": zq_executable, "nnue": nnue, "openings": openings_path,
                 "referee": ROOT / "tools/external/local_arena.py"}
    if "titanium" in bot_info:
        artifacts["titanium_executable"] = Path(bot_info["titanium"]["executable"])
    if "claustrophobia" in bot_info:
        artifacts["inference_worker"] = ROOT / "training/teachers/claustrophobia_inference_worker.py"
        artifacts["claustrophobia_checkpoint"] = Path(bot_info["claustrophobia"]["checkpoint"])
        artifacts["claustrophobia_bridge"] = Path(bot_info["claustrophobia"]["benchmark_bridge"])
    manifest = local_arena.make_manifest(identity_config, artifacts)
    output = Path(config["output"]).resolve()
    if config["resume"]:
        games_path, prior = local_arena.prepare_resume(output, manifest)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("the output directory is not empty; enable resume or select another directory")
        games_path, prior = local_arena.prepare_resume(output, manifest)

    latest = {_task_key(row): row for row in prior}
    tasks = []
    for opponent in config["opponents"]:
        for opening_index, opening in openings:
            for zq_player in (0, 1):
                key = (opponent, opening_index, zq_player)
                old = latest.get(key)
                if old and (old.get("status") == "ok" or not config["retry_failed"]):
                    continue
                tasks.append((opponent, opening_index, opening, zq_player))

    zq_command = [str(zq_executable), "--nnue", str(nnue), *config["zq_args"]]

    def one(task: tuple[str, int, list[str], int]) -> dict:
        opponent, opening_index, opening, zq_player = task
        zq_factory = lambda: local_arena.UciPlayer(
            zq_command, "zquoridor", startup_timeout_s=float(config["startup_timeout_s"])
        )
        if opponent == "titanium":
            info = bot_info[opponent]
            opponent_factory = lambda: local_arena.TitaniumPlayer(
                Path(info["executable"]), startup_timeout_s=float(config["startup_timeout_s"])
            )
            opponent_budget = int(config["titanium_move_time_ms"])
        else:
            info = bot_info[opponent]
            opponent_factory = lambda: local_arena.ClaustrophobiaPlayer(
                Path(info["benchmark_bridge"]), Path(info["checkpoint"]),
                sims=int(config["claustrophobia_sims"]),
                cpuct=float(config["claustrophobia_cpuct"]),
                device=str(config["claustrophobia_device"]),
                startup_timeout_s=float(config["startup_timeout_s"]),
            )
            opponent_budget = int(config["claustrophobia_sims"])
        return local_arena.play_game(
            opponent=opponent,
            opening_index=opening_index,
            opening=opening,
            zq_player=zq_player,
            zq_factory=zq_factory,
            opponent_factory=opponent_factory,
            zq_budget=int(config["zq_move_time_ms"]),
            opponent_budget=opponent_budget,
            move_timeout_s=float(config["move_timeout_s"]),
            max_plies=int(config["max_plies"]),
            run_id=manifest["run_id"],
        )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "pending_games": len(tasks),
        "pairs_per_opponent": config["pairs"],
        "workers": config["workers"],
        "budgets": {
            "zquoridor": {"type": "move_time_ms", "value": config["zq_move_time_ms"],
                           "device": "cpu"},
            "titanium": {"type": "move_time_ms", "value": config["titanium_move_time_ms"],
                         "device": "cpu"},
            "claustrophobia": {"type": "mcts_simulations", "value": config["claustrophobia_sims"],
                               "device": config["claustrophobia_device"]},
        },
    }, indent=2), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["workers"])) as pool:
            futures = [pool.submit(one, task) for task in tasks]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[_task_key(row)] = row
                print(f"[{done}/{len(tasks)}] {row['opponent']} opening={row['opening_index']} "
                      f"zq_player={row['zq_player']} status={row['status']}", flush=True)

    summaries = {}
    for offset, opponent in enumerate(config["opponents"]):
        rows = [row for key, row in latest.items() if key[0] == opponent]
        summaries[opponent] = local_arena.summarize_pairs(
            rows, bootstrap=int(config["bootstrap"]), seed=int(config["seed"]) + offset
        )
    report = {
        "schema": "zquoridor.local_benchmark.report.v1",
        "run_id": manifest["run_id"],
        "budgets": {
            "zquoridor": {"type": "move_time_ms", "value": config["zq_move_time_ms"],
                           "device": "cpu"},
            "titanium": {"type": "move_time_ms", "value": config["titanium_move_time_ms"],
                         "device": "cpu"},
            "claustrophobia": {"type": "mcts_simulations", "value": config["claustrophobia_sims"],
                               "device": config["claustrophobia_device"]},
        },
        "summaries": summaries,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        config = resolve_config(build_parser().parse_args(argv))
        report = run(config)
        return 1 if any(s["failed_games"] for s in report["summaries"].values()) else 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
