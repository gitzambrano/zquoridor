#!/usr/bin/env python3
"""Train and benchmark an NNUE architecture matrix in one resumable run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Edit this block for normal local use. Command-line options override these values.
CONFIG = {
    "data": str(ROOT / "data" / "teaching" / "replay-historical-2m-cuda" / "dataset.npz"),
    "out_root": str(ROOT / "results" / "matrix-anneal"),
    "architectures": ["base:256", "race:256", "base:384", "race:384", "base:512", "race:512"],
    "epochs": 80,
    "batch_size": 4096,
    "device": "cuda",
    "seed": 20260916,
    "benchmark_pairs": 20,
    "move_time_ms": 200,
    "claustrophobia_device": "cpu",
    "claustrophobia_max_sims": 4096,
    "resume": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs: dict[str, object] = {"default": None}
        if isinstance(value, bool):
            kwargs = {"action": argparse.BooleanOptionalAction, "default": None}
        elif isinstance(value, int):
            kwargs["type"] = int
        elif isinstance(value, list):
            kwargs["type"] = json.loads
        else:
            kwargs["type"] = str
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    config.update({key: value for key, value in vars(args).items() if value is not None})
    if not Path(config["data"]).is_file():
        raise FileNotFoundError(f"data does not exist: {config['data']}")
    if not config["architectures"]:
        raise ValueError("architectures must not be empty")
    for item in config["architectures"]:
        family, separator, width = str(item).partition(":")
        if family not in ("base", "race") or separator != ":" or int(width) not in (128, 256, 384, 512):
            raise ValueError(f"invalid architecture: {item}")
    for key in ("epochs", "batch_size", "benchmark_pairs", "move_time_ms", "claustrophobia_max_sims"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def _status_path(root: Path) -> Path:
    return root / "matrix_status.json"


def _write_status(root: Path, config: dict, rows: list[dict]) -> None:
    _status_path(root).write_text(json.dumps({"config": config, "jobs": rows}, indent=2) + "\n", encoding="utf-8")


def run(config: dict) -> dict:
    root = Path(config["out_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for offset, item in enumerate(config["architectures"]):
        family, _, raw_width = str(item).partition(":")
        width = int(raw_width)
        name = f"{family}{width}-s{int(config['seed']) + offset}"
        folder = root / name
        benchmark = root / "benchmarks" / name
        row = {"architecture": item, "folder": str(folder), "benchmark": str(benchmark), "stage": "pending"}
        rows.append(row)
        _write_status(root, config, rows)
        if not (folder / "train_report.json").is_file():
            row["stage"] = "training"
            _write_status(root, config, rows)
            subprocess.run([
                sys.executable, str(ROOT / "training" / "run_experiment.py"),
                "--data", str(config["data"]), "--out-dir", str(folder),
                "--architecture", family, "--hidden", str(width), "--epochs", str(config["epochs"]),
                "--batch-size", str(config["batch_size"]), "--device", str(config["device"]),
                "--seed", str(int(config["seed"]) + offset), "--no-teaching", "--no-benchmark",
            ], check=True, cwd=ROOT)
        if not (folder / "zquoridor.exe").is_file() or not (folder / "student_int8.bin").is_file():
            raise RuntimeError(f"training did not create the candidate artifacts for {item}")
        if not (benchmark / "summary.json").is_file():
            row["stage"] = "benchmarking"
            _write_status(root, config, rows)
            subprocess.run([
                sys.executable, str(ROOT / "tools" / "benchmark_candidate.py"),
                "--candidate-executable", str(folder / "zquoridor.exe"),
                "--candidate-nnue", str(folder / "student_int8.bin"),
                "--pairs", str(config["benchmark_pairs"]), "--move-time-ms", str(config["move_time_ms"]),
                "--seed", str(int(config["seed"]) + offset), "--output", str(benchmark),
                "--claustrophobia-device", str(config["claustrophobia_device"]),
                "--claustrophobia-max-sims", str(config["claustrophobia_max_sims"]),
            ], check=True, cwd=ROOT)
        row["stage"] = "complete"
        _write_status(root, config, rows)
    return {"status": "complete", "jobs": rows}


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(resolve_config(build_parser().parse_args(argv)))
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"architecture matrix error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
