#!/usr/bin/env python3
"""Measure UCI position replay overhead without charging it to search nodes."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

CONFIG = {
    "length": 128,
    "repetitions": 4,
    "output": "results/uci-fast-replay",
    "base_exe": "bin/zq_uci_base",
    "candidate_exe": "bin/zq_uci_fast",
    "weights": "data/nnue/nnue_weights_int8.bin",
}


def history() -> list[str]:
    return ["e2", "e8", "e1", "e9"] * 40


def run_sequence(executable: str, weights: str, moves: list[str]) -> dict:
    proc = subprocess.Popen(
        [executable, "--nnue", weights],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        def command(text: str) -> None:
            proc.stdin.write(text + "\n")
            proc.stdin.flush()

        def ready() -> None:
            command("isready")
            while True:
                line = proc.stdout.readline().strip()
                if line.startswith("info string error"):
                    raise RuntimeError(line)
                if line == "readyok":
                    return
                if not line:
                    raise RuntimeError("UCI engine closed before readyok")

        command("uci")
        while proc.stdout.readline().strip() != "uciok":
            if proc.poll() is not None:
                raise RuntimeError("UCI process terminated")
        ready()
        elapsed = []
        for index in range(1, len(moves) + 1):
            t0 = time.perf_counter_ns()
            command("position startpos moves " + " ".join(moves[:index]))
            ready()
            elapsed.append((time.perf_counter_ns() - t0) / 1e6)
        tail = " ".join(moves)
        same = []
        for _ in range(24):
            t0 = time.perf_counter_ns()
            command("position startpos moves " + tail)
            ready()
            same.append((time.perf_counter_ns() - t0) / 1e6)
        return {
            "positions": len(elapsed),
            "extension_elapsed_ms": sum(elapsed),
            "extension_mean_ms": statistics.mean(elapsed),
            "extension_last16_mean_ms": statistics.mean(elapsed[-16:]),
            "repeat_last_mean_ms": statistics.mean(same),
            "repeat_last_median_ms": statistics.median(same),
        }
    finally:
        try:
            command("quit")
            proc.wait(timeout=3.0)
        except Exception:
            proc.kill()
            proc.wait(timeout=3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base_exe", "candidate_exe", "weights", "output"):
        parser.add_argument("--" + name.replace("_", "-"), default=CONFIG[name])
    for name in ("length", "repetitions"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=CONFIG[name])
    args = parser.parse_args()
    config = {**CONFIG, **vars(args)}
    moves = history()[:config["length"]]
    base, candidate = [], []
    for index in range(config["repetitions"]):
        if index % 2:
            candidate.append(run_sequence(config["candidate_exe"], config["weights"], moves))
            base.append(run_sequence(config["base_exe"], config["weights"], moves))
        else:
            base.append(run_sequence(config["base_exe"], config["weights"], moves))
            candidate.append(run_sequence(config["candidate_exe"], config["weights"], moves))
    summary = {
        key: {
            "base": statistics.median(x[key] for x in base),
            "candidate": statistics.median(x[key] for x in candidate),
        } for key in base[0] if key != "positions"
    }
    for metrics in summary.values():
        metrics["speedup"] = metrics["base"] / metrics["candidate"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"config": config, "summary": summary,
                    "baseline": base, "candidate": candidate}, indent=2) + "\n"
    )
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
