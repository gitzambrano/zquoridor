#!/usr/bin/env python3
"""Validate deadline searches and native game clocks for local engines."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_claustrophobia_dynamic_bridge import ensure_bridge as claustro_bridge
from tools.build_titanium_clock_bridge import ensure_bridge as titanium_bridge
from tools.external import local_arena

CONFIG = {
    "shared_root": ROOT,
    "output": ROOT / "results/benchmarks/clock_protocol_validation.json",
    "checkpoint": ROOT / "external_bots/claustrophobia/champion.pt",
    "device": "cpu",
    "cpuct": 1.5,
    "workers": 1,
    "movetimes_ms": [80, 200, 333],
    "white_ms": 180000,
    "black_ms": 180000,
    "increment_ms": 2000,
    "max_single_wave_overshoot_ms": 100.0,
    "timeout_s": 120.0,
    "zq_executable": None,
    "weights": [],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legal_move(history: list[str], move: str) -> None:
    referee = local_arena.Referee()
    for item in history:
        referee.apply(item)
    referee.apply(move)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate(config: dict) -> dict:
    started = time.monotonic()
    report = {"protocol": "native-clock-v2", "status": "RUNNING", "cases": [],
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in config.items()}}
    target = Path(config["output"])
    try:
        cla = claustro_bridge(Path(config["shared_root"]))
        ti = titanium_bridge(Path(config["shared_root"]))
        report["executables"] = {"claustrophobia": str(cla), "titanium": str(ti)}
        report["executable_sha256"] = {"claustrophobia": digest(cla), "titanium": digest(ti)}
        report["source_sha256"] = {name: digest(ROOT / name) for name in (
            "tools/external/local_arena.py", "tools/build_claustrophobia_dynamic_bridge.py",
            "tools/external/claustrophobia_benchmark_bridge.rs", "tools/build_titanium_clock_bridge.py")}

        def create_player():
            return local_arena.ClaustrophobiaPlayer(
                cla, Path(config["checkpoint"]), move_time_ms=200, max_sims=4294967295,
                cpuct=float(config["cpuct"]), device=config["device"],
                startup_timeout_s=float(config["timeout_s"]))

        def record_claustro(case, history, response):
            move, elapsed, info = response
            search = json.loads(info[-1])
            legal_move(history, move)
            require(search["stop_reason"] == "solved" or search["search_ms"] >= search["move_time_ms"],
                    "Claustrophobia stopped before its deadline without a proof")
            require(search["overshoot_ms"] <= config["max_single_wave_overshoot_ms"],
                    f"Claustrophobia exceeded the permitted wave overshoot: {search['overshoot_ms']}")
            require(len(search["visit_counts"]) == 209 and (search["stop_reason"] == "solved" or
                    sum(search["visit_counts"]) == search["root_visits"]),
                    "Claustrophobia root visits do not match its policy")
            require(search["policy_frame"] == "claustrophobia-canonical-209", "Unexpected Claustrophobia policy frame")
            report["cases"].append({"engine": "claustrophobia", "case": case, "history": history,
                                    "driver_elapsed_ms": elapsed * 1000, "search": search})

        player = create_player()
        try:
            for budget in config["movetimes_ms"]:
                record_claustro("movetime", [], player.bestmove([], budget=budget, timeout_s=config["timeout_s"]))
            for history in ([], ["e2"]):
                record_claustro("game-clock", history, player.bestmove_clock(
                    history, white_ms=config["white_ms"], black_ms=config["black_ms"],
                    increment_ms=config["increment_ms"], timeout_s=config["timeout_s"]))
        finally:
            player.close()

        if config["workers"] > 1:
            players = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as pool:
                futures = [pool.submit(create_player) for _ in range(config["workers"])]
                try:
                    players = [future.result() for future in futures]
                    probes = [pool.submit(p.bestmove, [], budget=200, timeout_s=config["timeout_s"]) for p in players]
                    for probe in probes:
                        record_claustro("concurrent-movetime", [], probe.result())
                finally:
                    for future in futures:
                        if future.done() and not future.cancelled() and future.exception() is None:
                            future.result().close()

        player = local_arena.TitaniumPlayer(ti, startup_timeout_s=config["timeout_s"])
        try:
            for budget in config["movetimes_ms"]:
                move, elapsed, info = player.bestmove([], budget=budget, timeout_s=config["timeout_s"])
                legal_move([], move)
                report["cases"].append({"engine": "titanium", "case": "movetime",
                                        "budget_ms": budget, "driver_elapsed_ms": elapsed * 1000, "info": info})
            for history in ([], ["e2"]):
                move, elapsed, info = player.bestmove_clock(history, white_ms=config["white_ms"],
                    black_ms=config["black_ms"], increment_ms=config["increment_ms"], timeout_s=config["timeout_s"])
                legal_move(history, move)
                own = (config["white_ms"], config["black_ms"])[len(history) % 2]
                opp = (config["black_ms"], config["white_ms"])[len(history) % 2]
                require(any(f"remaining_ms={own} inc_ms={config['increment_ms']} opp_ms={opp}" in line for line in info),
                        "Titanium did not record the complete native clock request")
                report["cases"].append({"engine": "titanium", "case": "game-clock", "history": history,
                                        "driver_elapsed_ms": elapsed * 1000, "info": info})
        finally:
            player.close()

        if config["zq_executable"]:
            weights = config["weights"] or [None]
            for weight in weights:
                command = [str(config["zq_executable"])]
                if weight:
                    command += ["--nnue", str(weight)]
                player = local_arena.UciPlayer(command, "zquoridor", startup_timeout_s=config["timeout_s"])
                try:
                    for budget in config["movetimes_ms"]:
                        move, elapsed, info = player.bestmove([], budget=budget, timeout_s=config["timeout_s"])
                        legal_move([], move)
                        report["cases"].append({"engine": "zquoridor", "case": "movetime", "weights": weight,
                            "budget_ms": budget, "driver_elapsed_ms": elapsed * 1000, "info": info})
                    move, elapsed, info = player.bestmove_clock([], white_ms=config["white_ms"],
                        black_ms=config["black_ms"], increment_ms=config["increment_ms"], timeout_s=config["timeout_s"])
                    legal_move([], move)
                    report["cases"].append({"engine": "zquoridor", "case": "game-clock", "weights": weight,
                        "driver_elapsed_ms": elapsed * 1000, "info": info})
                finally:
                    player.close()
        report["status"] = "PASS"
        return report
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "output": str(target)}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("shared_root", "output", "checkpoint", "zq_executable"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--weights", action="append")
    parser.add_argument("--movetimes-ms", type=int, nargs="+")
    for key in ("workers", "white_ms", "black_ms", "increment_ms"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("cpuct", "timeout_s", "max_single_wave_overshoot_ms"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    config = dict(CONFIG)
    config.update({key: value for key, value in vars(parser.parse_args(argv)).items() if value is not None})
    for key in ("workers", "white_ms", "black_ms", "cpuct", "timeout_s"):
        require(config[key] > 0, f"{key} must be positive")
    require(config["increment_ms"] >= 0, "increment_ms must be nonnegative")
    require(all(value > 0 for value in config["movetimes_ms"]), "Move budgets must be positive")
    require(not config["weights"] or config["zq_executable"], "Weights require a Zquoridor executable")
    validate(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
