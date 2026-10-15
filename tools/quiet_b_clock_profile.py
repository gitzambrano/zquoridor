#!/usr/bin/env python3
"""Profile root preparation and simulations for quiet-wall search variants.

Creates opening books disjoint from the previous 500-game deep comparison.
The modes use the same network, executable, positions, and 200 ms per move.
All generated measurements remain outside Git.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "output": str(ROOT / "results" / "quiet_walls" / "clock_profile"),
    "normal_source": str(ROOT / "tools" / "external" / "openings_normal_confirm_400.jsonl"),
    "center_source": str(ROOT / "tools" / "external" / "openings_center_rush_sound_5k.jsonl"),
    "normal_count": 60,
    "center_count": 60,
    "sample_per_book": 24,
    "move_time_ms": 200,
    "executable": str(ROOT / "bin" / "quiet_walls_profile" / "zquoridor_uci"),
    "nnue": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "generate_only": False,
}

PREVIOUS_SCREEN = {
    "normal": (20261027, 125),
    "center": (20261028, 125),
}
NEW_SEEDS = {"normal": 20261105, "center": 20261106}
MODES = {
    "main": (),
    "main_charged": ("--charge-root-setup",),
    "B08": ("--quiet-walls-mode", "2", "--quiet-walls-weight", "0.08"),
    "B04": ("--quiet-walls-mode", "2", "--quiet-walls-weight", "0.04"),
}


def generate_books(config: dict) -> dict:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for category in ("normal", "center"):
        source = Path(config[f"{category}_source"])
        raw = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        all_indices = list(range(len(raw)))
        old_seed, old_pairs = PREVIOUS_SCREEN[category]
        excluded = all_indices.copy()
        random.Random(old_seed).shuffle(excluded)
        excluded = set(excluded[:old_pairs])
        fresh = [index for index in all_indices if index not in excluded]
        random.Random(NEW_SEEDS[category]).shuffle(fresh)
        n = int(config[f"{category}_count"])
        if n < 1 or len(fresh) < n:
            raise ValueError(f"{category}: {len(fresh)} fresh positions available, requested {n}")
        selected = fresh[:n]
        assert excluded.isdisjoint(selected), "opening leakage"
        path = output / f"{category}_fresh.jsonl"
        path.write_text("".join(json.dumps(raw[i], separators=(",", ":")) + "\n"
                                for i in selected))
        manifest[category] = {
            "source": str(source), "fresh_file": str(path),
            "selected_indices": selected, "prior_indices": sorted(excluded),
            "prior_seed": old_seed, "new_seed": NEW_SEEDS[category],
            "prior_pairs": old_pairs, "fresh_pairs": n,
        }
    (output / "openings_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def percentile(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, round((len(v) - 1) * q))]


def profile_engine(config: dict, args: tuple[str, ...], positions: list[tuple[str, list[str]]]):
    argv = [str(Path(config["executable"]).resolve()),
            "--nnue", str(Path(config["nnue"]).resolve()), *args]
    process = subprocess.Popen(argv, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    assert process.stdin is not None and process.stdout is not None

    def command(line: str, terminator: str) -> list[str]:
        process.stdin.write(line + "\n")
        process.stdin.flush()
        output = []
        while True:
            reply = process.stdout.readline()
            if not reply:
                raise RuntimeError("UCI terminated: " + process.stderr.read()[-2000:])
            output.append(reply.strip())
            if reply.strip() == terminator or reply.startswith(terminator + " "):
                return output

    rows = []
    try:
        command("uci", "uciok")
        command("isready", "readyok")
        for category, moves in positions:
            command("ucinewgame", "readyok") if False else None
            process.stdin.write("ucinewgame\n")
            process.stdin.flush()
            cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
            process.stdin.write(cmd + "\n")
            process.stdin.flush()
            logs = command(f"go movetime {config['move_time_ms']}", "bestmove")
            info = next((line for line in logs if line.startswith("info depth ")), None)
            if info is None:
                raise RuntimeError("Missing search stats for " + str(moves) + ": " + str(logs))
            numeric = {}
            for key in ("time", "sims", "rootprep_us", "quiet_us",
                        "tree_us", "treebudget_ms", "chargeSetup", "qmode",
                        "treeHit", "reusedNodes", "nodes"):
                match = re.search(r"(?:^|\\s)" + key + r"=(\\d+)", info)
                if key in ("time", "nodes"):
                    match = re.search(r"(?:^|\\s)" + key + r"\\s+(\\d+)", info)
                if match:
                    numeric[key] = int(match.group(1))
            if set(("sims", "rootprep_us", "quiet_us", "tree_us", "treebudget_ms")) - numeric.keys():
                raise RuntimeError("Incomplete stats: " + info)
            numeric.update({"category": category, "opening": moves,
                            "bestmove": logs[-1].split()[1]})
            rows.append(numeric)
    finally:
        try:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=4)
        except Exception:
            process.kill()
            process.wait()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in CONFIG.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true")
        elif isinstance(value, int):
            parser.add_argument(flag, type=int)
        else:
            parser.add_argument(flag)
    args = parser.parse_args()
    config = {name: getattr(args, name) if getattr(args, name) is not None else value
              for name, value in CONFIG.items()}
    manifest = generate_books(config)
    print("Fresh opening identities:", {k: v["selected_indices"][:8]
                                        for k, v in manifest.items()}, flush=True)
    if config["generate_only"]:
        return 0
    positions = []
    for category in ("normal", "center"):
        path = Path(manifest[category]["fresh_file"])
        for row in path.read_text().splitlines()[:config["sample_per_book"]]:
            positions.append((category, json.loads(row)["moves"]))
    profiles = {}
    for name, options in MODES.items():
        rows = profile_engine(config, options, positions)
        stats = {}
        for key in ("time", "sims", "rootprep_us", "quiet_us", "tree_us", "treebudget_ms", "nodes"):
            values = [float(row[key]) for row in rows if key in row]
            stats[key] = {"mean": round(statistics.mean(values), 3),
                          "median": round(statistics.median(values), 3),
                          "p95": round(percentile(values, .95), 3)} if values else None
        profiles[name] = {"args": options, "stats": stats, "samples": rows}
        print(name, json.dumps(stats), flush=True)
    target = Path(config["output"]) / "clock_profile.json"
    target.write_text(json.dumps({"schema": "zquoridor.quiet_clock_profile.v1",
                                  "requested_move_time_ms": config["move_time_ms"],
                                  "positions_per_mode": len(positions),
                                  "modes": profiles}, indent=2) + "\n")
    print("Profile saved:", target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
