#!/usr/bin/env python3
"""Generate resumable 100 ms self-play shards from weakness snapshots."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD_BYTES = 64

# Edit this block for a generic resumable weakness corpus. CLI options override it.
CONFIG = {
    "positions": "data/teaching/weakness/positions.jsonl",
    "out": "data/selfplay_canonical_v3/weakness",
    "exe": "bin/selfplay.exe",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "target": 500_000,
    "extra": 0,
    "games_per_shard": 512,
    "threads": 12,
    "time_ms": 100,
    "stage_report": "results/experiments/current/train_report.json",
    "seed": 20260920,
    "dry_run": True,
}


def count_positions(path: Path) -> int:
    size = path.stat().st_size
    if size % RECORD_BYTES:
        raise RuntimeError(f"partial V3 shard: {path} has {size} bytes")
    return size // RECORD_BYTES


def write_progress(out_dir: Path, payload: dict) -> None:
    temporary = out_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_dir / "progress.json")


def choose_effective_target(
    current: int,
    primary: int,
    extra: int,
    stage_complete: bool,
    already_activated: bool = False,
) -> tuple[int, bool]:
    activate = extra > 0 and (
        already_activated or (current >= primary and not stage_complete)
    )
    return (primary + extra, True) if activate else (primary, False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif key in {"positions", "out", "exe", "weights", "stage_report"}:
            parser.add_argument(flag, type=Path, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=None)
    parsed = parser.parse_args(argv)
    values = dict(CONFIG)
    for key, value in vars(parsed).items():
        if value is not None:
            values[key] = value
    if values["dry_run"]:
        print(json.dumps(values, indent=2, default=str), flush=True)
        return 0
    for key in ("positions", "out", "exe", "weights", "stage_report"):
        values[key] = Path(values[key])
    args = argparse.Namespace(**values)

    positions = args.positions.resolve()
    executable = args.exe.resolve()
    weights = args.weights.resolve()
    stage_report = args.stage_report.resolve()
    out_dir = args.out.resolve()
    for required in (positions, executable, weights):
        if not required.exists():
            raise FileNotFoundError(required)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(out_dir.glob("weakness_*.bin"))
    total = sum(count_positions(path) for path in shards)
    next_shard = max((int(path.stem.rsplit("_", 1)[1]) for path in shards), default=-1) + 1
    progress_path = out_dir / "progress.json"
    previous = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    effective_target, extra_activated = choose_effective_target(
        current=total,
        primary=args.target,
        extra=args.extra,
        stage_complete=stage_report.exists(),
        already_activated=bool(previous.get("extra_activated", False)),
    )

    while total < effective_target:
        shard = out_dir / f"weakness_{next_shard:04d}.bin"
        command = [
            str(executable),
            "--games", str(args.games_per_shard),
            "--chunk-games", str(args.games_per_shard),
            "--threads", str(args.threads),
            "--time-ms", str(args.time_ms),
            "--max-plies", "140",
            "--positions", str(positions),
            "--nnue-weights", str(weights),
            "--opening-plies", "0",
            "--opening-plies2", "0",
            "--epsilon", "0",
            "--epsilon-opening2", "0",
            "--epsilon-midgame", "0",
            "--seed", str(args.seed + next_shard * 999_983),
            "--out", str(shard),
        ]
        print(f"[weakness-selfplay] shard {next_shard}: {total:,}/{effective_target:,}", flush=True)
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"self-play shard {next_shard} failed with exit code {completed.returncode}")
        produced = count_positions(shard)
        if produced == 0:
            raise RuntimeError(f"self-play shard {next_shard} produced no positions")
        total += produced
        write_progress(out_dir, {
            "schema": "zquoridor.weakness_selfplay_progress.v1",
            "positions": total,
            "target": effective_target,
            "last_shard": next_shard,
            "last_shard_positions": produced,
            "last_shard_seconds": time.time() - started,
            "threads": args.threads,
            "time_ms": args.time_ms,
            "weights": str(weights),
            "seed_positions": str(positions),
            "extra_activated": extra_activated,
        })
        next_shard += 1

        if total >= args.target and not extra_activated:
            effective_target, extra_activated = choose_effective_target(
                current=total,
                primary=args.target,
                extra=args.extra,
                stage_complete=stage_report.exists(),
            )
            if extra_activated:
                print(
                    f"[weakness-selfplay] Stage C is still training; extending once to {effective_target:,}",
                    flush=True,
                )
            else:
                break

    write_progress(out_dir, {
        "schema": "zquoridor.weakness_selfplay_progress.v1",
        "status": "complete",
        "positions": total,
        "target": effective_target,
        "shards": next_shard,
        "threads": args.threads,
        "time_ms": args.time_ms,
        "weights": str(weights),
        "seed_positions": str(positions),
        "extra_activated": extra_activated,
    })
    print(f"[weakness-selfplay] complete: {total:,} positions", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[weakness-selfplay] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
