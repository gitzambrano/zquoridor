#!/usr/bin/env python3
"""Run a resilient, chunked self-play worker on Google Colab or Linux.

Saves self-play shards (V3 format + aligned metadata) directly to Google Drive.
Automatically detects existing shards to resume after disconnections without
re-generating games or duplicating seeds.

Edit the top-level CONFIG or pass CLI arguments to override settings.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "worker_id": 1,
    "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_10m",
    "total_games": 28000,
    "chunk_games": 250,
    "time_ms": 200,
    "threads": 2,
    "positions": "tools/external/openings_center_rush_sound_5k.jsonl",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "exe": "bin/selfplay",
    "mc_mode": True,
    "mc_obvious_plies": 2,
    "mc_temp_obvious": 0.05,
    "mc_temp_opening": 0.35,
    "mc_temp_decay_plies": 45,
    "mc_temp_end": 0.12,
    "base_seed": 20260925,
}


def find_next_shard_index(drive_dir: Path, worker_id: int) -> int:
    """Find the next shard index based on existing files in the drive directory."""
    if not drive_dir.exists():
        return 0
    pattern = re.compile(rf"^c{worker_id}_shard_(\d+)\.bin$")
    highest = -1
    for f in drive_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, default=CONFIG["worker_id"])
    parser.add_argument("--drive-dir", type=str, default=CONFIG["drive_dir"])
    parser.add_argument("--total-games", type=int, default=CONFIG["total_games"])
    parser.add_argument("--chunk-games", type=int, default=CONFIG["chunk_games"])
    parser.add_argument("--time-ms", type=int, default=CONFIG["time_ms"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--positions", type=str, default=CONFIG["positions"])
    parser.add_argument("--weights", type=str, default=CONFIG["weights"])
    parser.add_argument("--exe", type=str, default=CONFIG["exe"])
    args = parser.parse_args(argv)

    drive_dir = Path(args.drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)

    exe_path = ROOT / args.exe
    if not exe_path.exists() and (ROOT / (args.exe + ".exe")).exists():
        exe_path = ROOT / (args.exe + ".exe")

    if not exe_path.exists():
        sys.exit(f"Error: selfplay binary not found at {exe_path}. Build it first.")

    positions_path = ROOT / args.positions
    if not positions_path.exists():
        sys.exit(f"Error: opening positions file not found at {positions_path}")

    weights_path = ROOT / args.weights
    if not weights_path.exists():
        sys.exit(f"Error: NNUE weights not found at {weights_path}")

    total_chunks = (args.total_games + args.chunk_games - 1) // args.chunk_games
    print("=" * 70)
    print(f"ZQUORIDOR SELF-PLAY WORKER #{args.worker_id}")
    print(f"  Drive Output Directory: {drive_dir}")
    print(f"  Total Games Target:     {args.total_games:,} ({total_chunks} chunks of {args.chunk_games} games)")
    print(f"  Time Control:           {args.time_ms} ms/move (Monte Carlo mode)")
    print(f"  Opening Book:           {positions_path.name}")
    print(f"  NNUE Weights:           {weights_path.name}")
    print("=" * 70)

    start_shard = find_next_shard_index(drive_dir, args.worker_id)
    games_completed = start_shard * args.chunk_games
    print(f"Resuming at shard index {start_shard:04d} ({games_completed:,} games already in Drive).\n")

    for shard_idx in range(start_shard, total_chunks):
        shard_bin = drive_dir / f"c{args.worker_id}_shard_{shard_idx:04d}.bin"
        shard_meta = drive_dir / f"c{args.worker_id}_shard_{shard_idx:04d}.meta"

        # Unique seed per worker and per shard to prevent any collision
        shard_seed = CONFIG["base_seed"] + args.worker_id * 1_000_003 + shard_idx * 7_919

        cmd = [
            str(exe_path),
            "--games", str(args.chunk_games),
            "--chunk-games", str(args.chunk_games),
            "--threads", str(args.threads),
            "--time-ms", str(args.time_ms),
            "--positions", str(positions_path),
            "--nnue-weights", str(weights_path),
            "--seed", str(shard_seed),
            "--out", str(shard_bin),
            "--meta-out", str(shard_meta),
        ]

        if CONFIG["mc_mode"]:
            cmd += [
                "--mc-mode",
                "--mc-obvious-plies", str(CONFIG["mc_obvious_plies"]),
                "--mc-temp-opening", str(CONFIG["mc_temp_opening"]),
                "--mc-temp-decay-plies", str(CONFIG["mc_temp_decay_plies"]),
                "--mc-temp-end", str(CONFIG["mc_temp_end"]),
            ]

        t0 = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Shard {shard_idx:04d}/{total_chunks - 1}...")
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"ERROR: selfplay exited with code {proc.returncode} on shard {shard_idx}", file=sys.stderr)
            break

        if shard_bin.exists() and shard_bin.stat().st_size > 0:
            file_mb = shard_bin.stat().st_size / (1024 * 1024)
            print(f"OK: Shard {shard_idx:04d} completed in {elapsed:.1f}s ({file_mb:.2f} MB saved to Drive).")
        else:
            print(f"WARNING: Shard output {shard_bin} is missing or empty!", file=sys.stderr)

    print("\nWorker execution cycle finished.")


if __name__ == "__main__":
    main()
