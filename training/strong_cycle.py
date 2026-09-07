#!/usr/bin/env python3
"""
Run one search-driven self-play cycle and gate the candidate by Elo.

This script does not replace train_nnue.py or run_selfplay.py. It gives the
normal training tools a stricter data and promotion protocol:

1. Save the current production weights as the champion.
2. Generate a new self-play generation with MCAB search from ply zero.
3. Build a replay file from complete games that contain wall-poor race states.
4. Train only on the new generation and the focused replay.
5. Test the candidate against the saved champion with identical engine code.
6. Promote the candidate only when the lower 95 percent Elo bound is positive.

The exact pawn-wandering position already has a search-side safeguard in
mcab.hpp. This cycle targets the value-network cause so that later generations
do not depend only on that safeguard.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
DATA_DIR = ROOT / "data"
NNUE_DIR = DATA_DIR / "nnue"

sys.path.insert(0, str(TRAINING_DIR))
from read_selfplay import SAMPLE_DTYPE, game_start_mask  # noqa: E402


DEFAULT_GENERATION = "gen8-search"
DEFAULT_GAMES = 30000
DEFAULT_TIME_MS = 150
DEFAULT_THREADS = 12
DEFAULT_ARENA_GAMES = 1000
DEFAULT_ARENA_TIME_MS = 150
DEFAULT_FOCUS_COPIES = 2
DEFAULT_WL_GAMMA = 0.990
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 100

ELO_RE = re.compile(
    r"Diferenca Elo Engine 1 vs 2\s*:\s*([+-]?\d+(?:\.\d+)?)\s*"
    r"\(Margem\s*[±+\-]?\s*(\d+(?:\.\d+)?)\)"
)


def run(cmd: list[str], log_path: Path | None = None) -> str:
    """Run one command from the repository root and return its full output."""
    printable = subprocess.list2cmdline(cmd)
    print(f"\n$ {printable}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    log_file = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()
    code = proc.wait()
    output = "".join(lines)
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}: {printable}")
    return output


def current_weights() -> tuple[Path, Path]:
    """Return the float and quantized production-weight paths."""
    f32 = NNUE_DIR / "nnue_weights.bin"
    q8 = NNUE_DIR / "nnue_weights_int8.bin"
    missing = [str(p) for p in (f32, q8) if not p.is_file()]
    if missing:
        raise SystemExit(
            "production NNUE weights are missing:\n  - " + "\n  - ".join(missing)
        )
    return f32, q8


def snapshot_champion(generation: str) -> tuple[Path, Path]:
    """Copy the current production weights to an immutable cycle snapshot."""
    src_f32, src_q8 = current_weights()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = NNUE_DIR / "champions" / f"{generation}-{stamp}"
    dst.mkdir(parents=True, exist_ok=False)
    dst_f32 = dst / "nnue_weights.bin"
    dst_q8 = dst / "nnue_weights_int8.bin"
    shutil.copy2(src_f32, dst_f32)
    shutil.copy2(src_q8, dst_q8)
    print(f"champion snapshot: {dst}")
    return dst_f32, dst_q8


def selfplay_shards(generation: str) -> list[Path]:
    """Return the generated shards in deterministic order."""
    folder = DATA_DIR / "selfplay" / generation
    return [Path(p) for p in sorted(glob.glob(str(folder / "selfplay_*.bin")))]


def verify_current_format(paths: list[Path]) -> None:
    """Reject a shard that cannot be the current 32-byte TrainingSample format."""
    if not paths:
        raise SystemExit("no self-play shards were found")
    bad = [
        p
        for p in paths
        if p.stat().st_size == 0 or p.stat().st_size % SAMPLE_DTYPE.itemsize != 0
    ]
    if bad:
        joined = "\n  - ".join(str(p) for p in bad)
        raise SystemExit(
            "the new generation contains a file that is not a valid 32-byte shard:\n"
            f"  - {joined}"
        )


def game_ranges(arr: np.ndarray):
    """Yield complete [start, end) ranges from one current-format shard."""
    starts = np.flatnonzero(game_start_mask(arr))
    if len(starts) == 0:
        return
    if starts[0] != 0:
        raise ValueError("the shard does not start at a detected game boundary")
    ends = np.append(starts[1:], len(arr))
    for start, end in zip(starts, ends):
        yield int(start), int(end)


def risk_mask(game: np.ndarray) -> np.ndarray:
    """Mark positions that match the wall-poor pawn-race failure family."""
    own_walls = game["walls_left_own"].astype(np.int16)
    opp_walls = game["walls_left_opp"].astype(np.int16)
    own_dist = game["own_dist"].astype(np.int16)
    opp_dist = game["opp_dist"].astype(np.int16)

    no_walls = (own_walls == 0) & (opp_walls >= 4)
    asymmetric = (
        (own_walls <= 2)
        & (opp_walls - own_walls >= 4)
        & (own_dist + 2 <= opp_dist)
    )
    return no_walls | asymmetric


def build_focus_replay(
    shards: list[Path],
    output: Path,
    copies: int,
    max_fraction: float,
) -> dict:
    """Copy complete risk games into a second replay source."""
    selected_games: list[np.ndarray] = []
    total_games = 0
    risk_games = 0
    total_positions = 0
    risk_positions = 0

    for path in shards:
        arr = np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")
        for start, end in game_ranges(arr):
            total_games += 1
            game = np.asarray(arr[start:end])
            total_positions += len(game)
            mask = risk_mask(game)
            n_risk = int(mask.sum())
            risk_positions += n_risk
            if n_risk > 0:
                risk_games += 1
                selected_games.append(game.copy())

    if not selected_games:
        raise SystemExit(
            "the new generation has no wall-poor race games. "
            "Increase the self-play game count before training."
        )

    selected_positions = sum(len(g) for g in selected_games)
    max_positions = max(1, int(total_positions * max_fraction))
    if selected_positions > max_positions:
        rng = np.random.default_rng(20260905)
        order = rng.permutation(len(selected_games))
        kept: list[np.ndarray] = []
        count = 0
        for idx in order:
            game = selected_games[int(idx)]
            if kept and count + len(game) > max_positions:
                continue
            kept.append(game)
            count += len(game)
            if count >= max_positions:
                break
        selected_games = kept
        selected_positions = count

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for _ in range(max(1, copies)):
            for game in selected_games:
                game.tofile(f)

    stats = {
        "total_games": total_games,
        "risk_games": risk_games,
        "total_positions": total_positions,
        "risk_positions": risk_positions,
        "focus_games": len(selected_games),
        "focus_positions_per_copy": selected_positions,
        "copies": max(1, copies),
        "output_positions": selected_positions * max(1, copies),
    }
    print(
        "focus replay: "
        f"{stats['focus_games']:,} complete games, "
        f"{stats['output_positions']:,} positions after {stats['copies']} copy/copies"
    )
    return stats


def split_train_validation(shards: list[Path]) -> tuple[list[Path], Path]:
    """Reserve the last shard for validation and keep it out of training."""
    if len(shards) < 2:
        raise SystemExit(
            "at least two self-play shards are required so validation can use "
            "a complete held-out shard"
        )
    return shards[:-1], shards[-1]


def source_json(train_shards: list[Path], focus_path: Path) -> str:
    """Build a source list that contains only current-format, search-driven data."""
    sources = [
        {"path": str(path), "frac": 1.0, "k": 1.0}
        for path in train_shards
    ]
    sources.append({"path": str(focus_path), "frac": 1.0, "k": 1.0})
    return json.dumps(sources, separators=(",", ":"))


def generate_selfplay(args) -> list[Path]:
    """Generate a generation whose played move comes from search from ply zero."""
    out = f"data/selfplay/{args.generation}/selfplay_{{shard:03d}}.bin"
    cmd = [
        sys.executable,
        "tools/selfplay/run_selfplay.py",
        "--games",
        str(args.games),
        "--chunk-games",
        str(args.chunk_games),
        "--time-ms",
        str(args.time_ms),
        "--threads",
        str(args.threads),
        "--mode",
        "montecarlo",
        "--mc-obvious-plies",
        "0",
        "--mc-temp-decay-plies",
        "0",
        "--epsilon-midgame",
        str(args.epsilon_midgame),
        "--out",
        out,
    ]
    run(cmd, DATA_DIR / "logs" / f"{args.generation}-selfplay.log")
    paths = selfplay_shards(args.generation)
    verify_current_format(paths)
    return paths


def train_candidate(
    args,
    champion_f32: Path,
    train_shards: list[Path],
    val_shard: Path,
    focus_path: Path,
) -> tuple[Path, Path]:
    """Train a candidate without overwriting the production weights."""
    candidate_dir = NNUE_DIR / "candidates" / args.generation
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_f32 = candidate_dir / "nnue_weights.bin"
    candidate_q8 = candidate_dir / "nnue_weights_int8.bin"
    ckpt_dir = DATA_DIR / "checkpoints" / f"{args.generation}-candidate"
    plot_dir = DATA_DIR / "plots" / f"{args.generation}-candidate"

    cmd = [
        sys.executable,
        "training/train_nnue.py",
        "--fresh",
        "--init-from",
        str(champion_f32),
        "--data-sources",
        source_json(train_shards, focus_path),
        "--val-data",
        str(val_shard),
        "--out",
        str(candidate_f32),
        "--ckpt-dir",
        str(ckpt_dir),
        "--plot-dir",
        str(plot_dir),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--monitor",
        "val_outcome",
        "--policy-opening-plies",
        "0",
        "--wl-gamma",
        str(args.wl_gamma),
    ]
    run(cmd, DATA_DIR / "logs" / f"{args.generation}-train.log")
    if not candidate_f32.is_file() or not candidate_q8.is_file():
        raise SystemExit("training finished without both candidate weight files")
    return candidate_f32, candidate_q8


def arena_gate(args, candidate_q8: Path, champion_q8: Path) -> tuple[float, float, str]:
    """Play candidate against champion with the same local engine code."""
    cmd = [
        sys.executable,
        "tools/arena/run_arena.py",
        "--ref1",
        "",
        "--ref2",
        "",
        "--games",
        str(args.arena_games),
        "--time",
        str(args.arena_time_ms),
        "--threads",
        str(args.arena_threads),
        "--random-plies",
        str(args.arena_random_plies),
        "--e1-nnue",
        str(candidate_q8),
        "--e2-nnue",
        str(champion_q8),
    ]
    output = run(cmd, DATA_DIR / "logs" / f"{args.generation}-arena.log")
    matches = ELO_RE.findall(output)
    if not matches:
        raise SystemExit("could not parse the final Elo result from run_arena.py")
    elo, margin = map(float, matches[-1])
    return elo, margin, output


def promote(candidate_f32: Path, candidate_q8: Path) -> None:
    """Replace the local production weights after a successful gate."""
    NNUE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_f32, NNUE_DIR / "nnue_weights.bin")
    shutil.copy2(candidate_q8, NNUE_DIR / "nnue_weights_int8.bin")
    print("candidate promoted to data/nnue")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generation", default=DEFAULT_GENERATION)
    p.add_argument("--games", type=int, default=DEFAULT_GAMES)
    p.add_argument("--chunk-games", type=int, default=3000)
    p.add_argument("--time-ms", type=int, default=DEFAULT_TIME_MS)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--epsilon-midgame", type=float, default=0.01)
    p.add_argument("--focus-copies", type=int, default=DEFAULT_FOCUS_COPIES)
    p.add_argument("--focus-max-fraction", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--wl-gamma", type=float, default=DEFAULT_WL_GAMMA)
    p.add_argument("--arena-games", type=int, default=DEFAULT_ARENA_GAMES)
    p.add_argument("--arena-time-ms", type=int, default=DEFAULT_ARENA_TIME_MS)
    p.add_argument("--arena-threads", type=int, default=14)
    p.add_argument("--arena-random-plies", type=int, default=4)
    p.add_argument("--reuse-selfplay", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--promote",
        action="store_true",
        help="copy the candidate to data/nnue only if the lower 95 percent Elo bound is positive",
    )
    return p.parse_args()


def validate_args(args) -> None:
    if args.games <= 0 or args.chunk_games <= 0:
        raise SystemExit("--games and --chunk-games must be positive")
    if args.time_ms <= 0 or args.arena_time_ms <= 0:
        raise SystemExit("time budgets must be positive")
    if not (0.0 <= args.epsilon_midgame <= 1.0):
        raise SystemExit("--epsilon-midgame must be in [0, 1]")
    if not (0.0 < args.focus_max_fraction <= 1.0):
        raise SystemExit("--focus-max-fraction must be in (0, 1]")
    if not (0.0 < args.wl_gamma <= 1.0):
        raise SystemExit("--wl-gamma must be in (0, 1]")


def main() -> int:
    args = parse_args()
    validate_args(args)

    champion_f32, champion_q8 = snapshot_champion(args.generation)

    if args.reuse_selfplay:
        shards = selfplay_shards(args.generation)
        verify_current_format(shards)
    else:
        shards = generate_selfplay(args)

    train_shards, val_shard = split_train_validation(shards)
    focus_path = DATA_DIR / "selfplay" / f"{args.generation}-focus" / "focus_000.bin"
    focus_stats = build_focus_replay(
        train_shards,
        focus_path,
        args.focus_copies,
        args.focus_max_fraction,
    )
    stats_path = focus_path.with_suffix(".json")
    stats_path.write_text(json.dumps(focus_stats, indent=2), encoding="utf-8")

    if args.skip_train:
        print("self-play and focus replay are ready. Training was skipped.")
        return 0

    candidate_f32, candidate_q8 = train_candidate(
        args,
        champion_f32,
        train_shards,
        val_shard,
        focus_path,
    )
    elo, margin, _ = arena_gate(args, candidate_q8, champion_q8)
    lower = elo - margin
    print(
        f"\nGate result: candidate {elo:+.1f} Elo, margin ±{margin:.1f}, "
        f"lower 95 percent bound {lower:+.1f}."
    )

    if lower > 0.0:
        print("PASS: the candidate is statistically stronger than the saved champion.")
        if args.promote:
            promote(candidate_f32, candidate_q8)
        else:
            print("The candidate was not promoted. Run again with --promote after review.")
        return 0

    print("FAIL: keep the champion. The candidate does not clear the promotion gate.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
