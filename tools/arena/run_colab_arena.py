#!/usr/bin/env python3
"""Run candidate arena matches against main, Titanium, or Claustrophobia.

Designed for Google Colab workers or local parallel evaluation.
Uses 200 ms per move, paired colors, and shared openings.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_benchmark
from tools.external import local_arena

# Top-level configuration following repository standards.
CONFIG = {
    "model": "candidate",  # choices: "candidate", "baseline"
    "opponent": "main",    # choices: "main", "titanium", "claustrophobia"
    "pairs": 200,          # 200 pairs = 400 paired games
    "move_time_ms": 200,
    "workers": 2,
    "seed": 20260926,
    "openings": str(ROOT / "tools" / "external" / "openings_600g_300pairs.jsonl"),
    "candidate_weights": str(
        ROOT / "results" / "experiments" / "multipath_contact_bucketed_unified" / "student_int8.bin"
    ),
    "candidate_arch": "multipath_phase_contact_bucketed",
    "baseline_weights": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "baseline_arch": "multipath_phase_bucketed",
    "output_dir": None,    # auto-detected (Drive or local)
    "bootstrap": 20000,
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line interface overriding CONFIG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("candidate", "baseline"),
        default=CONFIG["model"],
        help="Player model to evaluate (candidate, baseline).",
    )
    parser.add_argument(
        "--opponent",
        choices=("main", "titanium", "claustrophobia"),
        default=CONFIG["opponent"],
        help="Select match opponent (main, titanium, claustrophobia).",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=CONFIG["pairs"],
        help="Number of paired openings to play (each played as White and Black).",
    )
    parser.add_argument(
        "--move-time-ms",
        type=int,
        default=CONFIG["move_time_ms"],
        help="Thinking time per move in milliseconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["workers"],
        help="Parallel game workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"],
        help="Random seed for opening selection and bootstrapping.",
    )
    parser.add_argument(
        "--openings",
        type=str,
        default=CONFIG["openings"],
        help="Path to opening book in JSONL format.",
    )
    parser.add_argument(
        "--candidate-weights",
        type=str,
        default=CONFIG["candidate_weights"],
        help="Path to candidate INT8 NNUE weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG["output_dir"],
        help="Output directory for game logs and summary.",
    )
    return parser


def detect_output_dir(opponent: str, override: str | None, model: str = "candidate") -> Path:
    """Resolve destination directory prioritizing Google Drive when present."""
    if override:
        path = Path(override).resolve()
    else:
        drive_dir = Path("/content/drive/MyDrive/zquoridor_data/arena_candidate_eval")
        sub = f"{model}_vs_{opponent}" if model != "candidate" else opponent
        if drive_dir.parent.is_dir():
            path = drive_dir / sub
        else:
            path = ROOT / "results" / "benchmarks" / "colab_eval" / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_executable(name: str, arch: str, compiler: str = "g++") -> Path:
    """Compile a Zquoridor UCI executable configured for the given architecture."""
    suffix = ".exe" if sys.platform == "win32" else ""
    exe = ROOT / "bin" / f"{name}{suffix}"
    if exe.is_file():
        return exe.resolve()

    exe.parent.mkdir(parents=True, exist_ok=True)

    flags = [
        "-DZQ_NNUE_RACE_FEATURES=1",
        "-DZQ_NNUE_MULTIPATH_FEATURES=1",
        "-DZQ_NNUE_PHASE_FEATURES=1",
        "-DZQ_NNUE_HIDDEN=512",
    ]
    if "contact" in arch:
        flags.append("-DZQ_NNUE_CONTACT_FEATURES=1")
    else:
        flags.append("-DZQ_NNUE_CONTACT_FEATURES=0")

    if "bucketed" in arch:
        flags.extend(["-DZQ_NNUE_VALUE_BUCKETS=6", "-DZQ_NNUE_VALUE_DEPTH=2"])
    else:
        flags.extend(["-DZQ_NNUE_VALUE_BUCKETS=1", "-DZQ_NNUE_VALUE_DEPTH=1"])

    source = ROOT / "tools" / "external" / "zquoridor_uci.cpp"
    cmd = [
        compiler,
        "-O3",
        "-std=c++17",
        "-march=native",
        "-pthread",
        *flags,
        "-I" + str(ROOT / "src"),
        str(source),
        "-o",
        str(exe),
    ]
    print(f"Building {name} ({arch}): {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return exe.resolve()


def run_vs_main(config: dict, candidate_exe: Path, out_dir: Path) -> dict:
    """Run candidate vs production baseline paired match."""
    baseline_exe = ensure_executable("baseline_zquoridor", config["baseline_arch"])
    baseline_weights = Path(config["baseline_weights"]).resolve()
    candidate_weights = Path(config["candidate_weights"]).resolve()
    openings_path = Path(config["openings"]).resolve()
    pairs_count = int(config["pairs"])
    seed = int(config["seed"])
    move_time_ms = int(config["move_time_ms"])
    workers = int(config["workers"])

    openings = run_benchmark._read_openings(openings_path, pairs_count, seed)
    print("\n" + "=" * 65, flush=True)
    print(f"MATCH: Candidate vs Main (Production Baseline)", flush=True)
    print(f"Pairs: {len(openings)} ({len(openings)*2} games) | Time: {move_time_ms} ms/move", flush=True)
    print(f"Candidate: {candidate_exe.name} | Weights: {candidate_weights.name}", flush=True)
    print(f"Baseline:  {baseline_exe.name} | Weights: {baseline_weights.name}", flush=True)
    print(f"Output:    {out_dir}", flush=True)
    print("=" * 65 + "\n", flush=True)

    manifest = local_arena.make_manifest(
        {
            "protocol": "candidate-vs-main-fixed-clock",
            "pairs": len(openings),
            "move_time_ms": move_time_ms,
            "seed": seed,
            "openings": [idx for idx, _ in openings],
        },
        {
            "candidate_executable": candidate_exe,
            "candidate_nnue": candidate_weights,
            "baseline_executable": baseline_exe,
            "baseline_nnue": baseline_weights,
            "referee": ROOT / "tools" / "external" / "local_arena.py",
        },
    )

    games_path, prior = local_arena.prepare_resume(out_dir, manifest)
    latest = {(int(r["opening_index"]), int(r["zq_player"])): r for r in prior}

    tasks = []
    for opening_idx, opening_moves in openings:
        for zq_player in (0, 1):
            key = (opening_idx, zq_player)
            existing = latest.get(key)
            if existing and existing.get("status") == "ok":
                continue
            tasks.append((opening_idx, opening_moves, zq_player))

    total_games = len(openings) * 2
    done_games = len(latest)
    print(f"Prior finished games: {done_games}/{total_games}. Pending: {len(tasks)}.\n", flush=True)

    def play_one(task: tuple[int, list[str], int]) -> dict:
        idx, moves, side = task
        cand_cmd = [str(candidate_exe), "--nnue", str(candidate_weights)]
        base_cmd = [str(baseline_exe), "--nnue", str(baseline_weights)]
        return local_arena.play_game(
            opponent="main",
            opening_index=idx,
            opening=moves,
            zq_player=side,
            zq_factory=lambda: local_arena.UciPlayer(cand_cmd, "candidate"),
            opponent_factory=lambda: local_arena.UciPlayer(base_cmd, "main"),
            zq_budget=move_time_ms,
            opponent_budget=move_time_ms,
            move_timeout_s=30.0,
            max_plies=240,
            run_id=manifest["run_id"],
        )

    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(play_one, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[int(row["opening_index"]), int(row["zq_player"])] = row
                done_games += 1

                # Format live stats
                side_str = "White" if row.get("zq_player") == 0 else "Black"
                res = float(row.get("result", 0.0))
                status_str = "WIN" if res == 1.0 else ("DRAW" if res == 0.5 else "LOSS")
                cand_score = sum(
                    float(r.get("result", 0.0))
                    for r in latest.values()
                    if r.get("status") == "ok"
                )
                print(
                    f"[{done_games}/{total_games}] Opening {int(row.get('opening_index', 0)):3d} ({side_str:5s}): "
                    f"Candidate {status_str:4s} ({row.get('plies', 0):2d} plies) | "
                    f"Candidate: {cand_score:.1f}/{done_games} ({cand_score/done_games*100:5.1f}%)",
                    flush=True,
                )

    all_rows = list(latest.values())
    summary = local_arena.summarize_pairs(
        all_rows, bootstrap=int(config["bootstrap"]), seed=seed
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_vs_external(
    config: dict,
    player_name: str,
    player_exe: Path,
    player_weights: Path,
    out_dir: Path,
    bot_name: str,
) -> dict:
    """Run player (candidate or baseline) vs external reference bot (Titanium or Claustrophobia)."""
    openings_path = Path(config["openings"]).resolve()

    bench_config = dict(
        run_benchmark.CONFIG,
        opponents=[bot_name],
        pairs=int(config["pairs"]),
        workers=int(config["workers"]),
        seed=int(config["seed"]),
        openings=str(openings_path),
        output=str(out_dir),
        resume=True,
        auto_setup=True,
        zq_executable=str(player_exe),
        nnue=str(player_weights),
        zq_move_time_ms=int(config["move_time_ms"]),
        titanium_move_time_ms=int(config["move_time_ms"]),
        claustrophobia_move_time_ms=int(config["move_time_ms"]),
        claustrophobia_device="cpu",
        bootstrap=int(config["bootstrap"]),
    )

    print("\n" + "=" * 65, flush=True)
    print(f"MATCH: {player_name.capitalize()} vs {bot_name.capitalize()}", flush=True)
    print(f"Pairs: {config['pairs']} ({config['pairs']*2} games) | Time: {config['move_time_ms']} ms/move", flush=True)
    print(f"Engine:    {player_exe.name} | Weights: {player_weights.name}", flush=True)
    print(f"Output:    {out_dir}", flush=True)
    print("=" * 65 + "\n", flush=True)

    report = run_benchmark.run(bench_config)
    summary = report["summaries"].get(bot_name, {})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = dict(CONFIG)
    for key, val in vars(args).items():
        if val is not None:
            config[key] = val

    model_name = config.get("model", "candidate").lower()
    out_dir = detect_output_dir(config["opponent"], config["output_dir"], model_name)

    # 1. Compile and select player executable and weights
    if model_name == "baseline":
        player_label = "Baseline"
        player_exe = ensure_executable("baseline_zquoridor", config["baseline_arch"])
        player_weights = Path(config["baseline_weights"]).resolve()
    else:
        player_label = "Candidate"
        player_exe = ensure_executable("candidate_zquoridor", config["candidate_arch"])
        player_weights = Path(config["candidate_weights"]).resolve()

    # 2. Execute match based on chosen opponent
    t0 = time.time()
    opponent = config["opponent"]
    if opponent == "main":
        if model_name == "baseline":
            raise ValueError("Cannot run baseline vs main (baseline vs baseline). Use --model candidate.")
        summary = run_vs_main(config, player_exe, out_dir)
    elif opponent in ("titanium", "claustrophobia"):
        summary = run_vs_external(config, player_label, player_exe, player_weights, out_dir, opponent)
    else:
        raise ValueError(f"Unsupported opponent: {opponent}")

    elapsed = time.time() - t0

    # 3. Print final results banner
    score = summary.get("score_pct", 0.0)
    elo = summary.get("elo", 0.0)
    ci = summary.get("elo_ci_95", [0.0, 0.0])
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    draws = summary.get("draws", 0)
    total = summary.get("games", 0)

    print("\n" + "=" * 65, flush=True)
    print(f"FINAL RESULT: {player_label} vs {opponent.upper()}", flush=True)
    print(f"Games: {total} | Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Score: {score:.2f}% | Elo: {elo:+.1f} [95% CI: {ci[0]:+.1f} to {ci[1]:+.1f}]", flush=True)
    print(f"Record ({player_label}): {wins} Wins, {losses} Losses, {draws} Draws", flush=True)
    print(f"Results saved to: {out_dir}", flush=True)
    print("=" * 65 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
