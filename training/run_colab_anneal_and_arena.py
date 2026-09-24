#!/usr/bin/env python3
"""Execute 60-epoch annealing runs for Zquoridor candidates and benchmark vs baseline.

Runs:
  1. Arm E: multipath_phase_contact_bucketed (20 epochs, warm-started from Arm C)
  2. Arm A2: multipath_phase anneal (60 epochs, warm-started from Arm A)
  3. Arm B2: multipath_phase anneal (60 epochs, mirror-h, warm-started from Arm B)
  4. Arm C2: multipath_phase_bucketed anneal (60 epochs, mirror-h, warm-started from Arm C)
  5. Arm D2: multipath_phase_deep anneal (60 epochs, mirror-h, warm-started from Arm D)
  6. Arm E2: multipath_phase_contact_bucketed anneal (60 epochs, mirror-h, warm-started from Arm E)

Followed by:
  - Arena matches (50 pairs = 100 games, 200 ms per move) of each candidate vs production baseline.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_benchmark
from tools.external import local_arena
from training.run_experiment import build_candidate, ARCH_CONFIGS

CONFIG = {
    "data": "/content/dataset.npz",
    "runs_dir": "/content/drive/MyDrive/zquoridor_data/runs",
    "baseline_weights": str(ROOT / "data/nnue/nnue_weights_int8.bin"),
    "arena_pairs": 50,
    "move_time_ms": 200,
    "arena_workers": 2,
    "device": "cuda",
    "skip_training": False,
    "skip_arena": False,
}


def ensure_baseline_executable(compiler="g++") -> Path:
    """Build the production baseline executable if missing."""
    exe = ROOT / "bin/local_benchmark/baseline_zquoridor"
    exe.parent.mkdir(parents=True, exist_ok=True)
    if exe.exists():
        return exe
    flags = [
        "-DZQ_NNUE_MULTIPATH_FEATURES=1",
        "-DZQ_NNUE_PHASE_FEATURES=1",
        "-DZQ_NNUE_HIDDEN=512",
        "-DZQ_NNUE_VALUE_BUCKETS=1",
        "-DZQ_NNUE_VALUE_DEPTH=1",
    ]
    cmd = [compiler, "-O3", "-std=c++17", "-march=native", "-pthread", *flags,
           "-I" + str(ROOT / "src"), str(ROOT / "tools/external/zquoridor_uci.cpp"),
           "-o", str(exe)]
    print(f"Building baseline executable: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return exe


def get_training_jobs(config: dict) -> list[dict]:
    data_path = str(Path(config["data"]).resolve())
    base_dir = Path(config["runs_dir"]).resolve()
    device = config["device"]

    jobs = [
        # --- Arm E: First pass of contact + 6 buckets + 2 layers ---
        {
            "name": "Arm E: multipath_phase_contact_bucketed (858 features, 6 buckets, 2 layers, mirror-h, 20ep)",
            "out_dir": str(base_dir / "arm_e_contact_bucketed_20ep"),
            "arch": "multipath_phase_contact_bucketed",
            "epochs": 20,
            "cmd": [
                sys.executable, "training/run_experiment.py",
                "--data", data_path,
                "--out-dir", str(base_dir / "arm_e_contact_bucketed_20ep"),
                "--architecture", "multipath_phase_contact_bucketed",
                "--hidden", "512",
                "--init-from", str(base_dir / "arm_c_candidate_20ep/student.bin"),
                "--init-architecture", "multipath_phase_bucketed",
                "--init-hidden", "512",
                "--mirror-h",
                "--epochs", "20",
                "--batch-size", "256",
                "--lr", "0.0001",
                "--min-lr", "0.000005",
                "--trunk-lr-scale", "0.1",
                "--schedule", "cosine",
                "--warmup-epochs", "4",
                "--patience", "12",
                "--device", device,
                "--seed", "20260924",
                "--no-teaching", "--no-benchmark",
            ]
        },
        # --- Arm A2: 60-epoch annealing pass (Control, no mirror) ---
        {
            "name": "Arm A2: multipath_phase anneal (control, no mirror, 60ep)",
            "out_dir": str(base_dir / "arm_a2_anneal_60ep"),
            "arch": "multipath_phase",
            "epochs": 60,
            "cmd": [
                sys.executable, "training/run_experiment.py",
                "--data", data_path,
                "--out-dir", str(base_dir / "arm_a2_anneal_60ep"),
                "--architecture", "multipath_phase",
                "--hidden", "512",
                "--init-from", str(base_dir / "arm_a_control_20ep/student.bin"),
                "--init-architecture", "multipath_phase",
                "--init-hidden", "512",
                "--no-mirror-h",
                "--epochs", "60",
                "--batch-size", "256",
                "--lr", "0.00001",
                "--min-lr", "0.0000001",
                "--trunk-lr-scale", "0.05",
                "--schedule", "cosine",
                "--warmup-epochs", "1",
                "--patience", "30",
                "--device", device,
                "--seed", "20260925",
                "--no-teaching", "--no-benchmark",
            ]
        },
        # --- Arm B2: 60-epoch annealing pass (Ablation, mirror-h) ---
        {
            "name": "Arm B2: multipath_phase anneal (ablation, mirror-h, 60ep)",
            "out_dir": str(base_dir / "arm_b2_anneal_60ep"),
            "arch": "multipath_phase",
            "epochs": 60,
            "cmd": [
                sys.executable, "training/run_experiment.py",
                "--data", data_path,
                "--out-dir", str(base_dir / "arm_b2_anneal_60ep"),
                "--architecture", "multipath_phase",
                "--hidden", "512",
                "--init-from", str(base_dir / "arm_b_mirror_20ep/student.bin"),
                "--init-architecture", "multipath_phase",
                "--init-hidden", "512",
                "--mirror-h",
                "--epochs", "60",
                "--batch-size", "256",
                "--lr", "0.00001",
                "--min-lr", "0.0000001",
                "--trunk-lr-scale", "0.05",
                "--schedule", "cosine",
                "--warmup-epochs", "1",
                "--patience", "30",
                "--device", device,
                "--seed", "20260926",
                "--no-teaching", "--no-benchmark",
            ]
        },
        # --- Arm C2: 60-epoch annealing pass (Candidate, 6 buckets, 2 layers, mirror-h) ---
        {
            "name": "Arm C2: multipath_phase_bucketed anneal (candidate, 6 buckets, 2 layers, mirror-h, 60ep)",
            "out_dir": str(base_dir / "arm_c2_anneal_60ep"),
            "arch": "multipath_phase_bucketed",
            "epochs": 60,
            "cmd": [
                sys.executable, "training/run_experiment.py",
                "--data", data_path,
                "--out-dir", str(base_dir / "arm_c2_anneal_60ep"),
                "--architecture", "multipath_phase_bucketed",
                "--hidden", "512",
                "--init-from", str(base_dir / "arm_c_candidate_20ep/student.bin"),
                "--init-architecture", "multipath_phase_bucketed",
                "--init-hidden", "512",
                "--mirror-h",
                "--epochs", "60",
                "--batch-size", "256",
                "--lr", "0.00001",
                "--min-lr", "0.0000001",
                "--trunk-lr-scale", "0.05",
                "--schedule", "cosine",
                "--warmup-epochs", "1",
                "--patience", "30",
                "--device", device,
                "--seed", "20260927",
                "--no-teaching", "--no-benchmark",
            ]
        },
        # --- Arm D2: 60-epoch annealing pass (Ablation, 1 bucket, 2 layers, mirror-h) ---
        {
            "name": "Arm D2: multipath_phase_deep anneal (ablation, 1 bucket, 2 layers, mirror-h, 60ep)",
            "out_dir": str(base_dir / "arm_d2_anneal_60ep"),
            "arch": "multipath_phase_deep",
            "epochs": 60,
            "cmd": [
                sys.executable, "training/run_experiment.py",
                "--data", data_path,
                "--out-dir", str(base_dir / "arm_d2_anneal_60ep"),
                "--architecture", "multipath_phase_deep",
                "--hidden", "512",
                "--init-from", str(base_dir / "arm_d_deep_20ep/student.bin"),
                "--init-architecture", "multipath_phase_deep",
                "--init-hidden", "512",
                "--mirror-h",
                "--epochs", "60",
                "--batch-size", "256",
                "--lr", "0.00001",
                "--min-lr", "0.0000001",
                "--trunk-lr-scale", "0.05",
                "--schedule", "cosine",
                "--warmup-epochs", "1",
                "--patience", "30",
                "--device", device,
                "--seed", "20260928",
                "--no-teaching", "--no-benchmark",
            ]
        },
        # --- Arm E2: 60-epoch annealing pass (Contact + 6 buckets + 2 layers) ---
        {
            "name": "Arm E2: multipath_phase_contact_bucketed anneal (mirror-h, 60ep)",
            "out_dir": str(base_dir / "arm_e2_anneal_60ep"),
            "arch": "multipath_phase_contact_bucketed",
            "epochs": 60,
            "cmd": [
                sys.executable, "training/run_experiment.py",
                "--data", data_path,
                "--out-dir", str(base_dir / "arm_e2_anneal_60ep"),
                "--architecture", "multipath_phase_contact_bucketed",
                "--hidden", "512",
                "--init-from", str(base_dir / "arm_e_contact_bucketed_20ep/student.bin"),
                "--init-architecture", "multipath_phase_contact_bucketed",
                "--init-hidden", "512",
                "--mirror-h",
                "--epochs", "60",
                "--batch-size", "256",
                "--lr", "0.00001",
                "--min-lr", "0.0000001",
                "--trunk-lr-scale", "0.05",
                "--schedule", "cosine",
                "--warmup-epochs", "1",
                "--patience", "30",
                "--device", device,
                "--seed", "20260929",
                "--no-teaching", "--no-benchmark",
            ]
        },
    ]
    return jobs


def run_training_pipeline(config: dict):
    jobs = get_training_jobs(config)
    print("=" * 70)
    print(f"STARTING TRAINING PIPELINE: {len(jobs)} RUNS QUEUED")
    print("=" * 70)

    for idx, job in enumerate(jobs, 1):
        out_dir = Path(job["out_dir"])
        report_file = out_dir / "train_report.json"
        if report_file.exists():
            print(f"\n[{idx}/{len(jobs)}] SKIPPING (already complete): {job['name']}")
            continue

        print("\n" + "=" * 70)
        print(f"[{idx}/{len(jobs)}] LAUNCHING: {job['name']}")
        print(f"Output directory: {job['out_dir']}")
        print("=" * 70)
        t0 = time.time()
        res = subprocess.run(job["cmd"], cwd=ROOT)
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(f"\nSUCCESS: {out_dir.name} finished in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        else:
            print(f"\nERROR: {out_dir.name} failed with code {res.returncode}")
            sys.exit(res.returncode)


def run_baseline_arena(config: dict):
    """Run paired 50-pair (100-game) arena matches of each candidate against baseline."""
    base_dir = Path(config["runs_dir"]).resolve()
    baseline_weights = Path(config["baseline_weights"]).resolve()
    baseline_exe = ensure_baseline_executable()
    pairs = int(config["arena_pairs"])
    move_time_ms = int(config["move_time_ms"])
    workers = int(config["arena_workers"])

    candidate_arms = [
        ("Arm A (Control, 20ep)", base_dir / "arm_a_control_20ep", "multipath_phase"),
        ("Arm B (Mirror, 20ep)", base_dir / "arm_b_mirror_20ep", "multipath_phase"),
        ("Arm C (Bucketed, 20ep)", base_dir / "arm_c_candidate_20ep", "multipath_phase_bucketed"),
        ("Arm D (Deep, 20ep)", base_dir / "arm_d_deep_20ep", "multipath_phase_deep"),
        ("Arm E (Contact Bucketed, 20ep)", base_dir / "arm_e_contact_bucketed_20ep", "multipath_phase_contact_bucketed"),
        ("Arm A2 (Control Anneal, 60ep)", base_dir / "arm_a2_anneal_60ep", "multipath_phase"),
        ("Arm B2 (Mirror Anneal, 60ep)", base_dir / "arm_b2_anneal_60ep", "multipath_phase"),
        ("Arm C2 (Bucketed Anneal, 60ep)", base_dir / "arm_c2_anneal_60ep", "multipath_phase_bucketed"),
        ("Arm D2 (Deep Anneal, 60ep)", base_dir / "arm_d2_anneal_60ep", "multipath_phase_deep"),
        ("Arm E2 (Contact Anneal, 60ep)", base_dir / "arm_e2_anneal_60ep", "multipath_phase_contact_bucketed"),
    ]

    openings_path = ROOT / "tools/external/openings_screen_v1.jsonl"
    openings = run_benchmark._read_openings(openings_path, pairs, seed=20260920)

    print("\n" + "=" * 70)
    print(f"STARTING BASELINE ARENA BENCHMARKS: {len(candidate_arms)} MODELS")
    print(f"Settings: {pairs} pairs ({pairs*2} games), {move_time_ms} ms/move, {workers} workers")
    print(f"Baseline executable: {baseline_exe}")
    print(f"Baseline weights:    {baseline_weights}")
    print("=" * 70)

    arena_results = []

    for name, cand_dir, arch in candidate_arms:
        report_path = cand_dir / "train_report.json"
        if not report_path.exists():
            print(f"\nSkipping {name}: training report not found at {cand_dir}")
            continue

        cand_int8 = cand_dir / "student_int8.bin"
        cand_exe = cand_dir / "zquoridor"

        # Ensure candidate binary is compiled
        if not cand_exe.exists() or not cand_int8.exists():
            print(f"Building candidate binary for {name} ({arch})...")
            exp_config = {
                "architecture": arch,
                "hidden": 512,
                "out_dir": str(cand_dir),
            }
            try:
                build_candidate(exp_config)
            except Exception as e:
                print(f"Failed to build {name}: {e}")
                continue

        out_arena = cand_dir / "arena_vs_baseline"
        summary_path = out_arena / "summary.json"
        if summary_path.exists():
            print(f"\nAlready benchmarked: {name}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            arena_results.append((name, summary))
            continue

        print(f"\nPlaying match: {name} vs Production Baseline ({pairs*2} games)...")
        manifest = local_arena.make_manifest(
            {"protocol": "candidate-vs-baseline-fixed-v1", "pairs": pairs,
             "move_time_ms": move_time_ms, "seed": 20260920,
             "openings": [idx for idx, _ in openings]},
            {"candidate": str(cand_exe), "candidate_nnue": str(cand_int8),
             "baseline": str(baseline_exe), "baseline_nnue": str(baseline_weights)},
        )
        games_path, old_rows = local_arena.prepare_resume(out_arena, manifest)
        latest = {(int(r["opening_index"]), int(r["zq_player"])): r for r in old_rows}

        def play(index: int, opening: list[str], side: int) -> dict:
            return local_arena.play_game(
                opponent="baseline", opening_index=index, opening=opening, zq_player=side,
                zq_factory=lambda: local_arena.UciPlayer([str(cand_exe), "--nnue", str(cand_int8)], "candidate"),
                opponent_factory=lambda: local_arena.UciPlayer([str(baseline_exe), "--nnue", str(baseline_weights)], "baseline"),
                zq_budget=move_time_ms, opponent_budget=move_time_ms,
                move_timeout_s=30.0, max_plies=240, run_id=manifest["run_id"],
            )

        pending = [(idx, opening, side) for idx, opening in openings for side in (0, 1)
                   if latest.get((idx, side), {}).get("status") != "ok"]

        t0 = time.time()
        with games_path.open("a", encoding="utf-8", buffering=1) as stream:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(play, *task) for task in pending]
                for f in concurrent.futures.as_completed(futures):
                    row = f.result()
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    latest[int(row["opening_index"]), int(row["zq_player"])] = row

        elapsed = time.time() - t0
        summary = local_arena.summarize_pairs(list(latest.values()), bootstrap=20000, seed=20260920)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        arena_results.append((name, summary))

        score_pct = summary.get("score_percent", summary.get("score", 0.0) * 100)
        print(f"Result for {name}: Score {score_pct:.1f}% ({summary.get('wins', 0)}W / {summary.get('draws', 0)}D / {summary.get('losses', 0)}L) in {elapsed:.1f}s")

    # Print Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL CANDIDATE VS BASELINE ARENA SUMMARY")
    print("=" * 80)
    header = f"{'Candidate Model':<35} | {'Games':<6} | {'Score %':<8} | {'Record (W-D-L)':<16} | {'Elo [95% CI]':<18}"
    print(header)
    print("-" * len(header))
    for name, s in arena_results:
        g = s.get("games", 0)
        score = s.get("score_percent", s.get("score", 0.0) * 100)
        rec = f"{s.get('wins', 0)}-{s.get('draws', 0)}-{s.get('losses', 0)}"
        elo = s.get("elo", 0.0)
        ci = s.get("elo_ci_95", [0.0, 0.0])
        elo_str = f"{elo:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else f"{elo:+.1f}"
        print(f"{name:<35} | {g:<6} | {score:>6.1f}% | {rec:<16} | {elo_str:<18}")
    print("=" * 80)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(value)
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    config = dict(CONFIG, **vars(parser.parse_args(argv)))

    if not config["skip_training"]:
        run_training_pipeline(config)

    if not config["skip_arena"]:
        run_baseline_arena(config)

    print("\nALL TASKS COMPLETED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
