#!/usr/bin/env python3
"""Train the contact-bucketed experimental candidate with high VRAM batch size and auto-resuming supervisor."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

import torch
from training import run_experiment

CONFIG: Dict[str, Any] = {
    "data": "results/experiments/production-central-finetune-20261002/mixed_dataset",
    "out_dir": "results/experiments/contact-bucketed512-central-20261003",
    "architecture": "multipath_phase_contact_bucketed",
    "hidden": 512,
    "init_from": "data/nnue/nnue_weights.bin",
    "init_architecture": "multipath_phase_bucketed",
    "init_hidden": 512,
    "from_scratch": False,
    "qat": True,
    "mirror_h": True,
    "epochs": 120,
    "batch_size": 131072,
    "lr": 5e-5,
    "min_lr": 1e-7,
    "trunk_lr_scale": 0.05,
    "warmup_epochs": 2,
    "schedule": "cosine",
    "weight_decay": 1e-5,
    "weight_decay_schedule": "constant",
    "min_weight_decay": 1e-5,
    "weight_boosts": [],
    "max_sample_weight": 30.0,
    "policy_weight": 1.0,
    "value_weight": 1.0,
    "train_scope": "full",
    "grad_clip": 1.0,
    "patience": 0,
    "seed": 20261003,
    "device": "cuda",
    "cpu_threads": 2,
    "resume": True,
    "checkpoint_every": 10,
    "teaching": False,
    "teaching_args": [],
    "build": True,
    "benchmark": False,
    "benchmark_args": [],
    "dry_run": False,
}


def parse_config(argv=None) -> tuple[dict, list[str], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", default=False, help="Run as worker process.")
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif isinstance(value, list):
            parser.add_argument(flag, type=json.loads, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)

    parsed, unknown = parser.parse_known_args(argv)
    parsed_dict = vars(parsed)
    is_worker = parsed_dict.pop("worker", False)

    config = copy.deepcopy(CONFIG)
    config.update(parsed_dict)

    worker_forward_args: List[str] = []
    if argv is not None:
        for a in argv:
            if a != "--worker":
                worker_forward_args.append(a)
    else:
        for a in sys.argv[1:]:
            if a != "--worker":
                worker_forward_args.append(a)

    return config, worker_forward_args, is_worker


def get_current_epoch(out_dir: Path) -> int:
    checkpoint = out_dir / "resume.pt"
    if not checkpoint.exists():
        return 0
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(state.get("epoch", 0))
    except Exception:
        return 0


def run_worker(config: dict) -> dict:
    return run_experiment.train(config)


def run_supervisor(config: dict, worker_forward_args: list[str]) -> dict:
    out_dir = Path(config["out_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    status_file = out_dir / "run_status.json"
    status_file.write_text(json.dumps({
        "status": "training_configured",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
    }, indent=2), encoding="utf-8")

    if config["dry_run"]:
        return dict(status="dry_run_complete", config=config)

    target_epochs = int(config["epochs"])
    script_path = str(Path(__file__).resolve())

    consecutive_failures = 0
    last_epoch = get_current_epoch(out_dir)

    print(f"[supervisor] Starting training supervision: target {target_epochs} epochs (current: {last_epoch})", flush=True)

    while True:
        current_epoch = get_current_epoch(out_dir)
        if current_epoch >= target_epochs:
            print(f"[supervisor] All {target_epochs} epochs reached (current: {current_epoch}).", flush=True)
            break

        status_file.write_text(json.dumps({
            "status": "training_in_progress",
            "architecture": config["architecture"],
            "batch_size": config["batch_size"],
            "current_epoch": current_epoch,
            "target_epochs": target_epochs,
        }, indent=2), encoding="utf-8")

        print(f"\n[supervisor] Spawning worker process from epoch {current_epoch}/{target_epochs}...", flush=True)
        cmd = [sys.executable, "-u", script_path, "--worker", *worker_forward_args]
        proc = subprocess.run(cmd, cwd=ROOT)

        new_epoch = get_current_epoch(out_dir)
        if proc.returncode == 0:
            print(f"[supervisor] Worker exited cleanly (returncode 0, epoch: {new_epoch}).", flush=True)
            if new_epoch >= target_epochs:
                break
        else:
            print(f"[supervisor] Worker exited with code {proc.returncode} (epoch: {new_epoch}).", flush=True)
            if new_epoch > last_epoch:
                consecutive_failures = 0
                last_epoch = new_epoch
                print(f"[supervisor] Checkpoint saved successfully. Advanced to epoch {new_epoch}. Cooling down 5s...", flush=True)
            else:
                consecutive_failures += 1
                wait_time = min(30, 5 * consecutive_failures)
                print(f"[supervisor] No epoch progress (consecutive failures: {consecutive_failures}). Cooling down {wait_time}s...", flush=True)
                if consecutive_failures >= 12:
                    raise RuntimeError(f"Training worker failed {consecutive_failures} consecutive times without advancing past epoch {current_epoch}.")
            time.sleep(wait_time if new_epoch <= last_epoch else 5)

    report_file = out_dir / "train_report.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}

    exe = None
    if config["build"]:
        print("\n[supervisor] Building candidate executable...", flush=True)
        exe = run_experiment.build_candidate(config)

    status_file.write_text(json.dumps({
        "status": "training_complete",
        "epochs_completed": get_current_epoch(out_dir),
        "best_epoch": report.get("best_epoch"),
        "best_val_loss": report.get("best_val_loss"),
        "executable": str(exe) if exe else None,
        "weights": str(out_dir / "student_int8.bin"),
    }, indent=2), encoding="utf-8")

    return report


def main(argv=None):
    config, forward_args, is_worker = parse_config(argv)
    if is_worker:
        report = run_worker(config)
        return 0

    report = run_supervisor(config, forward_args)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    main()
