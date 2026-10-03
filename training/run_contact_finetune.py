#!/usr/bin/env python3
"""Train the contact-bucketed experimental candidate with high VRAM batch size."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from training import run_experiment

CONFIG = {
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


def parse_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif isinstance(value, list):
            parser.add_argument(flag, type=json.loads, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def run(config: dict) -> dict:
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

    status_file.write_text(json.dumps({
        "status": "training_in_progress",
        "architecture": config["architecture"],
        "batch_size": config["batch_size"],
    }, indent=2), encoding="utf-8")

    report = run_experiment.train(config)

    exe = None
    if config["build"]:
        exe = run_experiment.build_candidate(config)

    status_file.write_text(json.dumps({
        "status": "training_complete",
        "best_epoch": report.get("best_epoch"),
        "best_val_loss": report.get("best_val_loss"),
        "executable": str(exe) if exe else None,
    }, indent=2), encoding="utf-8")

    return report


def main(argv=None):
    config = parse_config(argv)
    report = run(config)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    main()
