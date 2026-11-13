#!/usr/bin/env python3
"""Scaffold a new NNUE experiment workspace.

Creates a dedicated directory under experimental/experiments/<name>/
populated with a standardized config.json and structured README.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "name": "pilot_experiment",
    "architecture": "multipath_phase_contact_bucketed",
    "hidden": 512,
    "data": "data/teaching/multipath_unified_clean_15m/dataset.npz",
    "init_from": "data/nnue/nnue_weights.bin",
    "init_architecture": "multipath_phase_bucketed",
    "init_hidden": 512,
    "from_scratch": False,
    "qat": True,
    "mirror_h": True,
    "epochs": 60,
    "batch_size": 1024,
    "lr": 0.0001,
    "min_lr": 0.000005,
    "trunk_lr_scale": 0.1,
    "warmup_epochs": 4,
    "schedule": "cosine",
    "weight_decay": 0.00001,
    "weight_decay_schedule": "constant",
    "min_weight_decay": 0.00001,
    "device": "cuda",
    "cpu_threads": 2,
    "checkpoint_every": 10,
    "hypothesis": "Evaluate architecture performance against the production champion baseline.",
    "baseline_model": "V3.01 Production Tri-Model Soup",
    "force": False,
}


def build_readme(
    name: str,
    arch: str,
    hidden: int,
    data: str,
    epochs: int,
    hypothesis: str,
    baseline: str,
    date_str: str,
) -> str:
    """Format a standardized experiment documentation file."""
    return f"""# Experiment: {name}

## 1. Overview and Hypothesis

- **Created**: {date_str}
- **Baseline model**: {baseline}
- **Primary hypothesis**: {hypothesis}

## 2. Architecture and Training Specification

| Component | Setting |
| :--- | :--- |
| **Architecture** | `{arch}` |
| **Hidden width** | {hidden} |
| **Dataset** | `{data}` |
| **Target epochs** | {epochs} |
| **Quantization aware training (QAT)** | Enabled (`true`) |
| **Learning rate schedule** | Cosine decay with warmup |

## 3. Strict Promotion Criteria

A candidate network must strictly outperform the previous production network
across every evaluation dimension:

1. **Head-to-head vs Main**: Win rate > 50.0% across Normal and Center Rush opening books.
2. **vs Claustrophobia**: Strict outperformance across both opening suites.
3. **vs Titanium**: Strict outperformance across both opening suites.
4. **Underperformance condition**: Any single category deficit disqualifies promotion.

## 4. Execution Commands

To start training with the generated configuration:

```bash
python training/run_experiment.py --config experimental/experiments/{name}/config.json
```

To run the complete benchmark battery after training:

```bash
python tools/run_full_candidate_suite.py --candidate-executable bin/candidate/zquoridor_uci.exe --candidate-nnue experimental/experiments/{name}/student_int8.bin
```

## 5. Results and Verification Log

| Metric | Target | Observed | Status |
| :--- | :--- | :--- | :--- |
| **Best validation loss** | Baseline parity | Pending | Pending |
| **vs Main (Normal)** | > 50.0% | Pending | Pending |
| **vs Main (Center Rush)** | > 50.0% | Pending | Pending |
| **vs Claustrophobia (Normal)** | > Baseline | Pending | Pending |
| **vs Claustrophobia (Center Rush)**| > Baseline | Pending | Pending |
| **vs Titanium (Normal)** | > Baseline | Pending | Pending |
| **vs Titanium (Center Rush)** | > Baseline | Pending | Pending |
| **Promotion decision** | Strict unanimous pass | Pending | Pending |
"""


def scaffold_experiment(config: dict) -> Path:
    """Create directory structure, config.json, and README.md."""
    name = config["name"].strip()
    if not name:
        raise ValueError("Experiment name cannot be empty.")

    target_dir = ROOT / "experimental" / "experiments" / name
    if target_dir.exists() and not config["force"]:
        raise FileExistsError(
            f"Target experiment directory already exists: {target_dir}. Use --force to overwrite."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    # Prepare config.json for run_experiment.py
    run_config = {
        "config": {
            "data": config["data"],
            "out_dir": f"experimental/experiments/{name}",
            "architecture": config["architecture"],
            "hidden": config["hidden"],
            "init_from": config["init_from"],
            "init_architecture": config["init_architecture"],
            "init_hidden": config["init_hidden"],
            "from_scratch": config["from_scratch"],
            "qat": config["qat"],
            "mirror_h": config["mirror_h"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "lr": config["lr"],
            "min_lr": config["min_lr"],
            "trunk_lr_scale": config["trunk_lr_scale"],
            "warmup_epochs": config["warmup_epochs"],
            "schedule": config["schedule"],
            "weight_decay": config["weight_decay"],
            "weight_decay_schedule": config["weight_decay_schedule"],
            "min_weight_decay": config["min_weight_decay"],
            "weight_boosts": [],
            "max_sample_weight": 30.0,
            "policy_weight": 1.0,
            "value_weight": 1.0,
            "train_scope": "full",
            "grad_clip": 1.0,
            "patience": 0,
            "device": config["device"],
            "cpu_threads": config["cpu_threads"],
            "checkpoint_every": config["checkpoint_every"],
        }
    }

    config_path = target_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    readme_content = build_readme(
        name=name,
        arch=config["architecture"],
        hidden=config["hidden"],
        data=config["data"],
        epochs=config["epochs"],
        hypothesis=config["hypothesis"],
        baseline=config["baseline_model"],
        date_str=date_str,
    )

    readme_path = target_dir / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme_content)

    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default=CONFIG["name"], help="Experiment identifier name.")
    parser.add_argument("--architecture", type=str, default=CONFIG["architecture"], help="NNUE architecture variant.")
    parser.add_argument("--hidden", type=int, default=CONFIG["hidden"], help="Hidden layer unit width.")
    parser.add_argument("--data", type=str, default=CONFIG["data"], help="Path to training dataset.")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Training batch size.")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Base learning rate.")
    parser.add_argument("--hypothesis", type=str, default=CONFIG["hypothesis"], help="Hypothesis summary.")
    parser.add_argument("--baseline", type=str, default=CONFIG["baseline_model"], help="Baseline model for comparison.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing experiment directory.")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["name"] = args.name
    cfg["architecture"] = args.architecture
    cfg["hidden"] = args.hidden
    cfg["data"] = args.data
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["lr"] = args.lr
    cfg["hypothesis"] = args.hypothesis
    cfg["baseline_model"] = args.baseline
    cfg["force"] = args.force

    created = scaffold_experiment(cfg)
    print(f"Scaffolded experiment directory: {created}")
    print(f"Configuration file: {created / 'config.json'}")
    print(f"Documentation template: {created / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
