#!/usr/bin/env python3
"""Create a new NNUE experiment workspace.

The script creates ``experimental/experiments/<name>/`` with two files:

* ``config.json`` holds the training overrides for ``training/run_experiment.py``.
* ``README.md`` holds the hypothesis, the commands, and an empty results table.

Every ``config.json`` key must exist in ``run_experiment.CONFIG``. The script
reads the production version and architecture from ``data/nnue/``. It does not
train, build, or run games. The top-level ``CONFIG`` supplies defaults and CLI
options override them.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "name": "pilot_experiment",
    "hypothesis": "Describe the expected change against the production network.",
    "baseline_model": "",
    "force": False,
    # Training overrides for training/run_experiment.py. An empty string for an
    # init_* key selects the value from the production architecture file.
    "data": "data/teaching/multipath_unified_clean_15m/dataset.npz",
    "architecture": "multipath_phase_contact_bucketed",
    "hidden": 512,
    "init_from": "data/nnue/nnue_weights.bin",
    "init_architecture": "",
    "init_hidden": 0,
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
}

WORKSPACE_KEYS = ("name", "hypothesis", "baseline_model", "force")
EXPERIMENTS_DIR = Path("experimental") / "experiments"
RUNNER = ROOT / "training" / "run_experiment.py"
CANDIDATE_SUITE = "tools/run_full_candidate_suite.py"


def runner_keys(runner: Path = RUNNER) -> set[str]:
    """Return the ``CONFIG`` keys of ``run_experiment.py`` without importing it."""
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONFIG" for t in node.targets):
            return {key.value for key in node.value.keys}
    raise ValueError(f"{runner} has no top-level CONFIG dictionary")


def production_identity(root: Path) -> dict:
    """Return the production version, architecture, and hidden width."""
    nnue = root / "data" / "nnue"
    identity = {"version": "unknown", "architecture": "", "hidden": 0}
    release = nnue / "release.json"
    if release.is_file():
        identity["version"] = json.loads(release.read_text(encoding="utf-8")).get("version", "unknown")
    arch = nnue / "nnue_weights.architecture.json"
    if arch.is_file():
        values = json.loads(arch.read_text(encoding="utf-8"))
        identity["architecture"] = values.get("architecture", "")
        identity["hidden"] = int(values.get("hidden", 0))
    return identity


def training_overrides(config: dict, name: str, production: dict) -> dict:
    """Return the ``run_experiment.py`` overrides for one workspace."""
    values = {key: value for key, value in config.items() if key not in WORKSPACE_KEYS}
    values["out_dir"] = (EXPERIMENTS_DIR / name).as_posix()
    if not values["init_architecture"]:
        values["init_architecture"] = production["architecture"] or values["architecture"]
    if not values["init_hidden"]:
        values["init_hidden"] = production["hidden"] or values["hidden"]
    return values


def build_readme(name: str, overrides: dict, hypothesis: str, baseline: str, date_str: str) -> str:
    """Return the README text for one workspace."""
    folder = (EXPERIMENTS_DIR / name).as_posix()
    qat = "Enabled" if overrides["qat"] else "Disabled"
    return f"""# Experiment: {name}

## 1. Hypothesis

- **Created**: {date_str}
- **Baseline**: {baseline}
- **Hypothesis**: {hypothesis}

## 2. Training specification

`config.json` holds the complete set of overrides. This table repeats the main values.

| Setting | Value |
| :--- | :--- |
| Architecture | `{overrides["architecture"]}` |
| Hidden width | {overrides["hidden"]} |
| Dataset | `{overrides["data"]}` |
| Initial weights | `{overrides["init_from"]}` |
| Epochs | {overrides["epochs"]} |
| Quantization-aware training | {qat} |
| Learning rate schedule | `{overrides["schedule"]}`, {overrides["warmup_epochs"]} warmup epochs |

## 3. Promotion criteria

A candidate network must outperform the production network in every category:

1. Head-to-head against Main, with Normal and Center Rush opening books.
2. Against Claustrophobia, with both opening books.
3. Against Titanium, with both opening books.

A deficit in one category disqualifies the candidate.

## 4. Commands

Train the network and build the native candidate:

```bash
python training/run_experiment.py --config {folder}/config.json
```

Run the external battery against the candidate:

```bash
python {CANDIDATE_SUITE} --candidate-exe {folder}/zquoridor.exe --candidate-nnue {folder}/student_int8.bin
```

On Linux, the executable name is `zquoridor` without the `.exe` suffix.

## 5. Results

| Metric | Target | Observed | Status |
| :--- | :--- | :--- | :--- |
| Best validation loss | Baseline parity | Pending | Pending |
| Against Main (Normal) | Above 50.0% | Pending | Pending |
| Against Main (Center Rush) | Above 50.0% | Pending | Pending |
| Against Claustrophobia (Normal) | Above baseline | Pending | Pending |
| Against Claustrophobia (Center Rush) | Above baseline | Pending | Pending |
| Against Titanium (Normal) | Above baseline | Pending | Pending |
| Against Titanium (Center Rush) | Above baseline | Pending | Pending |
| Promotion decision | Pass in every category | Pending | Pending |

Record the final decision and measurements in `docs/plan.md`.
Add one row for this workspace to `experimental/experiments/README.md`.
"""


def scaffold_experiment(config: dict) -> Path:
    """Create the workspace directory, ``config.json``, and ``README.md``."""
    name = config["name"].strip()
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"experiment name must be one directory name: {config['name']!r}")
    target_dir = ROOT / EXPERIMENTS_DIR / name
    if target_dir.exists() and not config["force"]:
        raise FileExistsError(f"experiment directory exists: {target_dir}. Use --force to overwrite.")

    production = production_identity(ROOT)
    overrides = training_overrides(config, name, production)
    unknown = sorted(set(overrides) - runner_keys())
    if unknown:
        raise ValueError(f"run_experiment.py CONFIG has no keys {unknown}")
    baseline = config["baseline_model"] or (
        f"Production {production['version']} (`data/nnue/nnue_weights_int8.bin`)")

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.json").write_text(
        json.dumps({"config": overrides}, indent=2) + "\n", encoding="utf-8")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (target_dir / "README.md").write_text(
        build_readme(name, overrides, config["hypothesis"], baseline, date_str), encoding="utf-8")
    return target_dir


def parse_config(argv=None) -> dict:
    """Return ``CONFIG`` with the CLI overrides applied."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = dict(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def main(argv=None) -> int:
    created = scaffold_experiment(parse_config(argv))
    print(f"Experiment directory: {created}")
    print(f"Training overrides: {created / 'config.json'}")
    print(f"Documentation: {created / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
