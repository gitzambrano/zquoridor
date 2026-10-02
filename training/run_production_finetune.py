#!/usr/bin/env python3
"""Prepare all accepted new data and fine-tune the production architecture."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import run_experimental_anneal as anneal
from training.prepare_stored_replay_all import prepare

CONFIG = {
    "out_dir": "results/experiments/production-central-finetune-20261002",
    "raw_sources": [
        "data/selfplay/central-weakness-rollouts-16m/frozen_partial_manifest.json",
        *[f"data/selfplay/colab{worker}/manifest.json" for worker in (3, 4, 5)],
    ],
    "historical": "data/teaching/multipath_unified_clean_15m/dataset.npz",
    "epochs": 120,
    "lr": 1e-5,
    "min_lr": 1e-7,
    "trunk_lr_scale": 0.05,
    "batch_size": 1024,
    "checkpoint_every": 10,
    "seed": 20261002,
    "device": "cuda",
    "cpu_threads": 2,
    "chunk_size": 8192,
    "evaluate": False,
    "dry_run": True,
}


def training_plan(config):
    """Use the existing trainer for one champion-initialized candidate."""
    result = copy.deepcopy(anneal.CONFIG)
    output = anneal.path(config["out_dir"])
    result.update({key: config[key] for key in (
        "out_dir", "epochs", "lr", "min_lr", "trunk_lr_scale", "batch_size",
        "checkpoint_every", "seed", "device", "cpu_threads", "dry_run")})
    result["networks"] = [copy.deepcopy(anneal.NETWORKS[0])]
    result["data_sources"] = [
        dict(name="frozen_champion", path=config["historical"], fraction=0.25,
             weight_cap=30.0, historical=True),
        dict(name="new_all", path=str(output / "new_replay/dataset"), fraction=0.75,
             weight_cap=None, historical=False),
    ]
    return result


def run(config):
    recipe = training_plan(config)
    anneal.validate_config(recipe)
    report = anneal.run(dict(recipe, dry_run=True))
    report.update(raw_sources=config["raw_sources"],
                  new_deduplication="neutral mean across all accepted local and Colab records",
                  evaluation=dict(enabled=config["evaluate"], games_per_opponent=400, pairs_per_opponent=200,
                                  opponents=["claustrophobia", "frozen_main"],
                                  move_time_ms=200, workers=1),
                  local_selfplay="stopped")
    if config["dry_run"]:
        return report
    output = anneal.path(config["out_dir"])
    anneal.write_json(output / "run_status.json", dict(status="preparing_new_replay", plan=report))
    raw = [anneal.path(source) for source in config["raw_sources"]]
    replay = prepare(raw, output / "new_replay/dataset", seed=config["seed"],
                     chunk_size=config["chunk_size"])
    anneal.write_json(output / "run_status.json", dict(
        status="mixing_and_training", raw_records=replay["raw_records"],
        eligible_records=replay["accepted_records"], new_unique_states=replay["samples"]))
    result = anneal.run(recipe)
    if config["evaluate"]:
        from tools import run_central_finetune_evaluation as arena
        candidate = output / "production_bucketed512/train/student_int8.bin"
        anneal.write_json(output / "run_status.json", dict(status="evaluating", training=result))
        evaluation_config = arena.resolve_config([
            "--candidate-name", output.name, "--candidate-nnue", str(candidate)])
        evaluation = arena.run(evaluation_config)
        anneal.write_json(output / "run_status.json", dict(
            status="evaluation_complete_strength_review_pending", training=result, evaluation=evaluation))
    else:
        anneal.write_json(output / "run_status.json", dict(
            status="training_complete_evaluation_pending", training=result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=json.loads if isinstance(value, list) else type(value),
                                default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    print(json.dumps(run(config), indent=2), flush=True)


if __name__ == "__main__":
    main()
