#!/usr/bin/env python3
"""Prepare champion-initialized candidates and train a shared annealing matrix."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Shared feature encoders retain direct imports from the training directory.
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

NETWORKS = [
    {"name": "production_bucketed512", "architecture": "multipath_phase_bucketed", "hidden": 512},
    {"name": "contact_bucketed512", "architecture": "multipath_phase_contact_bucketed", "hidden": 512},
    {"name": "contact_bucketed768", "architecture": "multipath_phase_contact_bucketed", "hidden": 768},
    {"name": "contact_bucketed1024", "architecture": "multipath_phase_contact_bucketed", "hidden": 1024},
]
DATA_SOURCES = [
    {"name": "frozen_champion", "path": "data/teaching/multipath_unified_clean_15m/dataset.npz",
     "fraction": 0.25, "weight_cap": 30.0, "historical": True},
    {"name": "central_local", "path": "data/selfplay/central-weakness-rollouts-16m/replay-all-50-50/dataset",
     "raw_source": "data/selfplay/central-weakness-rollouts-16m",
     "fraction": None, "pool": "new", "pool_fraction": 0.75, "weight_cap": None, "historical": False},
    *[{"name": f"colab{worker}", "path": f"data/teaching/colab{worker}-stored-50-50/dataset",
       "raw_source": f"data/selfplay/colab{worker}",
       "fraction": None, "pool": "new", "pool_fraction": 0.75,
       "weight_cap": None, "historical": False} for worker in (3, 4, 5)],
]

# Edit this block for normal use. CLI options override these values.
CONFIG = {
    "out_dir": "results/experiments/central-multinetwork-anneal120",
    "champion": "results/experiments/multipath_unified_champion/student.bin",
    "production_weights": "data/nnue/nnue_weights.bin",
    "networks": NETWORKS,
    "data_sources": DATA_SOURCES,
    "epochs": 120,
    "lr": 1e-5,
    "min_lr": 1e-7,
    "trunk_lr_scale": 0.05,
    "warmup_epochs": 0,
    "weight_decay": 1e-5,
    "policy_weight": 1.0,
    "value_weight": 1.0,
    "batch_size": 1024,
    "checkpoint_every": 10,
    "seed": 20261001,
    "device": "cuda",
    "cpu_threads": 2,
    "build": True,
    "resume": True,
    "dry_run": True,
    "initialize_only": False,
    "prepare_only": False,
    "prepare_replays": False,
}


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(file, payload):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(file)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=json.loads if isinstance(default, list) else type(default),
                                default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def validate_config(config):
    import math
    from training.student_model import FEATURES
    if not config["networks"] or not config["data_sources"]:
        raise ValueError("networks and data_sources must not be empty")
    names = [entry["name"] for entry in config["networks"]]
    source_names = [entry["name"] for entry in config["data_sources"]]
    if len(set(names)) != len(names) or len(set(source_names)) != len(source_names):
        raise ValueError("network and source names must be unique")
    for name in names + source_names:
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name):
            raise ValueError("names must contain lowercase letters, digits, underscores, or hyphens")
    for entry in config["networks"]:
        if entry["architecture"] not in FEATURES or entry["hidden"] not in (512, 768, 1024):
            raise ValueError("unsupported experimental architecture or hidden width")
        if FEATURES[entry["architecture"]] < 504:
            raise ValueError("a champion warm start cannot shrink the feature set")
    pools = {}
    fractions = []
    for entry in config["data_sources"]:
        if entry.get("fraction") is not None:
            fractions.append(float(entry["fraction"]))
        else:
            pool, fraction = entry["pool"], float(entry["pool_fraction"])
            if pool in pools and pools[pool] != fraction:
                raise ValueError("pool fractions must agree across their sources")
            pools[pool] = fraction
    fractions.extend(pools.values())
    if any(not math.isfinite(value) or value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("source fractions must be positive and sum to one")
    if config["initialize_only"] and config["prepare_only"]:
        raise ValueError("choose only one preparation mode")
    for key in ("epochs", "batch_size", "checkpoint_every", "cpu_threads"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("lr", "min_lr", "trunk_lr_scale"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["min_lr"] > config["lr"]:
        raise ValueError("min_lr must not exceed lr")


def initialize(config):
    """Export each candidate from the same frozen champion function."""
    import torch
    from training.student_model import Student, export
    torch.set_num_threads(config["cpu_threads"])
    champion = path(config["champion"])
    manifest_path = champion.with_suffix(".architecture.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha = sha(champion)
    if (manifest["architecture"], manifest["hidden"], manifest["features"],
            manifest["value_buckets"], manifest["value_depth"]) != (
                "multipath_phase_bucketed", 512, 504, 6, 2):
        raise ValueError("the initializer must be the current bucketed champion")
    if manifest["float_sha256"] != source_sha or sha(path(config["production_weights"])) != source_sha:
        raise ValueError("champion float weights do not match the production weights and manifest")
    old = Student("multipath_phase_bucketed", 512, qat=True)
    old.load_float(champion)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows = []
    for entry in config["networks"]:
        folder = path(config["out_dir"]) / entry["name"] / "initial"
        identity = dict(network=entry, champion_sha256=source_sha, champion_manifest_sha256=sha(manifest_path),
                        model_sha256=sha(ROOT / "training/student_model.py"), seed=config["seed"])
        record_path = folder / "initialization.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["identity"] != identity:
                raise ValueError(f"initialization changed; choose a new out_dir: {folder}")
            for artifact, digest in record["artifacts"].items():
                if sha(folder / artifact) != digest:
                    raise ValueError(f"initialized artifact changed: {folder / artifact}")
        else:
            if folder.exists() and any(folder.iterdir()):
                raise ValueError(f"partial initialization requires a new out_dir: {folder}")
            torch.manual_seed(config["seed"])
            model = Student(entry["architecture"], entry["hidden"], qat=True)
            model.warm_start(old)
            folder.mkdir(parents=True, exist_ok=True)
            architecture = export(model, folder / "student.bin")
            temporary = folder / "initial.pt.tmp"
            torch.save(dict(model=model.state_dict(), architecture=architecture, identity=identity), temporary)
            temporary.replace(folder / "initial.pt")
            record = dict(identity=identity, source_commit=commit, source_weights=str(champion),
                          initialized_only=True, training_started=False,
                          artifacts={file: sha(folder / file) for file in (
                              "student.bin", "student_int8.bin", "student.architecture.json", "initial.pt")})
            write_json(record_path, record)
        rows.append(dict(**entry, initial=str(folder / "student.bin"),
                         architecture_manifest=str(folder / "student.architecture.json"),
                         initialization_sha256=sha(record_path)))
    return rows


def validate_replay_source(entry):
    """Require searched targets and a frozen provenance manifest for new data."""
    dataset = path(entry["path"])
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not ready: {entry['name']}: {dataset}")
    if entry.get("historical"):
        return
    manifest_path = dataset.parent / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = manifest["config"]
    if not manifest.get("complete") or (recipe.get("mode"), recipe.get("stored_outcome_weight"),
            recipe.get("stored_gamma")) != ("stored_search", 0.5, 1.0):
        raise ValueError(f"new source requires complete stored-search 50/50 replay: {entry['name']}")
    digest = sha(dataset / "dataset.manifest.json") if dataset.is_dir() else sha(dataset)
    expected = manifest.get("dataset_manifest_sha256") if dataset.is_dir() else manifest.get("dataset_sha256")
    if expected != digest:
        raise ValueError(f"replay dataset SHA mismatch: {entry['name']}")


def prepare_replays(config):
    """Consume every accepted record from frozen raw sources without sampling caps."""
    from training.prepare_stored_replay_all import prepare
    for entry in config["data_sources"]:
        if entry.get("historical"):
            continue
        source = path(entry["raw_source"])
        campaign_file = source / "campaign_manifest.json"
        if campaign_file.exists():
            manifest = json.loads(campaign_file.read_text(encoding="utf-8"))
            if manifest.get("status") != "assigned_sources_complete_reserve_unassigned":
                raise ValueError("finish the assigned local sources before the final replay freeze")
            shards = manifest["shards"]
        else:
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("complete") is not True:
                raise ValueError(f"freeze an accepted-shard manifest before replay: {entry['name']}")
            shards = manifest["accepted_shards"]
        records = 0
        for shard in shards:
            binary = (source / shard["v3"]).resolve()
            if source.resolve() not in binary.parents:
                raise ValueError("raw shard path escapes source directory")
            metadata = binary.with_suffix(".meta")
            if binary.stat().st_size % 64 or metadata.stat().st_size != binary.stat().st_size // 64 * 20:
                raise ValueError(f"raw V3 and metadata are incomplete: {binary}")
            if not shard.get("bin_sha256") or not shard.get("meta_sha256"):
                raise ValueError(f"accepted shards require binary and metadata SHAs: {binary}")
            if sha(binary) != shard["bin_sha256"] or sha(metadata) != shard["meta_sha256"]:
                raise ValueError(f"raw shard SHA mismatch: {binary}")
            records += binary.stat().st_size // 64
        if records < 2:
            raise ValueError("replay requires at least two eligible records")
        destination = path(entry["path"])
        prepare(source, destination, seed=config["seed"])


def training_config(config, entry, mixed_path):
    from training.run_experiment import CONFIG as TRAIN_DEFAULTS
    result = copy.deepcopy(TRAIN_DEFAULTS)
    result.update(data=str(mixed_path), out_dir=str(path(config["out_dir"]) / entry["name"] / "train"),
                  architecture=entry["architecture"], hidden=entry["hidden"], init_from=entry["initial"],
                  init_architecture=entry["architecture"], init_hidden=entry["hidden"],
                  from_scratch=False, qat=True, mirror_h=True, epochs=config["epochs"],
                  lr=config["lr"], min_lr=config["min_lr"], trunk_lr_scale=config["trunk_lr_scale"],
                  warmup_epochs=config["warmup_epochs"], schedule="cosine", weight_decay=config["weight_decay"],
                  min_weight_decay=config["weight_decay"], weight_decay_schedule="constant",
                  policy_weight=config["policy_weight"], value_weight=config["value_weight"],
                  batch_size=config["batch_size"], patience=0, checkpoint_every=config["checkpoint_every"],
                  max_sample_weight=1e12, weight_boosts=[], train_scope="full", seed=config["seed"],
                  device=config["device"], cpu_threads=config["cpu_threads"], resume=config["resume"],
                  teaching=False, build=config["build"], benchmark=False, dry_run=False)
    return result


def frozen_retention(config, entry, recipe):
    """Compare initial and final candidates against unchanged historical targets."""
    import numpy as np
    import torch
    from training.mix_experimental_datasets import load_arrays
    from training.run_experiment import _epoch
    from training.student_model import Student
    device = recipe["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {}
    for source in config["data_sources"]:
        if not source.get("historical"):
            continue
        data = load_arrays(path(source["path"]), path(config["out_dir"]) / "mixed_dataset/.input_cache")
        cap = source.get("weight_cap")
        if cap is not None:
            data["weight"] = np.minimum(data["weight"], cap).astype(np.float32)
        validation = np.flatnonzero(data["is_val"])
        metrics = {}
        for label, weights in (("initial", entry["initial"]),
                               ("best", str(path(recipe["out_dir"]) / "student.bin"))):
            model = Student(entry["architecture"], entry["hidden"], qat=True)
            model.load_float(weights)
            model.to(device)
            metrics[label] = _epoch(model, data, validation, recipe, device)
            del model
        metrics["target_basis"] = "unchanged_historical_targets"
        metrics["validation_samples"] = len(validation)
        metrics["loss_delta"] = metrics["best"]["loss"] - metrics["initial"]["loss"]
        result[source["name"]] = metrics
        for array in data.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
    write_json(path(recipe["out_dir"]) / "frozen_retention.json", result)
    return result


def run(config):
    validate_config(config)
    plan = dict(schema="zquoridor.experimental.anneal.v1", config=config,
                status="dry_run", networks=config["networks"],
                datasets=[dict(**entry, ready=path(entry["path"]).exists()) for entry in config["data_sources"]],
                targets=dict(new_value="0.5 * signed MCAB root + 0.5 * terminal result", gamma=1,
                             historical="frozen targets and clipped historical weight ratios",
                             policy="untempered top-eight root visits"),
                mixture=dict(historical_mass=0.25, new_mass=0.75, use_all_eligible_records=True,
                             new_source_balance="proportional to eligible weight mass in each split"),
                reserve_unassigned=4000000, additional_teaching=False, automatic_promotion=False)
    if config["dry_run"]:
        return plan
    networks = initialize(config)
    output = path(config["out_dir"])
    plan.update(networks=networks, status="initialized_training_not_started")
    write_json(output / "campaign.json", plan)
    if config["initialize_only"]:
        return plan
    if config["prepare_replays"]:
        prepare_replays(config)
    for source in config["data_sources"]:
        validate_replay_source(source)
    from training.mix_experimental_datasets import mix_datasets
    sources = [dict(entry, path=str(path(entry["path"]))) for entry in config["data_sources"]]
    mixed_path = output / "mixed_dataset"
    mixture = mix_datasets(sources, mixed_path)
    plan.update(status="datasets_prepared_training_not_started", mixture=mixture)
    write_json(output / "campaign.json", plan)
    if config["prepare_only"]:
        return plan
    from training.run_experiment import train, build_candidate
    plan["jobs"] = []
    for entry in networks:
        job = dict(name=entry["name"], status="training")
        plan["jobs"].append(job)
        plan["status"] = "training"
        write_json(output / "campaign.json", plan)
        recipe = training_config(config, entry, mixed_path)
        try:
            report = train(recipe)
            retention = frozen_retention(config, entry, recipe)
            executable = build_candidate(recipe) if config["build"] else None
        except Exception as error:
            job.update(status="failed", error=str(error))
            plan["status"] = "failed"
            write_json(output / "campaign.json", plan)
            raise
        job.update(status="complete", epochs=report["epochs"], best_epoch=report["best_epoch"],
                   best_val_loss=report["best_val_loss"], frozen_retention=retention,
                   executable=str(executable) if executable else None)
        write_json(output / "campaign.json", plan)
    plan["status"] = "training_complete_promotion_pending"
    write_json(output / "campaign.json", plan)
    return plan


def main(argv=None):
    try:
        print(json.dumps(run(parse_config(argv)), indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"experimental annealing error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
