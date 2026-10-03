#!/usr/bin/env python3
"""Train, export, and build one compact NNUE experiment.

Input is a teaching NPZ or a manifest-verified memory-map directory. Hidden
widths 128, 256, 384, 512, 768, and 1024 are supported. The output
directory receives float and int8 weights, an architecture manifest, a
restart checkpoint, a training report, and a native executable compiled with
matching feature and width flags.  The top-level ``CONFIG`` supplies defaults
and CLI options override them.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
try:
    from .student_model import Student, encode_features, export, ARCH_CONFIGS
    from .mirror_augmentation import stochastic_mirror_dict_h
except ImportError:  # Direct execution from the training directory.
    from student_model import Student, encode_features, export, ARCH_CONFIGS
    from mirror_augmentation import stochastic_mirror_dict_h

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "data": "data/teaching/multipath_unified_clean_15m/dataset.npz",
    "out_dir": "results/experiments/multipath_contact_bucketed_unified",
    "architecture": "multipath_phase_contact_bucketed",
    "hidden": 512,
    "init_from": "results/experiments/multipath_unified_champion/student.bin",
    "init_architecture": "multipath_phase_bucketed",
    "init_hidden": 512,
    "from_scratch": False,
    "qat": True,
    "mirror_h": True,
    "epochs": 60,
    "batch_size": 1024,
    "lr": 0.0001,
    "schedule": "cosine",
    "warmup_epochs": 4,
    "min_lr": 0.000005,
    "trunk_lr_scale": 0.1,
    "weight_decay": 0.00001,
    "weight_decay_schedule": "constant",
    "min_weight_decay": 0.00001,
    "weight_boosts": [],
    "max_sample_weight": 30.0,
    "policy_weight": 1.0,
    "value_weight": 1.0,
    "train_scope": "full",
    "grad_clip": 1.0,
    "patience": 12,
    "seed": 20260914,
    "device": "cuda",
    "cpu_threads": 4,
    "resume": True,
    "checkpoint_every": 0,
    "teaching": False,
    "teaching_args": [],
    "build": True,
    "benchmark": False,
    "benchmark_args": [],
    "dry_run": False,
}


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif isinstance(value, list):
            parser.add_argument(flag, type=json.loads, default=argparse.SUPPRESS,
                                help='JSON argument list, for example ["--games", "100"]')
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def split_indices(data):
    val = np.asarray(data["is_val"], dtype=bool)
    if not val.any() or val.all():
        raise ValueError("both training and validation groups are required; no random row split")
    for key in ("group_id", "opening_index"):
        if key in data and not data.get("_group_separation_verified", False):
            groups = np.asarray(data[key])
            if set(groups[val].tolist()) & set(groups[~val].tolist()):
                raise ValueError(f"training/validation group overlap in {key}")
    return np.flatnonzero(~val), np.flatnonzero(val)


def load_dataset(path):
    path = Path(path)
    if path.is_dir():
        manifest_path = path / "dataset.manifest.json"
        if not manifest_path.is_file():
            raise ValueError("dataset directory lacks dataset.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError("dataset manifest is not complete")
        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict):
            raise ValueError("dataset manifest lacks an arrays map")
        data = {}
        root = path.resolve()
        for key, entry in arrays.items():
            if not isinstance(entry, dict) or not entry.get("filename") or not entry.get("sha256"):
                raise ValueError(f"invalid dataset array entry: {key}")
            array_path = (path / entry["filename"]).resolve()
            try:
                array_path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"dataset array path escapes directory: {key}") from error
            if not array_path.is_file() or _hash(array_path) != entry["sha256"]:
                raise ValueError(f"dataset array is missing or has a hash mismatch: {key}")
            if key == "source_names" and "source_names" in manifest:
                continue
            data[key] = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if "source_names" in manifest:
            data["source_names"] = manifest["source_names"]
        elif "source_names" in data:
            names = data["source_names"]
            data["source_names"] = names.tolist()
            if isinstance(names, np.memmap):
                names._mmap.close()
        if manifest.get("group_separation_verified") is True:
            data["_group_separation_verified"] = True
    else:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
    required = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                "walls_left_own", "walls_left_opp", "policy", "value", "weight", "is_val")
    if any(key not in data for key in required):
        raise ValueError("dataset lacks required canonical state/target fields")
    n = len(data["value"])
    if n < 2 or any(len(data[key]) != n for key in required):
        raise ValueError("dataset has insufficient samples or inconsistent lengths")
    policy_arr = data["policy"]
    if policy_arr.shape != (n, 209):
        raise ValueError("policy must contain 209-action rows")
    chunk_size = 65_536
    for start_idx in range(0, n, chunk_size):
        chunk = policy_arr[start_idx:start_idx + chunk_size].astype(np.float32)
        if not np.isfinite(chunk).all() or (chunk < 0).any():
            raise ValueError("policy must contain finite nonnegative 209-action rows")
        if not np.allclose(chunk.sum(1), 1, atol=0.005):
            raise ValueError("policy rows are not normalized")
    value, weight = data["value"], data["weight"]
    if not np.isfinite(value).all() or (np.abs(value) > 1.001).any():
        raise ValueError("value must contain finite signed mover targets in [-1,1]")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("sample weights must be finite and positive")
    for key, maximum in (("own_pawn", 80), ("opp_pawn", 80),
                         ("walls_left_own", 10), ("walls_left_opp", 10),
                         ("own_dist", 81), ("opp_dist", 81)):
        if (data[key] < 0).any() or (data[key] > maximum).any():
            raise ValueError(f"invalid state field {key}")
    if data.get("source_mass") is not None:
        source_mass = data["source_mass"]
        if source_mass.ndim != 2 or source_mass.shape[0] != n:
            raise ValueError("source_mass must have shape (samples, sources)")
        for start_idx in range(0, n, 500_000):
            stop_idx = start_idx + 500_000
            chunk = source_mass[start_idx:stop_idx]
            if not np.isfinite(chunk).all() or (chunk < 0).any():
                raise ValueError("source_mass must contain finite nonnegative values")
            mass_sum = chunk.sum(axis=1, dtype=np.float64)
            sample_weight = np.asarray(data["weight"][start_idx:stop_idx], dtype=np.float64)
            if not np.allclose(mass_sum, sample_weight, rtol=2e-5, atol=2e-6):
                raise ValueError("source_mass row sums must match sample weights")
        names = data.get("source_names", [])
        if len(names) != source_mass.shape[1]:
            raise ValueError("source_names must match the source_mass columns")
    if not data.get("_group_separation_verified", False):
        split_indices(data)
    if "group_id" in data:
        del data["group_id"]
    return data


def apply_weight_boosts(weights, boosts, max_weight):
    """Scale nonoverlapping sample ranges and limit the final weights."""
    result = np.asarray(weights, dtype=np.float32).copy()
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("max_sample_weight must be finite and positive")
    previous_end = 0
    for item in sorted(boosts, key=lambda value: value[0]):
        if len(item) != 3:
            raise ValueError("each weight boost must contain start, end, and factor")
        start, end, factor = item
        if int(start) != start or int(end) != end or start < previous_end or end <= start or end > len(result):
            raise ValueError("weight boost ranges must be valid and nonoverlapping")
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("weight boost factors must be finite and positive")
        result[int(start):int(end)] *= float(factor)
        previous_end = int(end)
    np.clip(result, 0, float(max_weight), out=result)
    return result


def _path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dataset_hash(path):
    path = Path(path)
    if path.is_dir():
        return _hash(path / "dataset.manifest.json")
    return _hash(path)


def _atomic_torch_save(state, path):
    """Write a checkpoint without exposing a partial file."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def learning_rate(config, epoch: int) -> float:
    """Return the base learning rate for zero-based ``epoch``.

    QAT is active throughout training.  Warmup avoids an abrupt first update;
    cosine annealing then leaves a low-rate QAT tail for final quantized
    weight adjustment.  The final scheduled epoch is exactly ``min_lr``.
    """
    if not 0 <= epoch < config["epochs"]:
        raise ValueError("epoch is outside the configured training range")
    if config["schedule"] == "constant":
        return float(config["lr"])
    if config["schedule"] != "cosine":
        raise ValueError("schedule must be constant or cosine")
    warmup = min(int(config["warmup_epochs"]), int(config["epochs"]))
    if warmup and epoch < warmup:
        return float(config["lr"]) * (epoch + 1) / warmup
    remaining = config["epochs"] - warmup
    if remaining <= 1:
        return float(config["min_lr"])
    progress = (epoch - warmup) / (remaining - 1)
    return float(config["min_lr"]) + (float(config["lr"]) - float(config["min_lr"])) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def weight_decay(config, epoch: int) -> float:
    """Return the weight decay for zero-based ``epoch``."""
    if not 0 <= epoch < config["epochs"]:
        raise ValueError("epoch is outside the configured training range")
    schedule = config["weight_decay_schedule"]
    if schedule == "constant":
        return float(config["weight_decay"])
    if schedule != "cosine":
        raise ValueError("weight_decay_schedule must be constant or cosine")
    if config["epochs"] <= 1:
        return float(config["min_weight_decay"])
    progress = epoch / (config["epochs"] - 1)
    return float(config["min_weight_decay"]) + (
        float(config["weight_decay"]) - float(config["min_weight_decay"])
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))


_EPOCH_BLOCK_SIZE = 32_768
_EPOCH_FIELDS = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                 "walls_left_own", "walls_left_opp", "policy", "value", "weight")


def _pinned_tensor(array, use_cuda):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.pin_memory() if use_cuda else tensor


def _prepare_epoch_block(data, row_indices, config, use_cuda, training, row_order=None, flip_mask=None):
    """Read one mostly-contiguous block and prepare it while the GPU trains the prior block."""
    block = {key: np.asarray(data[key][row_indices]) for key in _EPOCH_FIELDS}
    if training:
        if flip_mask is not None:
            block = stochastic_mirror_dict_h(block, row_order, flip_mask)
        else:
            block = {key: values[row_order] for key, values in block.items()}
        feature_rows = np.arange(len(row_order))
    else:
        feature_rows = np.arange(len(row_indices))
    features = encode_features(block, feature_rows, config["architecture"])
    prepared = {
        "x": _pinned_tensor(features, use_cuda),
        "p": _pinned_tensor(block["policy"].astype(np.float32, copy=False), use_cuda),
        "v": _pinned_tensor(block["value"].astype(np.float32, copy=False), use_cuda),
        "w": _pinned_tensor(block["weight"].astype(np.float32, copy=False), use_cuda),
    }
    if not training and "source_mass" in data:
        prepared["source_mass"] = _pinned_tensor(
            np.asarray(data["source_mass"][row_indices], dtype=np.float32), use_cuda)
    return prepared


def _epoch(model, data, indices, config, device, optimizer=None, rng=None):
    model.train(optimizer is not None)
    training = optimizer is not None
    use_cuda = str(device).startswith("cuda")
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    source_names = data.get("source_names", [])
    source_totals = (torch.zeros((len(source_names), 5), dtype=torch.float64, device=device)
                     if not training and "source_mass" in data else None)
    bucket_counts = (torch.zeros(model.value_buckets, dtype=torch.int64, device=device)
                     if model.value_buckets > 1 and not training else None)
    bucket_mae_sums = (torch.zeros(model.value_buckets, dtype=torch.float64, device=device)
                       if bucket_counts is not None else None)
    bucket_weights = (torch.zeros(model.value_buckets, dtype=torch.float64, device=device)
                      if bucket_counts is not None else None)

    block_size = max(_EPOCH_BLOCK_SIZE, int(config["batch_size"]))
    n_blocks = (len(indices) + block_size - 1) // block_size
    block_order = np.arange(n_blocks)
    if training:
        rng.shuffle(block_order)

    def submit_block(pool, slot):
        start = int(block_order[slot]) * block_size
        rows = np.asarray(indices[start:start + block_size]).copy()
        if training:
            row_order = rng.permutation(len(rows))
            flip_mask = rng.random(len(rows)) < 0.5 if config.get("mirror_h", True) else None
        else:
            row_order = flip_mask = None
        return pool.submit(_prepare_epoch_block, data, rows, config, use_cuda,
                           training, row_order, flip_mask)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="zq-data-prefetch") as pool:
        pending = submit_block(pool, 0) if n_blocks else None
        for block_slot in range(n_blocks):
            prepared = pending.result()
            pending = submit_block(pool, block_slot + 1) if block_slot + 1 < n_blocks else None
            n_rows = len(prepared["v"])
            for start in range(0, n_rows, config["batch_size"]):
                end = min(start + config["batch_size"], n_rows)
                x = prepared["x"][start:end].to(device, non_blocking=use_cuda)
                p = prepared["p"][start:end].to(device, non_blocking=use_cuda)
                v = prepared["v"][start:end].to(device, non_blocking=use_cuda)
                w = prepared["w"][start:end].to(device, non_blocking=use_cuda)

                with torch.set_grad_enabled(training):
                    logits, policy = model(x)
                    kl = (p * (p.clamp_min(1e-12).log() - F.log_softmax(policy, dim=1))).sum(1)
                    target = (v + 1) / 2
                    vloss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                    probability = logits.sigmoid()
                    brier = (probability - target).square()
                    per_sample_loss = config["policy_weight"] * kl + config["value_weight"] * vloss
                    mass = w.sum()
                    loss = (per_sample_loss * w).sum() / mass
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                        optimizer.step()
                        model.clip_weights()

                val_diff = (2 * probability - 1 - v).abs()
                totals += torch.stack((loss.detach() * mass, (kl * w).sum(),
                                       (val_diff * w).sum(), mass,
                                       (vloss * w).sum(), (brier * w).sum())).detach().to(torch.float64)

                if source_totals is not None:
                    source_weight = prepared["source_mass"][start:end].to(device, non_blocking=use_cuda)
                    source_totals += torch.stack((
                        (per_sample_loss[:, None] * source_weight).sum(0),
                        (val_diff[:, None] * source_weight).sum(0),
                        (vloss[:, None] * source_weight).sum(0),
                        (brier[:, None] * source_weight).sum(0),
                        source_weight.sum(0),
                    ), dim=1).detach().to(torch.float64)

                if bucket_counts is not None:
                    bucket_ids = model._extract_buckets(x)
                    bucket_counts += torch.bincount(bucket_ids, minlength=model.value_buckets)
                    bucket_mae_sums.scatter_add_(0, bucket_ids, val_diff.to(torch.float64) * w)
                    bucket_weights.scatter_add_(0, bucket_ids, w.to(torch.float64))

            del prepared

    if not torch.isfinite(totals).all():
        raise ValueError("non-finite training metrics")
    totals = totals.cpu().numpy()
    res = dict(loss=float(totals[0] / totals[3]), policy_kl=float(totals[1] / totals[3]),
               value_mae=float(totals[2] / totals[3]), value_bce=float(totals[4] / totals[3]),
               value_brier=float(totals[5] / totals[3]))
    if source_totals is not None:
        source_totals = source_totals.cpu().numpy()
        res["by_source"] = {
            name: dict(loss=float(row[0] / row[4]) if row[4] else 0.0,
                       value_mae=float(row[1] / row[4]) if row[4] else 0.0,
                       value_bce=float(row[2] / row[4]) if row[4] else 0.0,
                       value_brier=float(row[3] / row[4]) if row[4] else 0.0,
                       mass=float(row[4]))
            for name, row in zip(source_names, source_totals)
        }
        res["by_source_target_basis"] = "blended_global_targets"
    if bucket_counts is not None:
        bucket_counts, bucket_mae_sums, bucket_weights = (
            value.cpu().numpy() for value in (bucket_counts, bucket_mae_sums, bucket_weights))
        res["bucket_counts"] = bucket_counts.tolist()
        res["bucket_mae"] = [
            float(bucket_mae_sums[b] / bucket_weights[b]) if bucket_weights[b] > 0 else 0.0
            for b in range(model.value_buckets)
        ]
    return res


def train(config):
    for key in ("epochs", "batch_size", "cpu_threads"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["checkpoint_every"] < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    for key in ("patience", "warmup_epochs"):
        if config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("lr", "min_lr", "trunk_lr_scale", "grad_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("policy_weight", "value_weight", "weight_decay", "min_weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["policy_weight"] + config["value_weight"] == 0:
        raise ValueError("at least one loss weight must be positive")
    if config["min_lr"] > config["lr"]:
        raise ValueError("min_lr must not exceed lr")
    if config["min_weight_decay"] > config["weight_decay"]:
        raise ValueError("min_weight_decay must not exceed weight_decay")
    if config["schedule"] not in ("constant", "cosine"):
        raise ValueError("schedule must be constant or cosine")
    if config["weight_decay_schedule"] not in ("constant", "cosine"):
        raise ValueError("weight_decay_schedule must be constant or cosine")
    if config["train_scope"] not in ("full", "policy", "heads"):
        raise ValueError("train_scope must be full, policy, or heads")
    path, folder = _path(config["data"]), _path(config["out_dir"])
    checkpoint = folder / "resume.pt"
    if not config["resume"] and checkpoint.exists():
        raise ValueError("resume.pt exists; choose a new out_dir or enable resume")
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset(path)
    data["weight"] = apply_weight_boosts(
        data["weight"], config["weight_boosts"], config["max_sample_weight"])
    train_idx, val_idx = split_indices(data)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Student(config["architecture"], config["hidden"], qat=config["qat"])
    if not config["from_scratch"]:
        old = Student(config["init_architecture"], config["init_hidden"])
        old.load_float(_path(config["init_from"]))
        model.warm_start(old)
    model.to(device)
    for name, param in model.named_parameters():
        enabled = config["train_scope"] == "full" or name.startswith("policy.")
        enabled |= config["train_scope"] == "heads" and name.startswith("value")
        param.requires_grad_(enabled)
    groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            scale = config["trunk_lr_scale"] if name.startswith("fc1") else 1.0
            groups.append({"params": [param], "lr": config["lr"] * scale, "lr_scale": scale})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    identity_config = {k: v for k, v in config.items() if k not in
                       ("resume", "build", "benchmark", "benchmark_args", "dry_run", "teaching", "teaching_args")}
    identity = dict(config=identity_config, data_sha256=_dataset_hash(path),
                    init_sha256=None if config["from_scratch"] else _hash(_path(config["init_from"])),
                    trainer_sha256=_hash(__file__), model_sha256=_hash(Path(__file__).with_name("student_model.py")))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    initial_checkpoint = folder / "initial.pt"
    best_checkpoint = folder / "best.pt"
    checkpoint_every = int(config["checkpoint_every"])
    history, start, bad = [], 0, 0
    if config["resume"] and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("resume configuration or inputs changed; choose a new out_dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best, best_loss, bad = state["best"], state["best_loss"], state["bad"]
        history, start, initial = state["history"], state["epoch"], state["initial"]
        best_epoch = state.get("best_epoch", min(
            (row["epoch"] for row in history if row["val"]["loss"] == best_loss), default=0))
        print(f"[resume] checkpoint valido na epoca {start}; pulando avaliacao inicial", flush=True)
    else:
        initial = _epoch(model, data, val_idx, config, device)
        best_loss = initial["loss"]
        best_epoch = 0
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    (folder / "config.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    if start == 0 and not initial_checkpoint.exists():
        initial_state = dict(fingerprint=fingerprint, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), best=best, best_loss=best_loss,
                             best_epoch=best_epoch, bad=0, history=[], epoch=0, initial=initial,
                             rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
        _atomic_torch_save(initial_state, initial_checkpoint)
        _atomic_torch_save(initial_state, best_checkpoint)
    stopped_early = False
    for epoch in range(start, config["epochs"]):
        if config["patience"] > 0 and bad >= config["patience"]:
            stopped_early = True
            break
        base_lr = learning_rate(config, epoch)
        epoch_weight_decay = weight_decay(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = base_lr * group["lr_scale"]
            group["weight_decay"] = epoch_weight_decay
        training = _epoch(model, data, train_idx, config, device, optimizer, rng)
        validation = _epoch(model, data, val_idx, config, device)
        row = dict(epoch=epoch + 1, lr=base_lr, weight_decay=epoch_weight_decay,
                   train=training, val=validation)
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss, bad = validation["loss"], 0
            best_epoch = epoch + 1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        state = dict(fingerprint=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     best=best, best_loss=best_loss, best_epoch=best_epoch, bad=bad, history=history, epoch=epoch + 1,
                     initial=initial, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
        _atomic_torch_save(state, checkpoint)
        if best_epoch == epoch + 1:
            best_state = dict(state, model=best, epoch=best_epoch)
            _atomic_torch_save(best_state, best_checkpoint)
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            _atomic_torch_save(state, folder / f"epoch_{epoch + 1:04d}.pt")
    if config["patience"] > 0 and bad >= config["patience"] and len(history) < config["epochs"]:
        stopped_early = True
    model.load_state_dict(best)
    architecture = export(model, folder / "student.bin")
    report = dict(fingerprint=fingerprint, architecture=architecture, initial_val=initial,
                  best_val_loss=best_loss, best_epoch=best_epoch,
                  samples=len(data["value"]), train_samples=len(train_idx),
                  val_samples=len(val_idx), epochs=len(history), configured_epochs=config["epochs"],
                  training_status="early_stopped" if stopped_early else "complete", history=history,
                  schedule=dict(name=config["schedule"], warmup_epochs=config["warmup_epochs"],
                                initial_lr=config["lr"], min_lr=config["min_lr"]),
                  improved_validation=best_loss < initial["loss"], promoted=False)
    (folder / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_candidate(config):
    compiler = shutil.which("g++")
    if compiler is None and Path("C:/mingw64/bin/g++.exe").is_file():
        compiler = "C:/mingw64/bin/g++.exe"
    if compiler is None:
        raise RuntimeError("g++ is required to build the native candidate")
    folder = _path(config["out_dir"])
    suffix = ".exe" if os.name == "nt" else ""
    exe = folder / ("zquoridor" + suffix)
    arch_cfg = ARCH_CONFIGS.get(config['architecture'], {})
    val_buckets = arch_cfg.get('buckets', 1)
    val_depth = arch_cfg.get('depth', 1)

    flags = [f"-DZQ_NNUE_RACE_FEATURES={int(config['architecture'] in ('race', 'multipath', 'margin_regime', 'phase', 'margin_phase', 'multipath_phase', 'multipath_phase_contact', 'multipath_phase_bucketed', 'multipath_phase_deep', 'multipath_phase_contact_bucketed'))}",
             f"-DZQ_NNUE_MULTIPATH_FEATURES={int(config['architecture'] in ('multipath', 'multipath_phase', 'multipath_phase_contact', 'multipath_phase_bucketed', 'multipath_phase_deep', 'multipath_phase_contact_bucketed'))}",
             f"-DZQ_NNUE_MARGIN_REGIME_FEATURES={int(config['architecture'] in ('margin_regime', 'margin_phase'))}",
             f"-DZQ_NNUE_PHASE_FEATURES={int(config['architecture'] in ('phase', 'margin_phase', 'multipath_phase', 'multipath_phase_contact', 'multipath_phase_bucketed', 'multipath_phase_deep', 'multipath_phase_contact_bucketed'))}",
             f"-DZQ_NNUE_CONTACT_FEATURES={int(config['architecture'] in ('multipath_phase_contact', 'multipath_phase_contact_bucketed'))}",
             f"-DZQ_NNUE_VALUE_BUCKETS={val_buckets}",
             f"-DZQ_NNUE_VALUE_DEPTH={val_depth}",
             f"-DZQ_NNUE_HIDDEN={config['hidden']}"]
    arch_json = folder / "student.architecture.json"
    if arch_json.exists():
        manifest = json.loads(arch_json.read_text(encoding="utf-8"))
        flags = manifest.get("cpp_flags", flags)
    build_inputs = dict(flags=flags, compiler=_hash(Path(compiler)),
        files={str(p.relative_to(ROOT)): _hash(p) for p in [ROOT/"tools/external/zquoridor_uci.cpp",
            ROOT/"tests/nnue_incremental_check.cpp", *sorted((ROOT/"src").glob("*.hpp"))]},
        weights=_hash(folder/"student_int8.bin"))
    build_manifest = folder / "build_manifest.json"
    if exe.exists() and build_manifest.exists():
        previous = json.loads(build_manifest.read_text())
        if previous.get("inputs") == build_inputs and previous.get("executable_sha256") == _hash(exe):
            return exe
    subprocess.run([compiler, "-O3", "-std=c++17", "-march=native", "-pthread", *flags,
                    "-I" + str(ROOT / "src"), str(ROOT / "tools/external/zquoridor_uci.cpp"),
                    "-o", str(exe)], check=True, cwd=ROOT)
    verify = folder / ("incremental_check" + suffix)
    subprocess.run([compiler, "-O2", "-std=c++17", *flags,
                    str(ROOT / "tests/nnue_incremental_check.cpp"), "-o", str(verify)], check=True, cwd=ROOT)
    subprocess.run([str(verify), str(folder / "student_int8.bin")], check=True, cwd=ROOT)
    build_manifest.write_text(json.dumps(dict(inputs=build_inputs, executable_sha256=_hash(exe)), indent=2)+"\n", encoding="utf-8")
    return exe


def main(argv=None):
    config = parse_config(argv)
    if config["dry_run"]:
        print(json.dumps(config, indent=2))
        return 0
    if config["teaching"]:
        subprocess.run([sys.executable, str(ROOT / "training/run_teaching.py"),
                        *config["teaching_args"]], check=True, cwd=ROOT)
    report = train(config)
    exe = build_candidate(config) if config["build"] else None
    if config["benchmark"]:
        if exe is None:
            raise ValueError("benchmark requires build")
        subprocess.run([sys.executable, str(ROOT / "tools/benchmark_candidate.py"),
                        "--candidate-executable", str(exe),
                        "--candidate-nnue", str(_path(config["out_dir"]) / "student_int8.bin"),
                        *config["benchmark_args"]], check=True, cwd=ROOT)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
