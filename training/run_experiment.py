#!/usr/bin/env python3
"""Train, export, and build one compact NNUE experiment.

Input is a teaching ``dataset.npz``.  ``base`` uses 354 features and ``race``
uses 456; hidden widths 128, 256, 384, and 512 are supported.  The output
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
import numpy as np
import torch
from torch.nn import functional as F
from student_model import Student, encode_features, export

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "data": "data/teaching/pilot/dataset.npz",
    "out_dir": "results/experiments/race256",
    "architecture": "race",
    "hidden": 256,
    "init_from": "checkpoints/gen9-teacher-head/nnue_weights.bin",
    "init_architecture": "base",
    "init_hidden": 256,
    "from_scratch": False,
    "qat": True,
    "epochs": 80,
    "batch_size": 256,
    "lr": 0.0001,
    "schedule": "cosine",
    "warmup_epochs": 4,
    "min_lr": 0.000005,
    "trunk_lr_scale": 0.1,
    "weight_decay": 0.00001,
    "policy_weight": 1.0,
    "value_weight": 1.0,
    "train_scope": "full",
    "grad_clip": 1.0,
    "patience": 12,
    "seed": 20260914,
    "device": "auto",
    "cpu_threads": 4,
    "resume": True,
    "teaching": True,
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
    for key in (("group_id",) if "group_id" in data else ("opening_index",)):
        if key in data:
            groups = np.asarray(data[key]).astype(str)
            if set(groups[val]) & set(groups[~val]):
                raise ValueError(f"training/validation group overlap in {key}")
    return np.flatnonzero(~val), np.flatnonzero(val)


def load_dataset(path):
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
    chunk_size = 500_000
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
    split_indices(data)
    return data


def _path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def _epoch(model, data, indices, config, device, optimizer=None, rng=None):
    model.train(optimizer is not None)
    order = rng.permutation(indices) if optimizer is not None else indices
    totals = np.zeros(4, dtype=np.float64)
    for start in range(0, len(order), config["batch_size"]):
        idx = order[start:start + config["batch_size"]]
        x = torch.from_numpy(encode_features(data, idx, config["architecture"])).to(device)
        p = torch.as_tensor(data["policy"][idx].astype(np.float32), device=device)
        v = torch.as_tensor(data["value"][idx].astype(np.float32), device=device)
        w = torch.as_tensor(data["weight"][idx].astype(np.float32), device=device)
        with torch.set_grad_enabled(optimizer is not None):
            logits, policy = model(x)
            kl = (p * (p.clamp_min(1e-12).log() - F.log_softmax(policy, dim=1))).sum(1)
            vloss = F.binary_cross_entropy_with_logits(logits, (v + 1) / 2, reduction="none")
            loss = ((config["policy_weight"] * kl + config["value_weight"] * vloss) * w).sum() / w.sum()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                model.clip_weights()
        mass = w.sum().item()
        totals += [loss.item() * mass, (kl * w).sum().item(),
                   ((2 * logits.sigmoid() - 1 - v).abs() * w).sum().item(), mass]
    return dict(loss=float(totals[0] / totals[3]), policy_kl=float(totals[1] / totals[3]),
                value_mae=float(totals[2] / totals[3]))


def train(config):
    for key in ("epochs", "batch_size", "patience", "cpu_threads", "warmup_epochs"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("lr", "min_lr", "trunk_lr_scale", "grad_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("policy_weight", "value_weight", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["policy_weight"] + config["value_weight"] == 0:
        raise ValueError("at least one loss weight must be positive")
    if config["min_lr"] > config["lr"]:
        raise ValueError("min_lr must not exceed lr")
    if config["schedule"] not in ("constant", "cosine"):
        raise ValueError("schedule must be constant or cosine")
    if config["train_scope"] not in ("full", "policy", "heads"):
        raise ValueError("train_scope must be full, policy, or heads")
    path, folder = _path(config["data"]), _path(config["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset(path)
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
    identity = dict(config=identity_config, data_sha256=_hash(path),
                    init_sha256=None if config["from_scratch"] else _hash(_path(config["init_from"])),
                    trainer_sha256=_hash(__file__), model_sha256=_hash(Path(__file__).with_name("student_model.py")))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    checkpoint = folder / "resume.pt"
    history, start, bad = [], 0, 0
    initial = _epoch(model, data, val_idx, config, device)
    best_loss = initial["loss"]
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
    (folder / "config.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    for epoch in range(start, config["epochs"]):
        if bad >= config["patience"]:
            break
        base_lr = learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = base_lr * group["lr_scale"]
        training = _epoch(model, data, train_idx, config, device, optimizer, rng)
        validation = _epoch(model, data, val_idx, config, device)
        row = dict(epoch=epoch + 1, lr=base_lr, train=training, val=validation)
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss, bad = validation["loss"], 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        state = dict(fingerprint=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     best=best, best_loss=best_loss, bad=bad, history=history, epoch=epoch + 1,
                     initial=initial, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)
        temp = checkpoint.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(checkpoint)
    model.load_state_dict(best)
    architecture = export(model, folder / "student.bin")
    report = dict(fingerprint=fingerprint, architecture=architecture, initial_val=initial,
                  best_val_loss=best_loss, samples=len(data["value"]), train_samples=len(train_idx),
                  val_samples=len(val_idx), epochs=len(history), history=history,
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
    flags = [f"-DZQ_NNUE_RACE_FEATURES={int(config['architecture'] == 'race')}",
             f"-DZQ_NNUE_HIDDEN={config['hidden']}"]
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
