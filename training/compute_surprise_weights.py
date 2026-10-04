#!/usr/bin/env python3
"""Compute policy surprise weights using the frozen baseline network.

The script evaluates Kullback-Leibler divergence between the stored search
policy and the prior policy predicted by the frozen baseline network.
Positions where search discovered unexpected moves receive higher sample
weights. The multiplier remains bounded and preserves the total sample mass.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from training.student_model import Student, encode_features

CONFIG: Dict[str, Any] = {
    "data": str(ROOT / "results" / "experiments" / "production-central-finetune-20261002" / "mixed_dataset"),
    "model": str(ROOT / "data" / "nnue" / "nnue_weights.bin"),
    "architecture": "multipath_phase_bucketed",
    "hidden": 512,
    "alpha": 0.5,
    "s_max": 4.0,
    "batch_size": 16384,
    "device": "cuda",
    "output_weights": "",
    "output_kl": "",
    "output_dataset_dir": "",
    "normalize_mass": True,
}


def parse_config(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse configuration with CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def compute_policy_kl_batch(
    model: Student,
    x: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Kullback-Leibler divergence from prior to search policy.

    The target policy p represents search visit frequencies.
    The model outputs raw policy logits. The divergence is computed as:
    KL(p || prior) = sum_a p(a) * (log p(a) - log_softmax(logits)(a)).
    """
    _, policy_logits = model(x)
    log_prior = F.log_softmax(policy_logits, dim=1)
    p_safe = p.clamp_min(eps)
    kl = (p * (p_safe.log() - log_prior)).sum(dim=1)
    return kl.clamp_min(0.0)


def compute_normalized_multipliers(
    kl_array: np.ndarray,
    base_weights: np.ndarray,
    alpha: float = 0.5,
    s_max: float = 4.0,
    normalize_mass: bool = True,
) -> np.ndarray:
    """Calculate bounded surprise weight multipliers.

    A linear scaling factor alpha magnifies surprise values up to s_max.
    When normalize_mass is True, the multipliers are scaled so that the
    total weighted mass remains identical to the original base weights.
    """
    if alpha < 0.0:
        raise ValueError("The surprise factor alpha must be nonnegative.")
    if s_max <= 0.0:
        raise ValueError("The maximum surprise cutoff s_max must be positive.")

    clipped_kl = np.clip(kl_array, 0.0, float(s_max))
    raw_mult = 1.0 + float(alpha) * clipped_kl

    if not normalize_mass:
        return raw_mult.astype(np.float32)

    total_base_mass = float(np.sum(base_weights, dtype=np.float64))
    total_boosted_mass = float(np.sum(base_weights * raw_mult, dtype=np.float64))

    if total_boosted_mass <= 0.0 or total_base_mass <= 0.0:
        return np.ones_like(raw_mult, dtype=np.float32)

    scale_factor = total_base_mass / total_boosted_mass
    normalized_mult = raw_mult * scale_factor
    return normalized_mult.astype(np.float32)


def evaluate_dataset_surprise(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate policy surprise across the complete dataset."""
    data_path = Path(config["data"])
    fields_dir = data_path / "fields" if (data_path / "fields").is_dir() else data_path

    model = Student(config["architecture"], config["hidden"])
    model.load_float(Path(config["model"]))
    model.eval()

    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    print(f"Loading dataset from: {data_path}")
    print(f"Using device: {device} | Batch size: {config['batch_size']}")

    policy_arr = np.load(fields_dir / "policy.npy", mmap_mode="r")
    base_weights = np.load(fields_dir / "weight.npy", mmap_mode="r")
    total_samples = len(base_weights)
    print(f"Total samples to evaluate: {total_samples}")

    batch_size = int(config["batch_size"])
    kl_results = np.zeros(total_samples, dtype=np.float32)

    block_keys = (
        "own_pawn", "opp_pawn", "walls_h", "walls_v",
        "own_dist", "opp_dist", "walls_left_own", "walls_left_opp"
    )
    blocks = {key: np.load(fields_dir / f"{key}.npy", mmap_mode="r") for key in block_keys}

    start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    if start_time:
        start_time.record()

    with torch.inference_mode():
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_slice = slice(start_idx, end_idx)
            current_batch_len = end_idx - start_idx

            batch_block = {key: blocks[key][batch_slice] for key in block_keys}
            batch_policy = np.asarray(policy_arr[batch_slice], dtype=np.float32)

            feature_rows = np.arange(current_batch_len)
            feats = encode_features(batch_block, feature_rows, config["architecture"])

            x_tensor = torch.from_numpy(feats).to(device)
            p_tensor = torch.from_numpy(batch_policy).to(device)

            kl_batch = compute_policy_kl_batch(model, x_tensor, p_tensor)
            kl_results[batch_slice] = kl_batch.cpu().numpy()

            if (start_idx // batch_size) % 500 == 0 or end_idx == total_samples:
                pct = 100.0 * end_idx / total_samples
                print(f"Progress: {end_idx:,}/{total_samples:,} ({pct:.1f}%)", flush=True)

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_sec = start_time.elapsed_time(end_time) / 1000.0
        print(f"Evaluation finished in {elapsed_sec:.2f} seconds.")

    stats = {
        "samples": total_samples,
        "mean_kl": float(np.mean(kl_results)),
        "median_kl": float(np.median(kl_results)),
        "p90_kl": float(np.percentile(kl_results, 90)),
        "p99_kl": float(np.percentile(kl_results, 99)),
        "min_kl": float(np.min(kl_results)),
        "max_kl": float(np.max(kl_results)),
    }
    print(f"\nSurprise Statistics (KL Divergence):")
    print(f"  Mean:   {stats['mean_kl']:.5f}")
    print(f"  Median: {stats['median_kl']:.5f}")
    print(f"  90th %: {stats['p90_kl']:.5f}")
    print(f"  99th %: {stats['p99_kl']:.5f}")
    print(f"  Min:    {stats['min_kl']:.5f}")
    print(f"  Max:    {stats['max_kl']:.5f}")

    multipliers = compute_normalized_multipliers(
        kl_results,
        np.asarray(base_weights, dtype=np.float32),
        alpha=float(config["alpha"]),
        s_max=float(config["s_max"]),
        normalize_mass=bool(config["normalize_mass"]),
    )

    new_weights = (np.asarray(base_weights, dtype=np.float32) * multipliers).astype(np.float32)

    stats["multiplier_mean"] = float(np.mean(multipliers))
    stats["multiplier_min"] = float(np.min(multipliers))
    stats["multiplier_max"] = float(np.max(multipliers))
    stats["multiplier_p90"] = float(np.percentile(multipliers, 90))
    stats["base_mass"] = float(np.sum(base_weights, dtype=np.float64))
    stats["new_mass"] = float(np.sum(new_weights, dtype=np.float64))

    print(f"\nWeight Multiplier Statistics (alpha={config['alpha']}, s_max={config['s_max']}):")
    print(f"  Multiplier Mean:   {stats['multiplier_mean']:.5f}")
    print(f"  Multiplier Min:    {stats['multiplier_min']:.5f}")
    print(f"  Multiplier Max:    {stats['multiplier_max']:.5f}")
    print(f"  Multiplier 90th %: {stats['multiplier_p90']:.5f}")
    print(f"  Base Total Mass:   {stats['base_mass']:.2f}")
    print(f"  New Total Mass:    {stats['new_mass']:.2f}")

    out_weights = config.get("output_weights")
    if out_weights:
        out_p = Path(out_weights)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_p, new_weights)
        print(f"Saved modified weights to: {out_p}")

    out_kl = config.get("output_kl")
    if out_kl:
        out_kl_p = Path(out_kl)
        out_kl_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_kl_p, kl_results)
        print(f"Saved raw KL divergence array to: {out_kl_p}")

    out_dataset = config.get("output_dataset_dir")
    if out_dataset:
        create_dataset_variant(Path(config["data"]), Path(out_dataset), new_weights, multipliers)

    return {
        "stats": stats,
        "kl": kl_results,
        "multipliers": multipliers,
        "new_weights": new_weights,
    }


compute_surprise_weights = evaluate_dataset_surprise


def create_dataset_variant(
    source_dir: Path,
    target_dir: Path,
    new_weights: np.ndarray,
    multipliers: np.ndarray,
) -> None:
    """Create a verified memory-map dataset variant with modified weights.

    Invariant arrays (state features, targets, policy) are linked via hardlinks
    to avoid disk duplication. The modified weight and source mass arrays are
    written afresh and re-hashed into a valid dataset.manifest.json.
    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    source_manifest_path = source_dir / "dataset.manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError(f"Source dataset lacks dataset.manifest.json: {source_dir}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    target_dir.mkdir(parents=True, exist_ok=True)
    target_fields = target_dir / "fields"
    target_fields.mkdir(parents=True, exist_ok=True)

    source_fields = source_dir / "fields" if (source_dir / "fields").is_dir() else source_dir
    new_manifest = copy.deepcopy(source_manifest)

    has_source_mass = "source_mass" in source_manifest.get("arrays", {})
    if has_source_mass:
        src_sm = source_fields / Path(source_manifest["arrays"]["source_mass"]["filename"]).name
        base_sm = np.load(src_sm, mmap_mode="r")
        new_sm = (base_sm.astype(np.float64) * multipliers[:, None].astype(np.float64)).astype(np.float32)
        tgt_sm = target_fields / "source_mass.npy"
        np.save(tgt_sm, new_sm)

        h_sm = hashlib.sha256()
        with open(tgt_sm, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h_sm.update(block)
        new_manifest["arrays"]["source_mass"]["sha256"] = h_sm.hexdigest()
        new_manifest["arrays"]["source_mass"]["filename"] = f"fields{os.sep}source_mass.npy"
        new_weights = new_sm.sum(axis=1, dtype=np.float32)

    tgt_w = target_fields / "weight.npy"
    np.save(tgt_w, new_weights)
    h_w = hashlib.sha256()
    with open(tgt_w, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h_w.update(block)
    new_manifest["arrays"]["weight"]["sha256"] = h_w.hexdigest()
    new_manifest["arrays"]["weight"]["filename"] = f"fields{os.sep}weight.npy"

    for key, entry in source_manifest["arrays"].items():
        if key in ("weight", "source_mass"):
            continue
        rel_fn = Path(entry["filename"]).name
        src_file = source_fields / rel_fn
        tgt_file = target_fields / rel_fn
        if tgt_file.exists():
            tgt_file.unlink()
        try:
            os.link(src_file, tgt_file)
        except OSError:
            import shutil
            shutil.copyfile(src_file, tgt_file)
        new_manifest["arrays"][key]["filename"] = f"fields{os.sep}{rel_fn}"

    target_manifest_path = target_dir / "dataset.manifest.json"
    target_manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print(f"Created complete dataset variant at: {target_dir}")


def main() -> None:
    """Execute surprise calculation workflow."""
    config = parse_config()
    evaluate_dataset_surprise(config)


if __name__ == "__main__":
    main()
