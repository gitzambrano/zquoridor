#!/usr/bin/env python3
"""Distill full soft policy/value targets into the current ZQuoridor NNUE."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

import train_nnue as base
from quantize_nnue import quantize_file
from train_teacher_policy import dense_features


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp(min=1e-8)


def split_soft_indices(data, seed: int, fallback_val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Use collector train/val flags for a full-soft-policy dataset.

    This intentionally does not reuse train_teacher_policy.split_indices():
    that helper is for the older one-hot dataset and keys its length from
    ``policy_idx``. Soft distillation stores one 209-way row in ``policy``.
    """
    n = len(data["policy"])
    if n < 2:
        raise ValueError("soft teacher training needs at least two samples")
    all_indices = np.arange(n, dtype=np.int64)
    is_val = data["is_val"].astype(bool)
    train_indices = all_indices[~is_val]
    val_indices = all_indices[is_val]
    if len(train_indices) and len(val_indices):
        return train_indices, val_indices

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, min(n - 1, int(round(n * fallback_val_fraction))))
    return perm[n_val:], perm[:n_val]


def soft_policy_loss(logits: torch.Tensor, target: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted cross entropy against the full teacher distribution."""
    per_sample = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    return weighted_mean(per_sample, weights)


def run_epoch(model, data, indices: np.ndarray, batch_size: int, device,
              value_weight: float, optimizer=None, rng=None,
              qa=base.QA_DEFAULT, qb=base.QB_DEFAULT) -> dict:
    training = optimizer is not None
    model.train(training)
    order = indices if not training else rng.permutation(indices)
    total_weight = 0.0
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "top1": 0.0, "value_mae": 0.0}

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x = torch.from_numpy(dense_features(data, idx)).to(device)
        target_policy = torch.from_numpy(data["policy"][idx].astype(np.float32)).to(device)
        target_value_signed = torch.from_numpy(data["value"][idx].astype(np.float32)).to(device)
        target_value_prob = (target_value_signed + 1.0) * 0.5
        weights = torch.from_numpy(data["weight"][idx].astype(np.float32)).to(device)

        with torch.set_grad_enabled(training):
            value_logits, policy_logits = model(x)
            p_loss = soft_policy_loss(policy_logits, target_policy, weights)
            v_per = F.binary_cross_entropy_with_logits(
                value_logits, target_value_prob, reduction="none"
            )
            v_loss = weighted_mean(v_per, weights)
            loss = p_loss + value_weight * v_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    head_limit = base.INT8_MAX / qb
                    acc_limit = base.INT16_MAX / qa
                    model.policy.weight.clamp_(-head_limit, head_limit)
                    model.value1_wl.weight.clamp_(-head_limit, head_limit)
                    model.value2_wl.weight.clamp_(-head_limit, head_limit)
                    if model.fc1.weight.requires_grad:
                        model.fc1.weight.clamp_(-acc_limit, acc_limit)
                        model.fc1.bias.clamp_(-acc_limit, acc_limit)

        batch_weight = float(weights.sum().item())
        total_weight += batch_weight
        teacher_top1 = target_policy.argmax(dim=1)
        student_top1 = policy_logits.argmax(dim=1)
        top1 = (teacher_top1 == student_top1).float()
        student_value_signed = 2.0 * torch.sigmoid(value_logits) - 1.0
        value_mae = torch.abs(student_value_signed - target_value_signed)
        totals["loss"] += float(loss.item()) * batch_weight
        totals["policy"] += float(p_loss.item()) * batch_weight
        totals["value"] += float(v_loss.item()) * batch_weight
        totals["top1"] += float((top1 * weights).sum().item())
        totals["value_mae"] += float((value_mae * weights).sum().item())

    denom = max(total_weight, 1e-8)
    return {key: value / denom for key, value in totals.items()} | {"weight": total_weight}


def snapshot_frozen(model) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in parameters and not parameters[name].requires_grad
    }


def verify_frozen(model, before: dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    changed = [name for name, value in before.items() if not torch.equal(value, after[name].detach().cpu())]
    if changed:
        raise RuntimeError("distillation changed frozen parameters: " + ", ".join(changed))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qa", type=int, default=base.QA_DEFAULT)
    parser.add_argument("--qb", type=int, default=base.QB_DEFAULT)
    parser.add_argument("--train-value-head", action="store_true")
    parser.add_argument("--train-trunk", action="store_true")
    parser.add_argument("--value-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0.0 or args.patience <= 0:
        raise SystemExit("epochs, batch-size, lr, and patience must be positive")
    if not 0.0 < args.val_fraction < 1.0 or args.value_weight < 0.0:
        raise SystemExit("val-fraction must be in (0,1) and value-weight non-negative")

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    data = np.load(args.data, allow_pickle=False)
    required = {
        "own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own",
        "walls_left_opp", "own_dist", "opp_dist", "policy", "value", "weight", "is_val",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise SystemExit(f"soft teacher dataset lacks fields: {missing}")
    n = len(data["policy"])
    if data["policy"].shape != (n, base.POLICY_OUT):
        raise SystemExit(f"soft teacher policy has invalid shape {data['policy'].shape}")
    if any(len(data[name]) != n for name in required):
        raise SystemExit("soft teacher dataset arrays have inconsistent lengths")
    sums = data["policy"].astype(np.float32).sum(axis=1)
    if not np.all(np.isfinite(sums)) or np.max(np.abs(sums - 1.0)) > 5e-3:
        raise SystemExit("soft teacher policy rows are not normalized")
    if np.any(data["value"] < -1.001) or np.any(data["value"] > 1.001):
        raise SystemExit("soft teacher values fall outside [-1,1]")

    try:
        train_idx, val_idx = split_soft_indices(data, args.seed, args.val_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = base.QuoridorNNUE().to(device)
    base._load_into_model(model, base._load_raw_weights(args.init_from))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    if args.train_value_head:
        for layer in (model.value1_wl, model.value2_wl):
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    if args.train_trunk:
        for parameter in model.fc1.parameters():
            parameter.requires_grad_(True)

    frozen = snapshot_frozen(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    effective_value_weight = args.value_weight if args.train_value_head else 0.0
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, data, train_idx, args.batch_size, device,
            effective_value_weight, optimizer=optimizer, rng=rng, qa=args.qa, qb=args.qb,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, data, val_idx, args.batch_size, device,
                effective_value_weight, optimizer=None, rng=rng, qa=args.qa, qb=args.qb,
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best_loss - 1e-7:
            best_loss = val_metrics["loss"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("soft distillation did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    verify_frozen(model, frozen)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.export_weights(model, str(out))
    quantized = out.with_name(out.stem + "_int8.bin")
    quantize_file(str(out), str(quantized), qa=args.qa, qb=args.qb)
    mode = ["policy"]
    if args.train_value_head:
        mode.append("value")
    if args.train_trunk:
        mode.append("trunk")
    manifest = {
        "schema": "zquoridor.teacher.soft_train.v1",
        "data": os.path.abspath(args.data),
        "init_from": os.path.abspath(args.init_from),
        "out": os.path.abspath(str(out)),
        "quantized_out": os.path.abspath(str(quantized)),
        "samples": n,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "mode": "+".join(mode),
        "value_weight": effective_value_weight,
        "frozen_parameters_verified": sorted(frozen),
        "best_val_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
