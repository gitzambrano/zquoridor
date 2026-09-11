#!/usr/bin/env python3
"""Fine-tune the ZQuoridor policy head from an external teacher dataset.

The default mode updates only the 256->209 policy head. The NNUE accumulator
and value head remain bit-for-bit unchanged, which isolates the first teaching
experiment from value-network and feature-representation changes. Use
--train-trunk only for a later, explicitly broader experiment.
"""
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


def dense_features(data, indices: np.ndarray) -> np.ndarray:
    """Expand compact canonical teacher states to the current 354 inputs."""
    n = len(indices)
    x = np.zeros((n, base.NUM_FEATURES), dtype=np.float32)
    rows = np.arange(n)
    own_pawn = data["own_pawn"][indices].astype(np.int64)
    opp_pawn = data["opp_pawn"][indices].astype(np.int64)
    x[rows, own_pawn] = 1.0
    x[rows, 81 + opp_pawn] = 1.0

    walls_h = data["walls_h"][indices].astype(np.uint64)
    walls_v = data["walls_v"][indices].astype(np.uint64)
    bits = np.arange(64, dtype=np.uint64)
    x[:, 162:226] = ((walls_h[:, None] >> bits) & 1).astype(np.float32)
    x[:, 226:290] = ((walls_v[:, None] >> bits) & 1).astype(np.float32)

    own_dist = np.minimum(
        data["own_dist"][indices].astype(np.int64), base.DIST_BUCKETS - 1
    )
    opp_dist = np.minimum(
        data["opp_dist"][indices].astype(np.int64), base.DIST_BUCKETS - 1
    )
    own_walls = np.minimum(
        data["walls_left_own"][indices].astype(np.int64),
        base.WALLS_LEFT_BUCKETS - 1,
    )
    opp_walls = np.minimum(
        data["walls_left_opp"][indices].astype(np.int64),
        base.WALLS_LEFT_BUCKETS - 1,
    )
    x[rows, 290 + own_dist] = 1.0
    x[rows, 290 + base.DIST_BUCKETS + opp_dist] = 1.0
    wall_base = 290 + 2 * base.DIST_BUCKETS
    x[rows, wall_base + own_walls] = 1.0
    x[rows, wall_base + base.WALLS_LEFT_BUCKETS + opp_walls] = 1.0
    return x


def split_indices(data, seed: int, fallback_val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Use opening-level collector splits and fall back only if one is empty."""
    n = len(data["policy_idx"])
    all_indices = np.arange(n, dtype=np.int64)
    is_val = data["is_val"].astype(bool)
    train_indices = all_indices[~is_val]
    val_indices = all_indices[is_val]
    if len(train_indices) and len(val_indices):
        return train_indices, val_indices

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, min(n - 1, int(round(n * fallback_val_fraction))))
    if n < 2:
        raise ValueError("teacher policy training needs at least two samples")
    return perm[n_val:], perm[:n_val]


def weighted_policy_loss(logits: torch.Tensor, targets: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    """Return confidence-weighted teacher cross entropy."""
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)


def run_epoch(model, data, indices: np.ndarray, batch_size: int, device,
              optimizer=None, rng=None, qa=base.QA_DEFAULT, qb=base.QB_DEFAULT) -> dict:
    """Run one train or validation epoch."""
    training = optimizer is not None
    model.train(training)
    order = indices if not training else rng.permutation(indices)
    total_weight = 0.0
    total_loss = 0.0
    correct1 = 0.0
    correct3 = 0.0

    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]
        x = torch.from_numpy(dense_features(data, batch_indices)).to(device)
        targets = torch.from_numpy(
            data["policy_idx"][batch_indices].astype(np.int64)
        ).to(device)
        weights = torch.from_numpy(
            data["weight"][batch_indices].astype(np.float32)
        ).to(device)

        with torch.set_grad_enabled(training):
            _, logits = model(x)
            loss = weighted_policy_loss(logits, targets, weights)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.policy.weight.clamp_(
                        -base.INT8_MAX / qb,
                        base.INT8_MAX / qb,
                    )
                    if model.fc1.weight.requires_grad:
                        limit = base.INT16_MAX / qa
                        model.fc1.weight.clamp_(-limit, limit)
                        model.fc1.bias.clamp_(-limit, limit)

        batch_weight = float(weights.sum().item())
        total_weight += batch_weight
        total_loss += float(loss.item()) * batch_weight
        top1 = logits.argmax(dim=1)
        correct1 += float(((top1 == targets).float() * weights).sum().item())
        top3 = logits.topk(k=3, dim=1).indices
        hits3 = (top3 == targets[:, None]).any(dim=1).float()
        correct3 += float((hits3 * weights).sum().item())

    denom = max(total_weight, 1e-8)
    return {
        "loss": total_loss / denom,
        "top1": correct1 / denom,
        "top3": correct3 / denom,
        "weight": total_weight,
    }


def frozen_state(model) -> dict[str, torch.Tensor]:
    """Snapshot parameters that head-only training must not change."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("policy.")
    }


def verify_frozen(model, before: dict[str, torch.Tensor]) -> None:
    """Fail if head-only distillation changed accumulator or value weights."""
    after = model.state_dict()
    changed = [
        name for name, tensor in before.items()
        if not torch.equal(tensor, after[name].detach().cpu())
    ]
    if changed:
        raise RuntimeError(
            "head-only teacher training changed frozen parameters: " + ", ".join(changed)
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Fine-tune a network from teacher policy labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qa", type=int, default=base.QA_DEFAULT)
    parser.add_argument("--qb", type=int, default=base.QB_DEFAULT)
    parser.add_argument("--train-trunk", action="store_true")
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0.0:
        raise SystemExit("epochs, batch-size, and lr must be positive")
    if args.patience <= 0 or not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("patience must be positive and val-fraction must be in (0,1)")

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    data = np.load(args.data, allow_pickle=False)
    required = {
        "own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own",
        "walls_left_opp", "own_dist", "opp_dist", "policy_idx", "weight", "is_val",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise SystemExit(f"teacher dataset lacks fields: {missing}")
    n = len(data["policy_idx"])
    if any(len(data[name]) != n for name in required):
        raise SystemExit("teacher dataset arrays have inconsistent lengths")
    if np.any(data["policy_idx"] >= base.POLICY_OUT):
        raise SystemExit("teacher dataset contains an invalid policy index")

    try:
        train_indices, val_indices = split_indices(data, args.seed, args.val_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    model = base.QuoridorNNUE().to(device)
    raw = base._load_raw_weights(args.init_from)
    base._load_into_model(model, raw)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    if args.train_trunk:
        for parameter in model.fc1.parameters():
            parameter.requires_grad_(True)

    frozen = None if args.train_trunk else frozen_state(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, data, train_indices, args.batch_size, device,
            optimizer=optimizer, rng=np_rng, qa=args.qa, qb=args.qb,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, data, val_indices, args.batch_size, device,
                optimizer=None, rng=np_rng, qa=args.qa, qb=args.qb,
            )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best_loss - 1e-7:
            best_loss = val_metrics["loss"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("teacher training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    if frozen is not None:
        verify_frozen(model, frozen)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.export_weights(model, str(out))
    quant_path = out.with_name(out.stem + "_int8.bin")
    quantize_file(str(out), str(quant_path), qa=args.qa, qb=args.qb)

    manifest = {
        "schema": "zquoridor.teacher.policy_train.v1",
        "data": os.path.abspath(args.data),
        "init_from": os.path.abspath(args.init_from),
        "out": os.path.abspath(str(out)),
        "quantized_out": os.path.abspath(str(quant_path)),
        "samples": n,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "mode": "trunk+policy" if args.train_trunk else "policy-head-only",
        "frozen_value_and_trunk_verified": not args.train_trunk,
        "best_val_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
        "qa": args.qa,
        "qb": args.qb,
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
