#!/usr/bin/env python3
"""Fine-tune only the Gen8 policy head on deep reanalysis targets.

This deliberately freezes the 354->256 trunk and the complete value head.
The experiment is meant to alter root/search priors for wall timing while
preserving the Gen8 evaluation representation that already has proven Elo.
Only V3 samples with a non-zero root visit distribution contribute loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from read_selfplay import load_multi_selfplay  # noqa: E402
from train_nnue import (  # noqa: E402
    QuoridorNNUE,
    WeightClipper,
    _load_into_model,
    _load_raw_weights,
    export_weights,
    to_chunk_tensors,
)
from quantize_nnue import quantize_file  # noqa: E402


def valid_indices(ds) -> np.ndarray:
    chunks = []
    offset = 0
    for path, n in ds.sizes():
        # Read lazily through the dataset abstraction so format handling stays
        # identical to the normal trainer.
        arr = ds[np.arange(offset, offset + n)]
        local = np.flatnonzero(arr["policy_top_prob"][:, 0] > 0)
        if len(local):
            chunks.append(offset + local)
        offset += n
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def soft_visit_loss(logits: torch.Tensor, idx: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(logp, 1, idx)
    mass = prob.sum(dim=1).clamp_min(1e-12)
    target = prob / mass[:, None]
    return -(target * gathered).sum(dim=1).mean()


def evaluate(model, ds, idx, device, batch_size):
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            take = idx[start:start + batch_size]
            chunk = ds[take]
            k = np.ones(len(chunk), dtype=np.float32)
            t = to_chunk_tensors(chunk, k, device)
            _, logits = model(t["x"])
            loss = soft_visit_loss(logits, t["policy_top_idx"], t["policy_top_prob"])
            n = len(chunk)
            total_loss += loss.item() * n
            top1 += (logits.argmax(dim=1) == t["policy_top_idx"][:, 0]).sum().item()
            total += n
    return total_loss / max(1, total), top1 / max(1, total)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="append", required=True)
    p.add_argument("--val", action="append", required=True)
    p.add_argument("--init-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    _, train_ds = load_multi_selfplay(args.train)
    _, val_ds = load_multi_selfplay(args.val)
    train_idx = valid_indices(train_ds)
    val_idx = valid_indices(val_ds)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise SystemExit(f"need non-empty deep policy targets: train={len(train_idx)} val={len(val_idx)}")

    model = QuoridorNNUE().to(device)
    _load_into_model(model, _load_raw_weights(args.init_from))

    # Preserve every proven Gen8 parameter except the direct policy head.
    for param in model.parameters():
        param.requires_grad_(False)
    model.policy.weight.requires_grad_(True)
    model.policy.bias.requires_grad_(True)

    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.policy.parameters(), lr=args.lr, weight_decay=1e-5)
    clipper = WeightClipper()
    rng = np.random.default_rng(args.seed)

    best = None
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        frac = (epoch - 1) / max(1, args.epochs - 1)
        lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + np.cos(np.pi * frac))
        for group in opt.param_groups:
            group["lr"] = float(lr)

        model.train()
        shuffled = rng.permutation(train_idx)
        total_loss = 0.0
        total = 0
        for start in range(0, len(shuffled), args.batch_size):
            take = shuffled[start:start + args.batch_size]
            chunk = train_ds[take]
            k = np.ones(len(chunk), dtype=np.float32)
            t = to_chunk_tensors(chunk, k, device)
            _, logits = model(t["x"])
            loss = soft_visit_loss(logits, t["policy_top_idx"], t["policy_top_prob"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            opt.step()
            clipper(model)
            total_loss += loss.item() * len(chunk)
            total += len(chunk)

        val_loss, val_acc = evaluate(model, val_ds, val_idx, device, args.batch_size)
        tr = total_loss / max(1, total)
        history.append({"epoch": epoch, "lr": float(lr), "train_policy": tr,
                        "val_policy": val_loss, "val_top1": val_acc})
        print(f"epoch {epoch:3d}/{args.epochs} lr={lr:.2e} train_policy={tr:.5f} "
              f"val_policy={val_loss:.5f} val_top1={val_acc:.3f}", flush=True)
        if best is None or val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    # Hard invariant: trunk and value parameters must be bit-identical to Gen8.
    changed = []
    for name, value in model.state_dict().items():
        if name.startswith("policy."):
            continue
        if not torch.equal(value.detach().cpu(), initial[name]):
            changed.append(name)
    if changed:
        raise RuntimeError(f"policy-only invariant violated; non-policy tensors changed: {changed}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_weights(model, str(out))
    qout = out.with_name(out.stem + "_int8.bin")
    quantize_file(str(out), str(qout))
    (out.parent / "policy_reanalysis_history.json").write_text(
        json.dumps({"train_positions": int(len(train_idx)), "val_positions": int(len(val_idx)),
                    "best_val_policy": float(best), "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"policy-only export: {out} and {qout}; train={len(train_idx)} val={len(val_idx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
