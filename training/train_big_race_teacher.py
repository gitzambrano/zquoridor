#!/usr/bin/env python3
"""Train the experimental 1340->512 race-aware NNUE from teacher targets.

This is intentionally separate from train_nnue.py: the experiment is an
architecture break, so it must not silently load or overwrite production
354->256 weights.  It consumes the compact datasets produced by
build_teacher_soft.py and exports both float and quantized weights in the exact
layout expected by the patched src/nnue.hpp.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 9
WS = 8
DIST_BUCKETS = 21
WALLS_PER_PLAYER = 10
WALLS_LEFT_BUCKETS = 11
BASE_NUM_FEATURES = 354
JOINT = DIST_BUCKETS * WALLS_LEFT_BUCKETS
OWN_DIST_OWN_WALL_BASE = BASE_NUM_FEATURES
OWN_DIST_OPP_WALL_BASE = OWN_DIST_OWN_WALL_BASE + JOINT
OPP_DIST_OWN_WALL_BASE = OWN_DIST_OPP_WALL_BASE + JOINT
OPP_DIST_OPP_WALL_BASE = OPP_DIST_OWN_WALL_BASE + JOINT
RACE_MARGIN_BUCKETS = 2 * DIST_BUCKETS - 1
RACE_MARGIN_BASE = OPP_DIST_OPP_WALL_BASE + JOINT
WALL_MARGIN_BUCKETS = 2 * WALLS_PER_PLAYER + 1
WALL_MARGIN_BASE = RACE_MARGIN_BASE + RACE_MARGIN_BUCKETS
NUM_FEATURES = WALL_MARGIN_BASE + WALL_MARGIN_BUCKETS  # 1340
HIDDEN = 512
VALUE_HIDDEN = 32
POLICY_OUT = 209
QA = 255
QB = 64
INT8_MAX = 127
INT16_MAX = 32767

REQUIRED = {
    "own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own",
    "walls_left_opp", "own_dist", "opp_dist", "policy", "value", "weight", "is_val",
}


class BigRaceNNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.value1_wl = nn.Linear(HIDDEN, VALUE_HIDDEN)
        self.value2_wl = nn.Linear(VALUE_HIDDEN, 1)
        self.policy = nn.Linear(HIDDEN, POLICY_OUT)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.clamp(self.fc1(x), 0.0, 1.0).square()
        vh = torch.clamp(self.value1_wl(a), 0.0, 1.0)
        value = self.value2_wl(vh).squeeze(1)
        policy = self.policy(a)
        return value, policy


def dense_features(data: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    n = len(idx)
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    rows = np.arange(n)
    own_pawn = data["own_pawn"][idx].astype(np.int64)
    opp_pawn = data["opp_pawn"][idx].astype(np.int64)
    x[rows, own_pawn] = 1.0
    x[rows, 81 + opp_pawn] = 1.0

    walls_h = data["walls_h"][idx].astype(np.uint64)
    walls_v = data["walls_v"][idx].astype(np.uint64)
    bits = np.arange(64, dtype=np.uint64)
    x[:, 162:226] = ((walls_h[:, None] >> bits) & 1).astype(np.float32)
    x[:, 226:290] = ((walls_v[:, None] >> bits) & 1).astype(np.float32)

    od = np.minimum(data["own_dist"][idx].astype(np.int64), DIST_BUCKETS - 1)
    pd = np.minimum(data["opp_dist"][idx].astype(np.int64), DIST_BUCKETS - 1)
    ow = np.clip(data["walls_left_own"][idx].astype(np.int64), 0, WALLS_PER_PLAYER)
    pw = np.clip(data["walls_left_opp"][idx].astype(np.int64), 0, WALLS_PER_PLAYER)

    x[rows, 290 + od] = 1.0
    x[rows, 290 + DIST_BUCKETS + pd] = 1.0
    wall_base = 290 + 2 * DIST_BUCKETS
    x[rows, wall_base + ow] = 1.0
    x[rows, wall_base + WALLS_LEFT_BUCKETS + pw] = 1.0

    x[rows, OWN_DIST_OWN_WALL_BASE + od * WALLS_LEFT_BUCKETS + ow] = 1.0
    x[rows, OWN_DIST_OPP_WALL_BASE + od * WALLS_LEFT_BUCKETS + pw] = 1.0
    x[rows, OPP_DIST_OWN_WALL_BASE + pd * WALLS_LEFT_BUCKETS + ow] = 1.0
    x[rows, OPP_DIST_OPP_WALL_BASE + pd * WALLS_LEFT_BUCKETS + pw] = 1.0
    race = np.clip(pd - od, -(DIST_BUCKETS - 1), DIST_BUCKETS - 1) + (DIST_BUCKETS - 1)
    wall = np.clip(ow - pw, -WALLS_PER_PLAYER, WALLS_PER_PLAYER) + WALLS_PER_PLAYER
    x[rows, RACE_MARGIN_BASE + race] = 1.0
    x[rows, WALL_MARGIN_BASE + wall] = 1.0
    return x


def load_one(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    missing = sorted(REQUIRED - set(raw.files))
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    n = len(raw["policy"])
    if raw["policy"].shape != (n, POLICY_OUT):
        raise ValueError(f"{path}: invalid policy shape {raw['policy'].shape}")
    if any(len(raw[name]) != n for name in REQUIRED):
        raise ValueError(f"{path}: inconsistent array lengths")
    return {name: raw[name] for name in raw.files}


def concatenate(parts: list[tuple[dict[str, np.ndarray], float]]) -> dict[str, np.ndarray]:
    keys = sorted(REQUIRED)
    out = {k: np.concatenate([p[k] for p, _ in parts], axis=0) for k in keys}
    source_weight = np.concatenate([
        np.full(len(p["policy"]), mult, dtype=np.float32) for p, mult in parts
    ])
    out["weight"] = out["weight"].astype(np.float32) * source_weight

    # The known production failure is concentrated in wall-poor states.
    ow = out["walls_left_own"].astype(np.int16)
    pw = out["walls_left_opp"].astype(np.int16)
    low = np.minimum(ow, pw) <= 2
    zero = np.minimum(ow, pw) == 0
    out["weight"][low] *= 2.5
    out["weight"][zero] *= 1.6
    return out


def weighted_mean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (v * w).sum() / w.sum().clamp_min(1e-8)


def clip_qat(model: BigRaceNNUE) -> None:
    with torch.no_grad():
        acc_limit = INT16_MAX / QA
        head_limit = INT8_MAX / QB
        model.fc1.weight.clamp_(-acc_limit, acc_limit)
        model.fc1.bias.clamp_(-acc_limit, acc_limit)
        model.value1_wl.weight.clamp_(-head_limit, head_limit)
        model.value2_wl.weight.clamp_(-head_limit, head_limit)
        model.policy.weight.clamp_(-head_limit, head_limit)


def run_epoch(model: BigRaceNNUE, data: dict[str, np.ndarray], indices: np.ndarray,
              batch_size: int, device: torch.device, value_weight: float,
              optimizer=None, rng=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    order = indices if not training else rng.permutation(indices)
    totals = dict(loss=0.0, policy_ce=0.0, policy_kl=0.0, value_bce=0.0,
                  top1=0.0, value_mae=0.0, lowwall_value_mae=0.0,
                  lowwall_weight=0.0, weight=0.0)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x = torch.from_numpy(dense_features(data, idx)).to(device)
        tp = torch.from_numpy(data["policy"][idx].astype(np.float32)).to(device)
        tv_signed = torch.from_numpy(data["value"][idx].astype(np.float32)).to(device)
        tv = (tv_signed + 1.0) * 0.5
        w = torch.from_numpy(data["weight"][idx].astype(np.float32)).to(device)
        low_np = (np.minimum(data["walls_left_own"][idx], data["walls_left_opp"][idx]) <= 2)
        low = torch.from_numpy(low_np.astype(np.float32)).to(device)

        with torch.set_grad_enabled(training):
            value_logits, policy_logits = model(x)
            logp = F.log_softmax(policy_logits, dim=1)
            ce = -(tp * logp).sum(dim=1)
            entropy = -(tp * torch.log(tp.clamp_min(1e-12))).sum(dim=1)
            kl = (ce - entropy).clamp_min(0.0)
            vbce = F.binary_cross_entropy_with_logits(value_logits, tv, reduction="none")
            p_loss = weighted_mean(ce, w)
            v_loss = weighted_mean(vbce, w)
            loss = p_loss + value_weight * v_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                clip_qat(model)

        sw = float(w.sum().item())
        pred_value = 2.0 * torch.sigmoid(value_logits) - 1.0
        mae = torch.abs(pred_value - tv_signed)
        top1 = (policy_logits.argmax(1) == tp.argmax(1)).float()
        totals["weight"] += sw
        totals["loss"] += float(loss.item()) * sw
        totals["policy_ce"] += float((ce * w).sum().item())
        totals["policy_kl"] += float((kl * w).sum().item())
        totals["value_bce"] += float((vbce * w).sum().item())
        totals["top1"] += float((top1 * w).sum().item())
        totals["value_mae"] += float((mae * w).sum().item())
        lw = w * low
        totals["lowwall_weight"] += float(lw.sum().item())
        totals["lowwall_value_mae"] += float((mae * lw).sum().item())

    d = max(totals["weight"], 1e-8)
    ld = max(totals["lowwall_weight"], 1e-8)
    return {
        "loss": totals["loss"] / d,
        "policy_ce": totals["policy_ce"] / d,
        "policy_kl": totals["policy_kl"] / d,
        "value_bce": totals["value_bce"] / d,
        "top1": totals["top1"] / d,
        "value_mae": totals["value_mae"] / d,
        "lowwall_value_mae": totals["lowwall_value_mae"] / ld,
        "weight": totals["weight"],
        "lowwall_weight": totals["lowwall_weight"],
    }


def write_float(model: BigRaceNNUE, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        def wr(a: np.ndarray) -> None:
            f.write(np.ascontiguousarray(a.astype("<f4")).tobytes())
        wr(model.fc1.weight.detach().cpu().numpy().T)
        wr(model.fc1.bias.detach().cpu().numpy())
        wr(model.value1_wl.weight.detach().cpu().numpy().T)
        wr(model.value1_wl.bias.detach().cpu().numpy())
        wr(model.value2_wl.weight.detach().cpu().numpy().reshape(-1))
        wr(model.value2_wl.bias.detach().cpu().numpy())
        wr(model.policy.weight.detach().cpu().numpy())
        wr(model.policy.bias.detach().cpu().numpy())


def q16(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(x * scale), -32768, 32767).astype("<i2")


def q8(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(x * scale), -128, 127).astype("i1")


def write_quantized(model: BigRaceNNUE, path: Path) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fcw = model.fc1.weight.detach().cpu().numpy().T
    fcb = model.fc1.bias.detach().cpu().numpy()
    v1w = model.value1_wl.weight.detach().cpu().numpy().T
    v1b = model.value1_wl.bias.detach().cpu().numpy()
    v2w = model.value2_wl.weight.detach().cpu().numpy().reshape(-1)
    v2b = float(model.value2_wl.bias.detach().cpu().numpy()[0])
    pw = model.policy.weight.detach().cpu().numpy()
    pb = model.policy.bias.detach().cpu().numpy()
    sat = {
        "w1": int(np.sum(np.abs(np.rint(fcw * QA)) > 32767)),
        "v1": int(np.sum(np.abs(np.rint(v1w * QB)) > 127)),
        "v2": int(np.sum(np.abs(np.rint(v2w * QB)) > 127)),
        "policy": int(np.sum(np.abs(np.rint(pw * QB)) > 127)),
    }
    with path.open("wb") as f:
        f.write(np.asarray([QA, QB], dtype="<i4").tobytes())
        f.write(q16(fcw, QA).tobytes())
        f.write(q16(fcb, QA).tobytes())
        f.write(q8(v1w, QB).tobytes())
        f.write(np.rint(v1b * QA * QB).astype("<i4").tobytes())
        f.write(q8(v2w, QB).tobytes())
        f.write(np.asarray([round(v2b * QA * QB * QB)], dtype="<i4").tobytes())
        f.write(q8(pw, QB).tobytes())
        f.write(np.rint(pb * QA * QB).astype("<i4").tobytes())
    return sat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--anchor-data", type=Path)
    ap.add_argument("--anchor-weight", type=float, default=8.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch-size", type=int, default=768)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--lr-min", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if min(args.epochs, args.batch_size, args.patience) <= 0 or args.lr <= 0 or args.anchor_weight <= 0:
        raise SystemExit("invalid positive training parameter")

    parts = [(load_one(args.data), 1.0)]
    if args.anchor_data:
        parts.append((load_one(args.anchor_data), args.anchor_weight))
    data = concatenate(parts)
    n = len(data["policy"])
    all_idx = np.arange(n, dtype=np.int64)
    val = data["is_val"].astype(bool)
    train_idx, val_idx = all_idx[~val], all_idx[val]
    if not len(train_idx) or not len(val_idx):
        raise SystemExit("dataset must contain both train and val samples")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = BigRaceNNUE().to(device)
    clip_qat(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    with torch.no_grad():
        initial_val = run_epoch(model, data, val_idx, args.batch_size, device, args.value_weight)
    best = math.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, data, train_idx, args.batch_size, device, args.value_weight, optimizer, rng)
        with torch.no_grad():
            va = run_epoch(model, data, val_idx, args.batch_size, device, args.value_weight)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": tr, "val": va}
        history.append(row)
        print(json.dumps(row), flush=True)
        if va["loss"] < best - 1e-5:
            best = va["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        scheduler.step()
        if bad >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)

    float_path = args.out
    int8_path = args.out.with_name(args.out.stem + "_int8.bin")
    write_float(model, float_path)
    saturation = write_quantized(model, int8_path)
    expected_qbytes = 8 + NUM_FEATURES * HIDDEN * 2 + HIDDEN * 2 + HIDDEN * VALUE_HIDDEN + VALUE_HIDDEN * 4 + VALUE_HIDDEN + 4 + POLICY_OUT * HIDDEN + POLICY_OUT * 4
    actual_qbytes = int8_path.stat().st_size
    if actual_qbytes != expected_qbytes:
        raise RuntimeError(f"quantized size {actual_qbytes} != expected {expected_qbytes}")
    if any(saturation.values()):
        raise RuntimeError(f"QAT saturation detected: {saturation}")

    manifest = {
        "schema": "zquoridor.big_race_teacher.v1",
        "architecture": {"features": NUM_FEATURES, "hidden": HIDDEN, "value_hidden": VALUE_HIDDEN, "policy": POLICY_OUT},
        "feature_design": [
            "production-354",
            "own_dist_x_own_walls", "own_dist_x_opp_walls",
            "opp_dist_x_own_walls", "opp_dist_x_opp_walls",
            "race_margin", "wall_margin",
        ],
        "samples": n,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "anchor_samples": 0 if not args.anchor_data else int(len(parts[1][0]["policy"])),
        "anchor_weight": args.anchor_weight if args.anchor_data else 0.0,
        "value_weight": args.value_weight,
        "initial_val": initial_val,
        "best_val_loss": best,
        "epochs_ran": len(history),
        "best_val": min(history, key=lambda r: r["val"]["loss"])["val"],
        "quantized_bytes": actual_qbytes,
        "qat_saturation": saturation,
        "device": str(device),
        "history": history,
    }
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
