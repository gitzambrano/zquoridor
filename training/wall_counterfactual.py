#!/usr/bin/env python3
"""Counterfactual benchmark for whether the NNUE understands wall resources.

For a fixed legal board, pawn locations, distances, and side to move, having
one extra wall available cannot make that player worse in game-theoretic
value: the player can always choose not to spend it. Conversely, giving the
opponent an extra wall cannot improve our value. These are useful conceptual
invariants that do not require a handcrafted evaluation or a second teacher.

This tool loads the raw float32 NNUE weights, samples real V3 self-play states,
changes only the walls-left one-hot features, and measures monotonicity plus
the network's marginal sensitivity to wall resources.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from read_selfplay import SAMPLE_DTYPE

N = 9
WS = 8
DIST_BUCKETS = 21
WALL_BUCKETS = 11
NUM_FEATURES = 354
HIDDEN = 256
POLICY_OUT = 209
DIST_BASE = 290
WALL_BASE = 332


def expand_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.bin")))
        elif any(c in item for c in "*?["):
            out.extend(Path(x) for x in sorted(glob.glob(item)))
        elif p.is_file():
            out.append(p)
    if not out:
        raise SystemExit("no V3 data found")
    return out


def load_weights(path: Path):
    x = np.fromfile(path, dtype="<f4")
    expected = (
        NUM_FEATURES * HIDDEN + HIDDEN
        + HIDDEN * 32 + 32 + 32 + 1
        + POLICY_OUT * HIDDEN + POLICY_OUT
    )
    if len(x) != expected:
        raise SystemExit(f"{path}: {len(x)} floats, expected {expected}")
    k = 0
    w1 = x[k:k + NUM_FEATURES * HIDDEN].reshape(NUM_FEATURES, HIDDEN); k += NUM_FEATURES * HIDDEN
    b1 = x[k:k + HIDDEN]; k += HIDDEN
    wv1 = x[k:k + HIDDEN * 32].reshape(HIDDEN, 32); k += HIDDEN * 32
    bv1 = x[k:k + 32]; k += 32
    wv2 = x[k:k + 32]; k += 32
    bv2 = float(x[k]); k += 1
    # policy weights are not needed for the value monotonicity benchmark.
    return w1.astype(np.float64), b1.astype(np.float64), wv1.astype(np.float64), bv1.astype(np.float64), wv2.astype(np.float64), bv2


def bit_indices(v: int):
    i = 0
    while v:
        if v & 1:
            yield i
        v >>= 1
        i += 1


def base_acc(sample, w1, b1):
    acc = b1.copy()
    acc += w1[int(sample["own_pawn"])]
    acc += w1[N * N + int(sample["opp_pawn"])]
    for i in bit_indices(int(sample["walls_h"])):
        acc += w1[2 * N * N + i]
    for i in bit_indices(int(sample["walls_v"])):
        acc += w1[2 * N * N + WS * WS + i]
    od = min(int(sample["own_dist"]), DIST_BUCKETS - 1)
    pd = min(int(sample["opp_dist"]), DIST_BUCKETS - 1)
    acc += w1[DIST_BASE + od]
    acc += w1[DIST_BASE + DIST_BUCKETS + pd]
    return acc


def value_from_acc(acc, own_walls: int, opp_walls: int, w1, wv1, bv1, wv2, bv2):
    z = acc + w1[WALL_BASE + own_walls] + w1[WALL_BASE + WALL_BUCKETS + opp_walls]
    a = np.clip(z, 0.0, 1.0) ** 2
    h = np.clip(a @ wv1 + bv1, 0.0, 1.0)
    logit = float(h @ wv2 + bv2)
    # Stable enough for observed logit scale.
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def sample_records(paths: list[Path], n: int, seed: int):
    arrays = []
    lengths = []
    for p in paths:
        size = p.stat().st_size
        if size == 0 or size % SAMPLE_DTYPE.itemsize:
            continue
        a = np.memmap(p, dtype=SAMPLE_DTYPE, mode="r")
        arrays.append(a)
        lengths.append(len(a))
    total = sum(lengths)
    if total == 0:
        raise SystemExit("no valid V3 records")
    rng = np.random.default_rng(seed)
    ids = rng.choice(total, size=min(n, total), replace=False)
    offsets = np.cumsum([0] + lengths)
    for gid in ids:
        fi = int(np.searchsorted(offsets, gid, side="right") - 1)
        yield arrays[fi][int(gid - offsets[fi])]


def summarize(vals: list[float]):
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)), "p90": float(np.quantile(x, 0.90)),
        "min": float(x.min()), "max": float(x.max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--tolerance", type=float, default=1e-7)
    p.add_argument("--json")
    args = p.parse_args()

    paths = expand_inputs(args.inputs)
    w1, b1, wv1, bv1, wv2, bv2 = load_weights(Path(args.weights))
    own_viol = 0
    opp_viol = 0
    own_steps = 0
    opp_steps = 0
    own_deltas: list[float] = []
    opp_deltas: list[float] = []
    own_total_swing: list[float] = []
    opp_total_swing: list[float] = []
    worst_own_drop = 0.0
    worst_opp_rise = 0.0
    used = 0

    for s in sample_records(paths, args.samples, args.seed):
        acc = base_acc(s, w1, b1)
        fixed_opp = int(np.clip(int(s["walls_left_opp"]), 0, 10))
        fixed_own = int(np.clip(int(s["walls_left_own"]), 0, 10))
        own_curve = np.asarray([
            value_from_acc(acc, w, fixed_opp, w1, wv1, bv1, wv2, bv2)
            for w in range(11)
        ])
        opp_curve = np.asarray([
            value_from_acc(acc, fixed_own, w, w1, wv1, bv1, wv2, bv2)
            for w in range(11)
        ])
        d_own = np.diff(own_curve)
        d_opp = np.diff(opp_curve)
        own_viol += int(np.sum(d_own < -args.tolerance))
        opp_viol += int(np.sum(d_opp > args.tolerance))
        own_steps += len(d_own)
        opp_steps += len(d_opp)
        own_deltas.extend(float(x) for x in d_own)
        # Benefit to the opponent is -our value delta; positive is sensible.
        opp_deltas.extend(float(-x) for x in d_opp)
        own_total_swing.append(float(own_curve[-1] - own_curve[0]))
        opp_total_swing.append(float(opp_curve[0] - opp_curve[-1]))
        worst_own_drop = min(worst_own_drop, float(d_own.min()))
        worst_opp_rise = max(worst_opp_rise, float(d_opp.max()))
        used += 1

    result = {
        "weights": str(args.weights),
        "samples": used,
        "own_wall_monotonicity": {
            "violations": own_viol,
            "steps": own_steps,
            "violation_pct": 100.0 * own_viol / own_steps if own_steps else 0.0,
            "worst_probability_drop": worst_own_drop,
            "marginal_value_per_extra_wall": summarize(own_deltas),
            "value_swing_0_to_10_walls": summarize(own_total_swing),
        },
        "opponent_wall_monotonicity": {
            "violations": opp_viol,
            "steps": opp_steps,
            "violation_pct": 100.0 * opp_viol / opp_steps if opp_steps else 0.0,
            "worst_probability_rise_for_us": worst_opp_rise,
            "marginal_cost_per_extra_opponent_wall": summarize(opp_deltas),
            "value_swing_0_to_10_opponent_walls": summarize(opp_total_swing),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
