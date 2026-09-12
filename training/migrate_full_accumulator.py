#!/usr/bin/env python3
"""Migrate a legacy 256-head NNUE into the 512-wide full-accumulator layout.

The migrated network is functionally identical to the legacy network at
initialization: legacy head weights are copied to the own-perspective half and
the opponent-perspective half is zeroed. The shared 354->256 transformer,
biases, output layer, and policy biases are copied bit-for-bit.

This gives full-accumulator training a safe warm start: before fine-tuning, the
extra opponent view contributes exactly zero; training can then learn how to
use it without sacrificing the production baseline at step zero.
"""
from __future__ import annotations

import argparse
import os
import numpy as np

NUM_FEATURES = 354
HIDDEN = 256
HEAD_INPUT = 512
VALUE_HIDDEN = 32
POLICY_OUT = 209

LEGACY_FLOATS = (
    NUM_FEATURES * HIDDEN + HIDDEN
    + HIDDEN * VALUE_HIDDEN + VALUE_HIDDEN
    + VALUE_HIDDEN + 1
    + POLICY_OUT * HIDDEN + POLICY_OUT
)
FULL_FLOATS = (
    NUM_FEATURES * HIDDEN + HIDDEN
    + HEAD_INPUT * VALUE_HIDDEN + VALUE_HIDDEN
    + VALUE_HIDDEN + 1
    + POLICY_OUT * HEAD_INPUT + POLICY_OUT
)


def take(raw: np.ndarray, offset: int, count: int):
    end = offset + count
    if end > raw.size:
        raise ValueError("truncated legacy weight file")
    return raw[offset:end], end


def migrate(src: str, dst: str) -> None:
    raw = np.fromfile(src, dtype=np.float32)
    if raw.size != LEGACY_FLOATS:
        raise ValueError(
            f"expected legacy layout with {LEGACY_FLOATS} float32 values "
            f"({LEGACY_FLOATS * 4} bytes), got {raw.size} values ({raw.nbytes} bytes)"
        )

    o = 0
    w1, o = take(raw, o, NUM_FEATURES * HIDDEN)
    b1, o = take(raw, o, HIDDEN)
    wv1, o = take(raw, o, HIDDEN * VALUE_HIDDEN)
    bv1, o = take(raw, o, VALUE_HIDDEN)
    wv2, o = take(raw, o, VALUE_HIDDEN)
    bv2, o = take(raw, o, 1)
    wp, o = take(raw, o, POLICY_OUT * HIDDEN)
    bp, o = take(raw, o, POLICY_OUT)
    assert o == raw.size

    # On disk the C++ layout stores one row per accumulator activation.
    # Append a zero opponent row block after the legacy own-perspective block.
    wv1_full = np.zeros((HEAD_INPUT, VALUE_HIDDEN), dtype=np.float32)
    wv1_full[:HIDDEN, :] = wv1.reshape(HIDDEN, VALUE_HIDDEN)

    wp_full = np.zeros((POLICY_OUT, HEAD_INPUT), dtype=np.float32)
    wp_full[:, :HIDDEN] = wp.reshape(POLICY_OUT, HIDDEN)

    out = np.concatenate([
        w1, b1,
        wv1_full.reshape(-1), bv1,
        wv2, bv2,
        wp_full.reshape(-1), bp,
    ]).astype(np.float32, copy=False)
    assert out.size == FULL_FLOATS

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    out.tofile(dst)

    # Strong invariants: all legacy information is preserved and the new half
    # is exactly zero, so full-acc inference equals legacy inference initially.
    reread = np.fromfile(dst, dtype=np.float32)
    assert reread.size == FULL_FLOATS
    cursor = NUM_FEATURES * HIDDEN + HIDDEN
    value = reread[cursor:cursor + HEAD_INPUT * VALUE_HIDDEN].reshape(HEAD_INPUT, VALUE_HIDDEN)
    assert np.array_equal(value[:HIDDEN], wv1.reshape(HIDDEN, VALUE_HIDDEN))
    assert np.count_nonzero(value[HIDDEN:]) == 0
    cursor += HEAD_INPUT * VALUE_HIDDEN + VALUE_HIDDEN + VALUE_HIDDEN + 1
    policy = reread[cursor:cursor + POLICY_OUT * HEAD_INPUT].reshape(POLICY_OUT, HEAD_INPUT)
    assert np.array_equal(policy[:, :HIDDEN], wp.reshape(POLICY_OUT, HIDDEN))
    assert np.count_nonzero(policy[:, HIDDEN:]) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="legacy float32 NNUE weight file")
    ap.add_argument("dst", help="full-accumulator float32 output file")
    args = ap.parse_args()
    migrate(args.src, args.dst)
    print(f"migrated {args.src} -> {args.dst}")
    print(f"layout: 354->256 shared accumulator, 512->32->1 value head, 512->209 policy head")
    print("warm start invariant: opponent half = 0, therefore initial behavior = legacy network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
