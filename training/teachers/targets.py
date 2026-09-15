#!/usr/bin/env python3
"""Common teacher-target helpers for multi-source policy/value distillation."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Sequence

POLICY_DIM = 209
PAWN_ACTIONS = 81
H_WALL_BASE = 81
V_WALL_BASE = 145


def sample_id(history: Sequence[str]) -> str:
    """Return a stable id for one exact move history."""
    payload = " ".join(move.strip().lower() for move in history).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def zq_move_to_policy_index(move: str, side_to_move: int) -> int:
    """Map one raw-board move string to ZQuoridor's canonical 209-action index.

    ZQ keeps columns fixed and mirrors only rows for side 1. Wall orientation is
    preserved. This matches ``mirrorMoveForPerspective`` + ``moveToPolicyIndex``
    in ``src/nnue.hpp`` / rules code.
    """
    text = str(move).strip().lower()
    if side_to_move not in (0, 1):
        raise ValueError(f"invalid side_to_move: {side_to_move}")
    if len(text) not in (2, 3):
        raise ValueError(f"invalid move syntax: {move!r}")
    col_ch, row_ch = text[0], text[1]
    if col_ch not in "abcdefghi" or row_ch not in "123456789":
        raise ValueError(f"invalid move syntax: {move!r}")
    col = ord(col_ch) - ord("a")
    row = ord(row_ch) - ord("1")

    if len(text) == 2:
        canonical_row = row if side_to_move == 0 else 8 - row
        return canonical_row * 9 + col

    orientation = text[2]
    if orientation not in "hv" or row >= 8 or col >= 8:
        raise ValueError(f"invalid wall move syntax: {move!r}")
    canonical_row = row if side_to_move == 0 else 7 - row
    slot = canonical_row * 8 + col
    return (H_WALL_BASE if orientation == "h" else V_WALL_BASE) + slot


def mirror_action_lr(index: int) -> int:
    """Mirror one canonical 209-action index left/right."""
    if not 0 <= index < POLICY_DIM:
        raise ValueError(f"invalid policy index: {index}")
    if index < PAWN_ACTIONS:
        row, col = divmod(index, 9)
        return row * 9 + (8 - col)
    if index < V_WALL_BASE:
        slot = index - H_WALL_BASE
        row, col = divmod(slot, 8)
        return H_WALL_BASE + row * 8 + (7 - col)
    slot = index - V_WALL_BASE
    row, col = divmod(slot, 8)
    return V_WALL_BASE + row * 8 + (7 - col)


def claustrophobia_policy_to_zq(policy: Sequence[float], side_to_move: int) -> list[float]:
    """Convert Claustrophobia canonical policy probabilities to ZQ canonical indices.

    Side 0 uses the same board frame. For side 1, Claustrophobia rotates both
    row and column by 180 degrees, while ZQ flips only the row. Their canonical
    frames therefore differ by one left/right mirror.
    """
    if len(policy) != POLICY_DIM:
        raise ValueError(f"expected {POLICY_DIM} policy entries, got {len(policy)}")
    if side_to_move not in (0, 1):
        raise ValueError(f"invalid side_to_move: {side_to_move}")
    values = [float(x) for x in policy]
    if side_to_move == 0:
        return values
    out = [0.0] * POLICY_DIM
    for teacher_idx, value in enumerate(values):
        out[mirror_action_lr(teacher_idx)] = value
    return out


def normalize_policy(policy: Sequence[float]) -> list[float]:
    """Normalize non-negative policy mass and reject malformed targets."""
    if len(policy) != POLICY_DIM:
        raise ValueError(f"expected {POLICY_DIM} policy entries, got {len(policy)}")
    values = [float(x) for x in policy]
    if any(not math.isfinite(x) or x < 0.0 for x in values):
        raise ValueError("policy contains negative or non-finite values")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("policy has no positive mass")
    return [x / total for x in values]


def policy_entropy(policy: Sequence[float]) -> float:
    """Return natural-log entropy of a normalized or unnormalized policy."""
    p = normalize_policy(policy)
    return -sum(x * math.log(x) for x in p if x > 0.0)


def top_action(policy: Sequence[float]) -> int:
    """Return the deterministic lowest-index argmax."""
    p = normalize_policy(policy)
    return max(range(POLICY_DIM), key=lambda idx: (p[idx], -idx))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Return Jensen-Shannon divergence in nats."""
    a = normalize_policy(p)
    b = normalize_policy(q)
    m = [(x + y) * 0.5 for x, y in zip(a, b)]

    def kl(x: Iterable[float], y: Iterable[float]) -> float:
        return sum(px * math.log(px / py) for px, py in zip(x, y) if px > 0.0)

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def make_position(history: Sequence[str], *, split: str, source: str,
                  opening_index: int = -1, ply: int | None = None,
                  metadata: dict | None = None) -> dict:
    """Build an architecture-neutral position record."""
    hist = [str(move).strip().lower() for move in history]
    return {
        "schema": "zquoridor.position.v1",
        "id": sample_id(hist),
        "history": hist,
        "side_to_move": len(hist) & 1,
        "split": split,
        "source": source,
        "opening_index": int(opening_index),
        "ply": len(hist) if ply is None else int(ply),
        "metadata": metadata or {},
    }


def make_label(position: dict, *, teacher: str, mode: str,
               policy: Sequence[float] | None = None,
               value: float | None = None, budget: dict | None = None,
               bestmove: str | None = None, action_q: dict[int, float] | None = None,
               metadata: dict | None = None) -> dict:
    """Build one teacher annotation independent of the student architecture."""
    if position.get("schema") != "zquoridor.position.v1":
        raise ValueError("unsupported position schema")
    target: dict = {}
    if policy is not None:
        target["policy"] = normalize_policy(policy)
        target["policy_entropy"] = policy_entropy(policy)
        target["policy_top1"] = top_action(policy)
    if value is not None:
        value = float(value)
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError("teacher value must be finite and in [-1,1]")
        target["value"] = value
    if action_q is not None:
        clean_q = {}
        for key, val in action_q.items():
            idx = int(key)
            q = float(val)
            if not 0 <= idx < POLICY_DIM or not math.isfinite(q) or not -1.0 <= q <= 1.0:
                raise ValueError("invalid action-Q target")
            clean_q[str(idx)] = q
        target["action_q"] = clean_q
    if bestmove is not None:
        target["bestmove"] = str(bestmove)
    if not target:
        raise ValueError("label contains no target")
    return {
        "schema": "zquoridor.teacher.label.v1",
        "id": position["id"],
        "teacher": teacher,
        "mode": mode,
        "budget": budget or {},
        "target": target,
        "metadata": metadata or {},
    }


def dumps(record: dict) -> str:
    """Stable compact JSON encoding for JSONL outputs."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))
