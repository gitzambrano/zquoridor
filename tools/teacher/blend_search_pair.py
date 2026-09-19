#!/usr/bin/env python3
"""Blend ZQuoridor and Claustrophobia search targets into a canonical teaching dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from build_teacher_soft import encode_states, load_positions


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = (left + right) * 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(left > 0, left * (np.log(left) - np.log(midpoint)), 0).sum(axis=1)
        b = np.where(right > 0, right * (np.log(right) - np.log(midpoint)), 0).sum(axis=1)
    return (a + b).astype(np.float32) * 0.5


def blend(
    positions_path: Path,
    claustro_path: Path,
    zq_path: Path,
    encoder_path: Path,
    out_path: Path,
    claustro_weight: float = 0.75,
    zq_weight: float = 0.25,
    base_weight: float = 2.0,
    disagreement_scale: float = 3.0,
    max_weight: float = 8.0,
) -> dict:
    rows = load_positions(positions_path)
    expected_ids = [row["id"] for row in rows]

    claustro = np.load(claustro_path)
    c_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in claustro["id"]]
    c_map = {sid: i for i, sid in enumerate(c_ids)}
    c_idx = [c_map[sid] for sid in expected_ids]
    c_policy = claustro["policy"][c_idx].astype(np.float32)
    c_value = claustro["value"][c_idx].astype(np.float32)

    zq = np.load(zq_path)
    z_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in zq["id"]]
    z_map = {sid: i for i, sid in enumerate(z_ids)}
    z_idx = [z_map[sid] for sid in expected_ids]
    z_policy = zq["policy"][z_idx].astype(np.float32)
    z_value = zq["value"][z_idx].astype(np.float32)

    pw = claustro_weight + zq_weight
    vw = claustro_weight + zq_weight
    blended_policy = (c_policy * claustro_weight + z_policy * zq_weight) / pw
    blended_value = (c_value * claustro_weight + z_value * zq_weight) / vw

    # Normalize policy
    p_sum = blended_policy.sum(axis=1, keepdims=True)
    blended_policy = np.where(p_sum > 0, blended_policy / p_sum, 1.0 / 209.0)

    # Disagreement weight scaling
    disagreement = _js(z_policy, c_policy) + np.abs(z_value - c_value)
    weight = base_weight * (1.0 + disagreement_scale * disagreement)
    weight = np.clip(weight, np.finfo(np.float32).eps, max_weight).astype(np.float32)

    # Encode canonical state features
    state = encode_states(rows, encoder_path)

    # Validation split: 20% deterministic split based on hex ID
    is_val = np.asarray([int(sid[:8], 16) % 5 == 0 for sid in expected_ids], dtype=np.bool_)

    arrays = dict(state)
    arrays.update(
        id=np.asarray(expected_ids, dtype="S24"),
        policy=blended_policy.astype(np.float32),
        value=blended_value.astype(np.float32),
        weight=weight,
        is_val=is_val,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    manifest = {
        "schema": "zquoridor.teacher.blended_crisis_dataset.v1",
        "positions": str(positions_path.resolve()),
        "claustro_targets": str(claustro_path.resolve()),
        "zq_targets": str(zq_path.resolve()),
        "dataset": str(out_path.resolve()),
        "samples": len(rows),
        "validation_samples": int(is_val.sum()),
        "weights": {"claustrophobia": claustro_weight, "zquoridor": zq_weight},
        "mean_weight": float(weight.mean()),
        "mean_search_disagreement": float(disagreement.mean()),
        "dataset_sha256": _sha(out_path),
    }

    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--claustro", type=Path, required=True)
    parser.add_argument("--zq", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=ROOT / "bin/encode_state.exe")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-weight", type=float, default=2.0)
    args = parser.parse_args()

    manifest = blend(args.positions, args.claustro, args.zq, args.encoder, args.out, base_weight=args.base_weight)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
