#!/usr/bin/env python3
"""Blend Claustrophobia and ZQuoridor search targets on mined weakness positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = (left + right) * 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(left > 0, left * (np.log(np.maximum(left, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), 0).sum(axis=1)
        b = np.where(right > 0, right * (np.log(np.maximum(right, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), 0).sum(axis=1)
    return np.maximum((a + b).astype(np.float32) * 0.5, 0.0)


def blend_targets(
    zq_targets_path: Path,
    claustro_targets_path: Path,
    out_path: Path,
    claustro_weight: float = 0.75,
    zq_weight: float = 0.25,
    disagreement_scale: float = 2.0,
    base_weight: float = 1.0,
    max_weight: float = 5.0,
) -> dict:
    zq = np.load(zq_targets_path, allow_pickle=False)
    claustro = np.load(claustro_targets_path, allow_pickle=False)

    if not np.array_equal(zq["id"], claustro["id"]):
        raise ValueError("ZQ and Claustrophobia target IDs are not identical")

    n = len(zq["id"])
    zq_pol = zq["policy"].astype(np.float32)
    cl_pol = claustro["policy"].astype(np.float32)
    zq_val = zq["value"].astype(np.float32)
    cl_val = claustro["value"].astype(np.float32)

    total_w = claustro_weight + zq_weight
    blended_pol = (cl_pol * claustro_weight + zq_pol * zq_weight) / total_w
    # Normalize
    pol_sum = blended_pol.sum(axis=1, keepdims=True)
    blended_pol = np.where(pol_sum > 0, blended_pol / pol_sum, blended_pol)

    blended_val = (cl_val * claustro_weight + zq_val * zq_weight) / total_w

    disagreement = _js(cl_pol, zq_pol) + np.abs(cl_val - zq_val)
    weight = base_weight * (1.0 + disagreement_scale * disagreement)
    weight = np.clip(weight, 0.1, max_weight).astype(np.float32)

    arrays = {
        "id": zq["id"],
        "policy": blended_pol.astype(np.float32),
        "value": blended_val.astype(np.float32),
        "weight": weight,
        "disagreement": disagreement.astype(np.float32),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    manifest = {
        "schema": "zquoridor.teacher.blended_targets.v1",
        "out": str(out_path),
        "samples": n,
        "claustro_weight": claustro_weight,
        "zq_search_weight": zq_weight,
        "mean_weight": float(weight.mean()),
        "mean_disagreement": float(disagreement.mean()),
        "p90_disagreement": float(np.quantile(disagreement, 0.90)),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zq-targets", type=Path, default=ROOT / "data/teaching/weakness-mining-35k/zq-search.npz")
    parser.add_argument("--claustro-targets", type=Path, default=ROOT / "data/teaching/weakness-mining-35k/claustro-direct.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "data/teaching/weakness-mining-35k/blended_targets.npz")
    parser.add_argument("--claustro-weight", type=float, default=0.75)
    parser.add_argument("--zq-weight", type=float, default=0.25)
    args = parser.parse_args()

    manifest = blend_targets(args.zq_targets, args.claustro_targets, args.out, args.claustro_weight, args.zq_weight)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
