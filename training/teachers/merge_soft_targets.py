#!/usr/bin/env python3
"""Merge aligned soft-target files into one confidence-aware teacher ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_source(text: str) -> tuple[str, Path, float]:
    parts = text.split("=", 2)
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("source must be NAME=PATH or NAME=PATH=WEIGHT")
    name = parts[0].strip()
    path = Path(parts[1])
    weight = float(parts[2]) if len(parts) == 3 else 1.0
    if not name or weight <= 0.0:
        raise argparse.ArgumentTypeError("source name and weight must be positive/non-empty")
    return name, path, weight


def load_target(name: str, path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"id", "policy", "value"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{name}: missing target fields {missing}")
    ids = np.asarray(data["id"])
    policy = np.asarray(data["policy"], dtype=np.float32)
    value = np.asarray(data["value"], dtype=np.float32)
    if policy.shape != (len(ids), 209) or value.shape != (len(ids),):
        raise ValueError(f"{name}: invalid target shapes")
    sums = policy.sum(axis=1)
    if np.max(np.abs(sums - 1.0)) > 5e-3:
        raise ValueError(f"{name}: policy is not normalized")
    return {"name": name, "path": path, "id": ids, "policy": policy, "value": value}


def js_to_mean(policies: np.ndarray, mean_policy: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(policies, eps, 1.0)
    m = np.clip(mean_policy[None, :, :], eps, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(m)), axis=2)
    return kl.mean(axis=0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.source) < 2:
        raise SystemExit("ensemble requires at least two --source inputs")

    try:
        loaded = [load_target(name, path) for name, path, _ in args.source]
        base_ids = loaded[0]["id"]
        for target in loaded[1:]:
            if not np.array_equal(base_ids, target["id"]):
                raise ValueError(
                    f"{target['name']}: ids/order differ; align position corpora before merging"
                )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    weights = np.asarray([weight for _, _, weight in args.source], dtype=np.float32)
    weights /= weights.sum()
    policies = np.stack([item["policy"] for item in loaded], axis=0)
    values = np.stack([item["value"] for item in loaded], axis=0)
    mean_policy = np.tensordot(weights, policies, axes=(0, 0)).astype(np.float32)
    mean_policy /= mean_policy.sum(axis=1, keepdims=True)
    mean_value = np.tensordot(weights, values, axes=(0, 0)).astype(np.float32)

    top1 = policies.argmax(axis=2)
    consensus_top1 = mean_policy.argmax(axis=1)
    agreement_fraction = (top1 == consensus_top1[None, :]).mean(axis=0).astype(np.float32)
    value_std = values.std(axis=0).astype(np.float32)
    policy_js = js_to_mean(policies, mean_policy).astype(np.float32)
    confidence = (agreement_fraction / (1.0 + policy_js + value_std)).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        id=base_ids,
        policy=mean_policy.astype(np.float16),
        value=mean_value.astype(np.float16),
        agreement_fraction=agreement_fraction,
        policy_js=policy_js,
        value_std=value_std,
        confidence=confidence,
    )
    manifest = {
        "schema": "zquoridor.teacher.ensemble.v1",
        "out": str(args.out),
        "samples": len(base_ids),
        "sources": [
            {"name": name, "path": str(path), "weight": float(weight)}
            for name, path, weight in args.source
        ],
        "agreement_mean": float(agreement_fraction.mean()),
        "policy_js_mean": float(policy_js.mean()),
        "value_std_mean": float(value_std.mean()),
        "confidence_mean": float(confidence.mean()),
        "high_confidence_fraction": float((agreement_fraction == 1.0).mean()),
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
