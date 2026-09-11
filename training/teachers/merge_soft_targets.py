#!/usr/bin/env python3
"""Merge aligned teacher-target files into a confidence-aware ensemble.

A source must contain ``id`` and may provide ``policy``, ``value``, or both.
This lets independent engines such as Titanium contribute policy evidence
without pretending that their score scale is calibrated to the neural value
heads. Policy and value weights are normalized independently over sources that
actually provide that head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

POLICY_DIM = 209


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
    if "id" not in data.files:
        raise ValueError(f"{name}: missing id field")
    ids = np.asarray(data["id"])
    policy = None
    value = None
    if "policy" in data.files:
        policy = np.asarray(data["policy"], dtype=np.float32)
        if policy.shape != (len(ids), POLICY_DIM):
            raise ValueError(f"{name}: invalid policy shape {policy.shape}")
        if not np.isfinite(policy).all() or np.any(policy < -1e-7):
            raise ValueError(f"{name}: policy contains invalid values")
        sums = policy.sum(axis=1)
        if np.max(np.abs(sums - 1.0)) > 5e-3:
            raise ValueError(f"{name}: policy is not normalized")
    if "value" in data.files:
        value = np.asarray(data["value"], dtype=np.float32)
        if value.shape != (len(ids),):
            raise ValueError(f"{name}: invalid value shape {value.shape}")
        if not np.isfinite(value).all() or np.any(value < -1.001) or np.any(value > 1.001):
            raise ValueError(f"{name}: value must be finite and in [-1,1]")
    if policy is None and value is None:
        raise ValueError(f"{name}: source contains neither policy nor value")
    return {
        "name": name,
        "path": path,
        "id": ids,
        "policy": policy,
        "value": value,
    }


def normalized_weights(entries: list[tuple[dict, float]], head: str) -> tuple[list[dict], np.ndarray]:
    present = [(item, weight) for item, weight in entries if item[head] is not None]
    if not present:
        return [], np.asarray([], dtype=np.float32)
    weights = np.asarray([weight for _, weight in present], dtype=np.float32)
    weights /= weights.sum()
    return [item for item, _ in present], weights


def weighted_js(policies: np.ndarray, mean_policy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(policies, eps, 1.0)
    m = np.clip(mean_policy[None, :, :], eps, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(m)), axis=2)
    return np.tensordot(weights, kl, axes=(0, 0)).astype(np.float32)


def weighted_std(values: np.ndarray, mean_value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = values - mean_value[None, :]
    variance = np.tensordot(weights, centered * centered, axes=(0, 0))
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.source) < 2:
        raise SystemExit("ensemble requires at least two --source inputs")

    try:
        entries: list[tuple[dict, float]] = []
        for name, path, weight in args.source:
            entries.append((load_target(name, path), weight))
        base_ids = entries[0][0]["id"]
        for target, _ in entries[1:]:
            if not np.array_equal(base_ids, target["id"]):
                raise ValueError(
                    f"{target['name']}: ids/order differ; align position corpora before merging"
                )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    policy_sources, policy_weights = normalized_weights(entries, "policy")
    value_sources, value_weights = normalized_weights(entries, "value")
    if not policy_sources:
        raise SystemExit("ensemble needs at least one policy source")

    policies = np.stack([item["policy"] for item in policy_sources], axis=0)
    mean_policy = np.tensordot(policy_weights, policies, axes=(0, 0)).astype(np.float32)
    mean_policy /= mean_policy.sum(axis=1, keepdims=True)
    top1 = policies.argmax(axis=2)
    consensus_top1 = mean_policy.argmax(axis=1)
    agreement_fraction = (top1 == consensus_top1[None, :]).mean(axis=0).astype(np.float32)
    policy_js = weighted_js(policies, mean_policy, policy_weights)

    arrays: dict[str, np.ndarray] = {
        "id": base_ids,
        "policy": mean_policy.astype(np.float16),
        "agreement_fraction": agreement_fraction,
        "policy_js": policy_js,
    }

    if value_sources:
        values = np.stack([item["value"] for item in value_sources], axis=0)
        mean_value = np.tensordot(value_weights, values, axes=(0, 0)).astype(np.float32)
        value_std = weighted_std(values, mean_value, value_weights)
        arrays["value"] = mean_value.astype(np.float16)
        arrays["value_std"] = value_std
        value_confidence = 1.0 / (1.0 + value_std)
    else:
        value_std = np.zeros(len(base_ids), dtype=np.float32)
        value_confidence = np.ones(len(base_ids), dtype=np.float32)

    policy_confidence = agreement_fraction / (1.0 + policy_js)
    confidence = (policy_confidence * value_confidence).astype(np.float32)
    arrays["confidence"] = confidence

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)

    weight_by_name = {name: float(weight) for name, _path, weight in args.source}
    manifest = {
        "schema": "zquoridor.teacher.ensemble.v2",
        "out": str(args.out),
        "samples": len(base_ids),
        "sources": [
            {
                "name": item["name"],
                "path": str(item["path"]),
                "weight": weight_by_name[item["name"]],
                "has_policy": item["policy"] is not None,
                "has_value": item["value"] is not None,
            }
            for item, _ in entries
        ],
        "policy_sources": [item["name"] for item in policy_sources],
        "policy_weights_normalized": [float(x) for x in policy_weights],
        "value_sources": [item["name"] for item in value_sources],
        "value_weights_normalized": [float(x) for x in value_weights],
        "agreement_mean": float(agreement_fraction.mean()),
        "policy_js_mean": float(policy_js.mean()),
        "value_std_mean": float(value_std.mean()),
        "confidence_mean": float(confidence.mean()),
        "unanimous_policy_fraction": float((agreement_fraction == 1.0).mean()),
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
