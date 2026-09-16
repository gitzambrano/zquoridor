#!/usr/bin/env python3
"""Export the hardest direct-relabel states for expensive search teaching.

Input is a completed ``prepare_replay.py`` directory: its ``dataset.npz``
provides canonical V3 states and its ``direct_*.npz`` caches provide separate
old-NNUE and Claustrophobia predictions.  Output is a JSONL corpus of
``zquoridor.position.v1`` rows whose history uses the explicit ``@state`` V3
snapshot protocol.  These rows can be sent to both deep-search teachers.

Edit CONFIG or override the same keys through CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "replay_dir": str(ROOT / "data" / "teaching" / "replay-old-gen1-500k"),
    "out": str(ROOT / "data" / "teaching" / "search-priority" / "positions.jsonl"),
    "max_positions": 10000,
    "policy_weight": 1.0,
    "value_weight": 1.0,
}

STATE_FIELDS = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own", "walls_left_opp")


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in nats for paired legal policy targets."""
    midpoint = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(left > 0, left * (np.log(left) - np.log(midpoint)), 0.0).sum(axis=1)
        second = np.where(right > 0, right * (np.log(right) - np.log(midpoint)), 0.0).sum(axis=1)
    return (0.5 * (first + second)).astype(np.float32)


def load_direct_predictions(replay_dir: Path, expected_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(replay_dir.glob("direct_*.npz"))
    if not paths:
        raise ValueError(f"no direct caches under {replay_dir}")
    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            required = {"id", "old_policy", "old_value", "claustro_policy", "claustro_value"}
            if not required.issubset(cache.files):
                raise ValueError(f"{path} is not a complete direct replay cache")
            chunks.append({name: cache[name] for name in required})
    ids = np.concatenate([chunk["id"] for chunk in chunks])
    if not np.array_equal(ids, expected_ids):
        raise ValueError("direct cache ids do not match dataset ids; do not mix replay runs")
    policy = js_divergence(
        np.concatenate([chunk["old_policy"] for chunk in chunks]),
        np.concatenate([chunk["claustro_policy"] for chunk in chunks]),
    )
    value = np.abs(np.concatenate([chunk["old_value"] for chunk in chunks])
                   - np.concatenate([chunk["claustro_value"] for chunk in chunks])).astype(np.float32)
    return ids, policy, value


def select_indices(policy: np.ndarray, value: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    if len(policy) != len(value) or not len(policy):
        raise ValueError("empty or inconsistent teacher predictions")
    for key in ("policy_weight", "value_weight"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["policy_weight"] + config["value_weight"] <= 0:
        raise ValueError("at least one disagreement weight must be positive")
    n = min(int(config["max_positions"]), len(policy))
    if n <= 0:
        raise ValueError("max_positions must be positive")
    score = config["policy_weight"] * policy + config["value_weight"] * value
    order = np.argsort(score, kind="stable")[-n:][::-1]
    return order, score


def run(config: dict) -> dict:
    replay_dir = Path(config["replay_dir"])
    with np.load(replay_dir / "dataset.npz", allow_pickle=False) as data:
        dataset = {name: data[name] for name in (*STATE_FIELDS, "id", "is_val")}
    ids, policy, value = load_direct_predictions(replay_dir, dataset["id"])
    chosen, score = select_indices(policy, value, config)
    out = Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for rank, index in enumerate(chosen):
            history = ["@state", *(str(int(dataset[field][index])) for field in STATE_FIELDS)]
            row = {
                "schema": "zquoridor.position.v1",
                "id": ids[index].decode(),
                "history": history,
                "side_to_move": 0,
                "opening_index": int(index),
                "ply": 0,
                "split": "val" if bool(dataset["is_val"][index]) else "train",
                "metadata": {
                    "source": "v3-snapshot-no-repetition-history",
                    "rank": rank,
                    "policy_js": float(policy[index]),
                    "value_abs_diff": float(value[index]),
                    "priority_score": float(score[index]),
                },
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema": "zquoridor.teacher.search_selection.v1",
        "replay_dir": str(replay_dir.resolve()),
        "output": str(out.resolve()),
        "samples": int(len(chosen)),
        "policy_weight": config["policy_weight"],
        "value_weight": config["value_weight"],
        "snapshot_repetition_history": "empty",
        "sha256": digest,
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=argparse.SUPPRESS)
    config = dict(CONFIG, **vars(parser.parse_args(argv)))
    print(json.dumps(run(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
