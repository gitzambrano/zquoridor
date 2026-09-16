#!/usr/bin/env python3
"""Build weighted teaching data from two expensive search teachers.

Input is one canonical replay dataset, the JSONL selected by
``select_replay_disagreement.py``, and matching ZQuoridor and Claustrophobia
search target files.  The output is a normal teaching ``dataset.npz``.  It
therefore trains with ``run_experiment.py`` and can be mixed by
``run_campaign.combine_datasets`` without a special training path.

Rows on which either search changes its best action between its requested
budgets remain usable, but receive less weight.  Rows where the search
teachers disagree with each other receive more weight.  Edit CONFIG or pass
the corresponding --cli-option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "source_dataset": str(ROOT / "data/teaching/replay-old-gen1-500k/dataset.npz"),
    "positions": str(ROOT / "data/teaching/search-priority-gen1/top2000.jsonl"),
    "zq_targets": str(ROOT / "data/teaching/search-priority-gen1/zq-search.npz"),
    "claustro_targets": str(ROOT / "data/teaching/search-priority-gen1/claustro-search.npz"),
    "out": str(ROOT / "data/teaching/search-priority-gen1/dataset.npz"),
    # ZQuoridor search is useful but may wander in wall-poor races.  Keep it
    # as a corroborating teacher until arena results support a larger share.
    "zq_policy_weight": 0.25,
    "claustro_policy_weight": 0.75,
    "zq_value_weight": 0.25,
    "claustro_value_weight": 0.75,
    "base_weight": 1.0,
    "disagreement_scale": 3.0,
    "stability_floor": 0.25,
    "max_weight": 8.0,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _id(value) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode()


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("schema") != "zquoridor.position.v1" for row in rows):
        raise ValueError("positions must be nonempty zquoridor.position.v1 JSONL")
    return rows


def _load_targets(path: Path, expected: list[bytes]) -> dict:
    with np.load(path, allow_pickle=False) as raw:
        required = {"id", "policy", "value", "budget_agreement"}
        if not required.issubset(raw.files):
            raise ValueError(f"{path} lacks required search target arrays")
        data = {key: raw[key] for key in required}
    if [_id(value) for value in data["id"]] != expected:
        raise ValueError(f"{path} ids do not exactly match selected positions")
    if data["policy"].ndim != 2 or data["policy"].shape[1] != 209:
        raise ValueError(f"{path} has an invalid policy shape")
    return data


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = (left + right) * 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(left > 0, left * (np.log(left) - np.log(midpoint)), 0).sum(axis=1)
        b = np.where(right > 0, right * (np.log(right) - np.log(midpoint)), 0).sum(axis=1)
    return (a + b).astype(np.float32) * 0.5


def run(config: dict) -> dict:
    for key in ("zq_policy_weight", "claustro_policy_weight", "zq_value_weight", "claustro_value_weight",
                "base_weight", "disagreement_scale", "stability_floor", "max_weight"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["zq_policy_weight"] + config["claustro_policy_weight"] <= 0:
        raise ValueError("policy teacher weights must sum to a positive value")
    if config["zq_value_weight"] + config["claustro_value_weight"] <= 0:
        raise ValueError("value teacher weights must sum to a positive value")
    if config["max_weight"] <= 0:
        raise ValueError("max_weight must be positive")

    rows = _load_rows(Path(config["positions"]))
    expected = [_id(row["id"]) for row in rows]
    zq = _load_targets(Path(config["zq_targets"]), expected)
    claustro = _load_targets(Path(config["claustro_targets"]), expected)
    with np.load(config["source_dataset"], allow_pickle=False) as raw:
        source = {key: raw[key] for key in raw.files}
    source_ids = {_id(value): i for i, value in enumerate(source["id"])}
    if len(source_ids) != len(source["id"]):
        raise ValueError("source dataset contains duplicate ids")
    try:
        indices = np.asarray([source_ids[value] for value in expected], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"selected id is absent from source dataset: {exc}") from exc

    zq_policy, claustro_policy = zq["policy"].astype(np.float32), claustro["policy"].astype(np.float32)
    zq_value, claustro_value = zq["value"].astype(np.float32), claustro["value"].astype(np.float32)
    pw = config["zq_policy_weight"] + config["claustro_policy_weight"]
    vw = config["zq_value_weight"] + config["claustro_value_weight"]
    policy = (zq_policy * config["zq_policy_weight"] + claustro_policy * config["claustro_policy_weight"]) / pw
    value = (zq_value * config["zq_value_weight"] + claustro_value * config["claustro_value_weight"]) / vw
    disagreement = _js(zq_policy, claustro_policy) + np.abs(zq_value - claustro_value)
    stability = np.minimum(zq["budget_agreement"], claustro["budget_agreement"]).astype(np.float32)
    weight = config["base_weight"] * (1.0 + config["disagreement_scale"] * disagreement)
    weight *= config["stability_floor"] + (1.0 - config["stability_floor"]) * stability
    weight = np.clip(weight, np.finfo(np.float32).eps, config["max_weight"]).astype(np.float32)

    out_data = {key: source[key][indices] for key in source if key not in {"policy", "value", "weight"}}
    out_data.update(policy=policy.astype(np.float32), value=value.astype(np.float32), weight=weight)
    if not np.allclose(out_data["policy"].sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("blended policy is not normalized")
    if not np.isfinite(weight).all() or not np.isfinite(value).all():
        raise ValueError("search targets contain non-finite values")
    out = Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **out_data)
    manifest = {
        "schema": "zquoridor.teacher.search_priority_dataset.v1",
        "samples": len(indices),
        "source_dataset": str(Path(config["source_dataset"]).resolve()),
        "positions": str(Path(config["positions"]).resolve()),
        "zq_targets": str(Path(config["zq_targets"]).resolve()),
        "claustro_targets": str(Path(config["claustro_targets"]).resolve()),
        "snapshot_repetition_history": "empty",
        "policy_weights": {"zquoridor": config["zq_policy_weight"], "claustrophobia": config["claustro_policy_weight"]},
        "value_weights": {"zquoridor": config["zq_value_weight"], "claustrophobia": config["claustro_value_weight"]},
        "weight_formula": "clip(base*(1+scale*(policy_js+value_abs_diff))*(floor+(1-floor)*min_budget_agreement), eps, max)",
        "mean_weight": float(weight.mean()),
        "mean_search_disagreement": float(disagreement.mean()),
        "mean_min_budget_agreement": float(stability.mean()),
        "dataset_sha256": _sha(out),
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
