#!/usr/bin/env python3
"""Relabel positions with Claustrophobia MCTS visit policy and root value."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from targets import POLICY_DIM, mirror_action_lr


def parse_budgets(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("sims must be a comma-separated list of positive integers")
    return values


def load_positions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "zquoridor.position.v1":
                raise ValueError(f"{path}:{lineno}: unsupported position schema")
            rows.append(row)
    if not rows:
        raise ValueError("position corpus is empty")
    return rows


def write_tsv(positions: Sequence[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in positions:
            fh.write(str(row["id"]) + "\t" + " ".join(row["history"]) + "\n")


def map_action(index: int, side: int) -> int:
    return index if side == 0 else mirror_action_lr(index)


def run_budget(bridge: Path, checkpoint: Path, tsv: Path, sims: int,
               cpuct: float, positions: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proc = subprocess.run(
        [str(bridge), str(tsv), str(checkpoint), str(sims), str(cpuct)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if len(rows) != len(positions):
        raise ValueError(f"search bridge returned {len(rows)} rows for {len(positions)} positions")
    policy = np.zeros((len(rows), POLICY_DIM), dtype=np.float32)
    value = np.empty(len(rows), dtype=np.float32)
    best = np.empty(len(rows), dtype=np.uint16)
    for i, (out, position) in enumerate(zip(rows, positions)):
        if out.get("id") != position.get("id"):
            raise ValueError(f"search bridge id mismatch at row {i}")
        side = int(position["side_to_move"])
        if int(out.get("side_to_move")) != side:
            raise ValueError(f"search bridge side mismatch at row {i}")
        total = 0
        for action, visits in out.get("visits", []):
            mapped = map_action(int(action), side)
            n = max(0, int(visits))
            policy[i, mapped] += n
            total += n
        mapped_best = map_action(int(out["best_action"]), side)
        best[i] = mapped_best
        if total > 0:
            policy[i] /= float(total)
        else:
            policy[i, mapped_best] = 1.0
        value[i] = float(out["root_value"])
    return policy, value, best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sims", default="512,1024,2048")
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher-name", default="claustrophobia-v1.3.1")
    args = parser.parse_args(argv)

    if args.cpuct <= 0.0:
        raise SystemExit("cpuct must be positive")
    try:
        budgets = parse_budgets(args.sims)
        positions = load_positions(args.positions)
        with tempfile.TemporaryDirectory(prefix="zq_claustro_search_") as tmp:
            tsv = Path(tmp) / "positions.tsv"
            write_tsv(positions, tsv)
            results = {
                sims: run_budget(args.bridge, args.checkpoint, tsv, sims, args.cpuct, positions)
                for sims in budgets
            }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    highest = budgets[-1]
    policy, value, best = results[highest]
    best_matrix = np.stack([results[sims][2] for sims in budgets], axis=0)
    budget_agreement = (best_matrix == best_matrix[-1:]).mean(axis=0).astype(np.float32)
    arrays = {
        "id": np.asarray([row["id"] for row in positions], dtype="S24"),
        "side_to_move": np.asarray([row["side_to_move"] for row in positions], dtype=np.uint8),
        "policy": policy.astype(np.float16),
        "value": value.astype(np.float16),
        "best_action": best,
        "budget_agreement": budget_agreement,
    }
    for sims in budgets:
        p, v, b = results[sims]
        arrays[f"policy_sims_{sims}"] = p.astype(np.float16)
        arrays[f"value_sims_{sims}"] = v.astype(np.float16)
        arrays[f"best_action_sims_{sims}"] = b
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    manifest = {
        "schema": "zquoridor.teacher.search_targets.v1",
        "positions": str(args.positions),
        "out": str(args.out),
        "samples": len(positions),
        "teacher": args.teacher_name,
        "mode": "mcts-search",
        "sims": budgets,
        "selected_sims": highest,
        "cpuct": args.cpuct,
        "target": "root visit distribution plus root side-to-move value",
        "action_q_available": False,
        "budget_agreement_mean": float(budget_agreement.mean()),
        "stable_all_budgets_fraction": float((budget_agreement == 1.0).mean()),
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
