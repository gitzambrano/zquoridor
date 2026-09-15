#!/usr/bin/env python3
"""Relabel positions with a large-budget ZQuoridor MCAB teacher.

The bridge exposes the current search tree root directly, so unlike external
teachers this path can store dense visit targets and per-action Q without any
notation or action-space ambiguity.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

POLICY_DIM = 209


def parse_budgets(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or any(value <= 1 for value in values):
        raise ValueError("node-budgets must contain integers greater than one")
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


def input_text(positions: Sequence[dict]) -> str:
    return "".join(
        str(row["id"]) + "\t" + " ".join(row["history"]) + "\n"
        for row in positions
    )


def run_budget(bridge: Path, nnue: Path, positions: Sequence[dict], nodes: int,
               time_ms: int, leaf_depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cmd = [
        str(bridge), "--nnue", str(nnue), "--nodes", str(nodes),
        "--time-ms", str(time_ms), "--leaf-depth", str(leaf_depth),
    ]
    proc = subprocess.run(
        cmd,
        input=input_text(positions),
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if len(rows) != len(positions):
        raise ValueError(f"ZQ-deep returned {len(rows)} rows for {len(positions)} positions")

    policy = np.zeros((len(rows), POLICY_DIM), dtype=np.float32)
    value = np.empty(len(rows), dtype=np.float32)
    action_q = np.full((len(rows), POLICY_DIM), np.nan, dtype=np.float32)
    best = np.empty(len(rows), dtype=np.uint16)
    for i, (out, position) in enumerate(zip(rows, positions)):
        if out.get("error"):
            raise ValueError(f"ZQ-deep {position['id']}: {out['error']}")
        if out.get("id") != position.get("id"):
            raise ValueError(f"ZQ-deep id mismatch at row {i}")
        if int(out.get("side_to_move")) != int(position["side_to_move"]):
            raise ValueError(f"ZQ-deep side mismatch at row {i}")
        visit_sum = 0.0
        for action, visits, q_prob in out.get("edges", []):
            action = int(action)
            visits = float(visits)
            q_prob = float(q_prob)
            if not 0 <= action < POLICY_DIM or visits < 0.0 or not 0.0 <= q_prob <= 1.0:
                raise ValueError(f"ZQ-deep malformed edge at row {i}")
            policy[i, action] += visits
            visit_sum += visits
            if visits > 0.0:
                action_q[i, action] = 2.0 * q_prob - 1.0
        best[i] = int(out["best_action"])
        if visit_sum > 0.0:
            policy[i] /= visit_sum
        else:
            policy[i, int(best[i])] = 1.0
        root_prob = float(out["root_value_prob"])
        if not 0.0 <= root_prob <= 1.0:
            raise ValueError(f"ZQ-deep malformed root value at row {i}")
        value[i] = 2.0 * root_prob - 1.0
    return policy, value, action_q, best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--nnue", required=True, type=Path)
    parser.add_argument("--node-budgets", default="100000,500000")
    parser.add_argument("--time-ms", type=int, default=0)
    parser.add_argument("--leaf-depth", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher-name", default="zquoridor-deep")
    args = parser.parse_args(argv)

    if args.time_ms < 0 or args.leaf_depth < 0:
        raise SystemExit("time-ms and leaf-depth must be non-negative")
    try:
        budgets = parse_budgets(args.node_budgets)
        positions = load_positions(args.positions)
        results = {
            nodes: run_budget(
                args.bridge, args.nnue, positions, nodes, args.time_ms, args.leaf_depth
            )
            for nodes in budgets
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    selected = budgets[-1]
    policy, value, action_q, best = results[selected]
    best_matrix = np.stack([results[nodes][3] for nodes in budgets], axis=0)
    budget_agreement = (best_matrix == best_matrix[-1:]).mean(axis=0).astype(np.float32)
    q_coverage = np.isfinite(action_q).mean(axis=1).astype(np.float32)

    arrays = {
        "id": np.asarray([row["id"] for row in positions], dtype="S24"),
        "side_to_move": np.asarray([row["side_to_move"] for row in positions], dtype=np.uint8),
        "policy": policy.astype(np.float16),
        "value": value.astype(np.float16),
        "action_q": action_q.astype(np.float16),
        "best_action": best,
        "budget_agreement": budget_agreement,
        "action_q_coverage": q_coverage,
    }
    for nodes in budgets:
        p, v, q, b = results[nodes]
        arrays[f"policy_nodes_{nodes}"] = p.astype(np.float16)
        arrays[f"value_nodes_{nodes}"] = v.astype(np.float16)
        arrays[f"action_q_nodes_{nodes}"] = q.astype(np.float16)
        arrays[f"best_action_nodes_{nodes}"] = b

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    manifest = {
        "schema": "zquoridor.teacher.search_targets.v1",
        "positions": str(args.positions),
        "out": str(args.out),
        "samples": len(positions),
        "teacher": args.teacher_name,
        "mode": "zquoridor-mcab-deep",
        "node_budgets": budgets,
        "selected_node_budget": selected,
        "time_ms": args.time_ms,
        "leaf_depth": args.leaf_depth,
        "target": "root visit policy, root value, and visited-action Q",
        "action_q_available": True,
        "action_q_coverage_mean": float(q_coverage.mean()),
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
