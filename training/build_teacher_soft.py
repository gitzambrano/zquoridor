#!/usr/bin/env python3
"""Join a position corpus with soft teacher targets and canonical NNUE states."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTO = "ZQSTATE"
STATE_FIELDS = (
    ("own_pawn", np.uint8),
    ("opp_pawn", np.uint8),
    ("walls_h", np.uint64),
    ("walls_v", np.uint64),
    ("walls_left_own", np.int8),
    ("walls_left_opp", np.int8),
    ("own_dist", np.uint8),
    ("opp_dist", np.uint8),
    ("mover", np.uint8),
)


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


def encode_states(positions: Sequence[dict], encoder: Path) -> dict[str, np.ndarray]:
    if not encoder.is_file():
        raise FileNotFoundError(f"state encoder not found: {encoder}")
    text = "\n".join(" ".join(row["history"]) for row in positions) + "\n"
    proc = subprocess.run([str(encoder)], input=text, capture_output=True, text=True, check=True)
    protocol = [line for line in proc.stdout.splitlines() if line.startswith(PROTO + "\t")]
    if len(protocol) != len(positions):
        raise ValueError(f"state encoder returned {len(protocol)} rows for {len(positions)} positions")
    columns: dict[str, list[int]] = {name: [] for name, _ in STATE_FIELDS}
    for index, line in enumerate(protocol):
        parts = line.split("\t")
        if len(parts) < 2 or parts[1] != "ok":
            raise ValueError(f"state encoder failed at sample {index}: {line}")
        values = parts[2:]
        if len(values) != len(STATE_FIELDS):
            raise ValueError(f"state encoder returned {len(values)} fields at sample {index}")
        for (name, _), text_value in zip(STATE_FIELDS, values):
            columns[name].append(int(text_value))
    return {name: np.asarray(columns[name], dtype=dtype) for name, dtype in STATE_FIELDS}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--encoder", default=str(ROOT / "bin" / "teacher_encode_state"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--disagreement-boost", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.disagreement_boost <= 0.0:
        raise SystemExit("disagreement-boost must be positive")
    try:
        positions = load_positions(args.positions)
        target = np.load(args.targets, allow_pickle=False)
        required = {"id", "policy", "value"}
        missing = sorted(required - set(target.files))
        if missing:
            raise ValueError(f"soft-target file lacks fields: {missing}")
        target_ids = [value.decode("ascii") for value in target["id"]]
        by_id = {sid: index for index, sid in enumerate(target_ids)}
        if len(by_id) != len(target_ids):
            raise ValueError("soft-target file contains duplicate ids")
        order = []
        for row in positions:
            if row["id"] not in by_id:
                raise ValueError(f"missing teacher target for position {row['id']}")
            order.append(by_id[row["id"]])
        order = np.asarray(order, dtype=np.int64)
        policy = target["policy"][order].astype(np.float32)
        value = target["value"][order].astype(np.float32)
        if policy.shape != (len(positions), 209):
            raise ValueError(f"invalid teacher policy shape: {policy.shape}")
        state = encode_states(positions, args.encoder)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    weight = np.ones(len(positions), dtype=np.float32)
    student_disagrees = np.zeros(len(positions), dtype=np.bool_)
    for i, row in enumerate(positions):
        agrees = row.get("metadata", {}).get("student_agrees")
        if agrees is False:
            student_disagrees[i] = True
            weight[i] *= args.disagreement_boost

    arrays = dict(state)
    arrays.update(
        id=np.asarray([row["id"] for row in positions], dtype="S24"),
        policy=policy,
        value=value,
        weight=weight,
        student_disagrees=student_disagrees,
        is_val=np.asarray([row.get("split") == "val" for row in positions], dtype=np.bool_),
        opening_index=np.asarray([int(row.get("opening_index", -1)) for row in positions], dtype=np.int32),
        ply=np.asarray([int(row.get("ply", len(row["history"]))) for row in positions], dtype=np.int16),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    manifest = {
        "schema": "zquoridor.teacher.soft_dataset.v1",
        "positions": str(args.positions),
        "targets": str(args.targets),
        "dataset": str(args.out),
        "samples": len(positions),
        "validation_samples": int(arrays["is_val"].sum()),
        "student_disagreements": int(student_disagrees.sum()),
        "disagreement_boost": args.disagreement_boost,
        "policy_target": "full 209-way teacher soft policy in ZQ canonical frame",
        "value_target": "teacher side-to-move value in [-1,1]",
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
