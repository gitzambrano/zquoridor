#!/usr/bin/env python3
"""Convert teacher JSONL records into a compact canonical NNUE policy dataset."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = Path(__file__).resolve().parent / "teachers"
sys.path.insert(0, str(TEACHERS))

from select_samples import iter_records  # noqa: E402

FIELDS = (
    ("own_pawn", np.uint8),
    ("opp_pawn", np.uint8),
    ("walls_h", np.uint64),
    ("walls_v", np.uint64),
    ("walls_left_own", np.int8),
    ("walls_left_opp", np.int8),
    ("own_dist", np.uint8),
    ("opp_dist", np.uint8),
    ("mover", np.uint8),
    ("policy_idx", np.uint16),
)


def teacher_weight(record: dict, disagreement_boost: float) -> tuple[float, bool]:
    """Return one confidence weight and whether teacher and student disagree."""
    diagnostics = record.get("diagnostics", {})
    confidence = float(diagnostics.get("highest_budget_vote_fraction", 1.0))
    confidence = max(0.0, min(1.0, confidence))
    disagrees = diagnostics.get("student_agrees") is False
    if disagrees:
        confidence *= disagreement_boost
    return confidence, disagrees


def write_encoder_input(corpus: Path, path: Path, disagreement_boost: float) -> dict[str, list]:
    """Write the encoder line protocol and retain compact sample metadata."""
    metadata: dict[str, list] = {
        "weight": [],
        "student_disagrees": [],
        "is_val": [],
        "opening_index": [],
        "ply": [],
    }
    teachers: set[str] = set()
    with path.open("w", encoding="utf-8") as output:
        for record in iter_records(corpus):
            move = record.get("bestmove")
            history = record.get("history")
            if not isinstance(move, str) or not isinstance(history, list):
                raise ValueError("teacher record lacks bestmove/history")
            if not all(isinstance(item, str) for item in history):
                raise ValueError("teacher history contains a non-string move")
            weight, disagrees = teacher_weight(record, disagreement_boost)
            if weight <= 0.0:
                continue
            output.write(move + "\t" + " ".join(history) + "\n")
            metadata["weight"].append(weight)
            metadata["student_disagrees"].append(disagrees)
            metadata["is_val"].append(record.get("split") == "val")
            metadata["opening_index"].append(int(record.get("opening_index", -1)))
            metadata["ply"].append(int(record.get("ply", len(history))))
            teachers.add(str(record.get("teacher", "unknown")))
    metadata["teachers"] = sorted(teachers)
    return metadata


def run_encoder(encoder: Path, input_path: Path, output_path: Path) -> None:
    """Run the C++ rules engine over the teacher records."""
    if not encoder.is_file():
        raise FileNotFoundError(f"teacher encoder not found: {encoder}")
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        subprocess.run([str(encoder)], stdin=source, stdout=target, check=True, text=True)


def read_encoder_output(path: Path, expected: int) -> dict[str, np.ndarray]:
    """Parse canonical compact states emitted by encode_position.cpp."""
    columns: dict[str, list[int]] = {name: [] for name, _ in FIELDS}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected:
        raise ValueError(f"encoder returned {len(lines)} rows for {expected} samples")
    for index, line in enumerate(lines):
        parts = line.split("\t")
        if not parts or parts[0] != "ok":
            raise ValueError(f"encoder failed at sample {index}: {line}")
        if len(parts) != 1 + len(FIELDS):
            raise ValueError(f"encoder returned {len(parts) - 1} fields at sample {index}")
        for (name, _), text in zip(FIELDS, parts[1:]):
            columns[name].append(int(text))
    return {
        name: np.asarray(columns[name], dtype=dtype)
        for name, dtype in FIELDS
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build one policy-distillation dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--encoder", default=str(ROOT / "bin" / "teacher_encode"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--disagreement-boost", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.disagreement_boost <= 0.0:
        raise SystemExit("disagreement-boost must be positive")

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="zq_teacher_") as tmp:
            input_path = Path(tmp) / "encoder_input.tsv"
            output_path = Path(tmp) / "encoder_output.tsv"
            metadata = write_encoder_input(corpus, input_path, args.disagreement_boost)
            sample_count = len(metadata["weight"])
            if sample_count == 0:
                raise ValueError("teacher corpus contains no usable samples")
            run_encoder(Path(args.encoder), input_path, output_path)
            encoded = read_encoder_output(output_path, sample_count)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    arrays = dict(encoded)
    arrays.update(
        weight=np.asarray(metadata["weight"], dtype=np.float32),
        student_disagrees=np.asarray(metadata["student_disagrees"], dtype=np.bool_),
        is_val=np.asarray(metadata["is_val"], dtype=np.bool_),
        opening_index=np.asarray(metadata["opening_index"], dtype=np.int32),
        ply=np.asarray(metadata["ply"], dtype=np.int16),
    )
    np.savez(out, **arrays)

    manifest = {
        "schema": "zquoridor.teacher.policy_dataset.v1",
        "source": str(corpus),
        "dataset": str(out),
        "encoder": str(Path(args.encoder)),
        "samples": sample_count,
        "teachers": metadata["teachers"],
        "validation_samples": int(arrays["is_val"].sum()),
        "student_disagreements": int(arrays["student_disagrees"].sum()),
        "disagreement_boost": args.disagreement_boost,
        "policy_target": "teacher bestmove, canonical mover perspective",
        "state_encoding": "compact canonical NNUE state; features expanded by trainer",
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
