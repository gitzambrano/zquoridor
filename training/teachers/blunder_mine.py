#!/usr/bin/env python3
"""Mine high-regret student positions and local teaching windows.

Preferred signal is exact search regret from a target file that carries dense
``action_q``: max_a Q(a) - Q(played). If action-Q is unavailable, the fallback
uses consecutive teacher values on the same trajectory. For side-to-move values
in [-1,1], the mover's post-move value is ``-V(next)``, so the one-step drop is
``V(now) + V(next)``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from targets import dumps, make_position

PROTO = "ZQTEACH"


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


def map_played_actions(positions: Sequence[dict], encoder: Path) -> np.ndarray:
    """Use the exact ZQ C++ encoder to map played notation to canonical indices."""
    lines = []
    mapped_rows = []
    for i, row in enumerate(positions):
        played = row.get("metadata", {}).get("played_move")
        if played:
            lines.append(str(played) + "\t" + " ".join(row["history"]))
            mapped_rows.append(i)
    indices = np.full(len(positions), -1, dtype=np.int16)
    if not lines:
        return indices
    if not encoder.is_file():
        raise FileNotFoundError(f"teacher move encoder not found: {encoder}")
    proc = subprocess.run(
        [str(encoder)], input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=True,
    )
    protocol = [line for line in proc.stdout.splitlines() if line.startswith(PROTO + "\t")]
    if len(protocol) != len(mapped_rows):
        raise ValueError(f"move encoder returned {len(protocol)} rows for {len(mapped_rows)} moves")
    for source_idx, line in zip(mapped_rows, protocol):
        parts = line.split("\t")
        if len(parts) < 2 or parts[1] != "ok":
            raise ValueError(f"move encoder failed for position {positions[source_idx]['id']}: {line}")
        indices[source_idx] = int(parts[-1])
    return indices


def align_targets(positions: Sequence[dict], target_path: Path) -> dict[str, np.ndarray]:
    data = np.load(target_path, allow_pickle=False)
    required = {"id", "policy", "value"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"teacher target file lacks fields {missing}")
    ids = [value.decode("ascii") for value in data["id"]]
    by_id = {sid: i for i, sid in enumerate(ids)}
    if len(by_id) != len(ids):
        raise ValueError("teacher target file contains duplicate ids")
    order = []
    for row in positions:
        sid = row["id"]
        if sid not in by_id:
            raise ValueError(f"missing teacher target for {sid}")
        order.append(by_id[sid])
    order = np.asarray(order, dtype=np.int64)
    aligned = {
        "policy": np.asarray(data["policy"][order], dtype=np.float32),
        "value": np.asarray(data["value"][order], dtype=np.float32),
    }
    if "action_q" in data.files:
        aligned["action_q"] = np.asarray(data["action_q"][order], dtype=np.float32)
    return aligned


def compute_signals(positions: Sequence[dict], targets: dict[str, np.ndarray],
                    played_index: np.ndarray) -> list[dict]:
    """Return one diagnostic row per position with a played move."""
    if targets["policy"].shape != (len(positions), 209):
        raise ValueError("policy target shape does not match position corpus")
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(positions):
        groups[int(row.get("opening_index", -1))].append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: int(positions[i].get("ply", len(positions[i]["history"]))))

    next_in_game: dict[int, int] = {}
    for indices in groups.values():
        for a, b in zip(indices, indices[1:]):
            pa = int(positions[a].get("ply", len(positions[a]["history"])))
            pb = int(positions[b].get("ply", len(positions[b]["history"])))
            if pb == pa + 1:
                next_in_game[a] = b

    diagnostics = []
    for i, row in enumerate(positions):
        action = int(played_index[i])
        if action < 0:
            continue
        policy = targets["policy"][i]
        policy_gap = float(policy.max() - policy[action])
        result = {
            "index": i,
            "id": row["id"],
            "opening_index": int(row.get("opening_index", -1)),
            "ply": int(row.get("ply", len(row["history"]))),
            "played_action": action,
            "teacher_top_action": int(policy.argmax()),
            "policy_gap": policy_gap,
            "method": "policy-gap",
            "regret": policy_gap,
        }
        action_q = targets.get("action_q")
        if action_q is not None:
            q_row = action_q[i]
            finite = np.isfinite(q_row)
            if finite.any() and np.isfinite(q_row[action]):
                regret = float(np.nanmax(q_row) - q_row[action])
                result.update(
                    method="action-q",
                    regret=max(0.0, regret),
                    played_q=float(q_row[action]),
                    best_q=float(np.nanmax(q_row)),
                )
                diagnostics.append(result)
                continue
        nxt = next_in_game.get(i)
        if nxt is not None:
            drop = float(targets["value"][i] + targets["value"][nxt])
            result.update(
                method="value-drop",
                regret=max(0.0, drop),
                value_before=float(targets["value"][i]),
                value_after_for_mover=float(-targets["value"][nxt]),
            )
        diagnostics.append(result)
    return diagnostics


def select_windows(positions: Sequence[dict], diagnostics: Sequence[dict],
                   threshold: float, radius: int, max_anchors: int) -> tuple[list[dict], list[dict]]:
    anchors = [row for row in diagnostics if float(row["regret"]) >= threshold]
    anchors.sort(key=lambda row: (-float(row["regret"]), row["opening_index"], row["ply"]))
    if max_anchors > 0:
        anchors = anchors[:max_anchors]

    by_game_ply = {
        (int(row.get("opening_index", -1)), int(row.get("ply", len(row["history"])))): row
        for row in positions
    }
    selected: dict[str, dict] = {}
    for anchor_rank, anchor in enumerate(anchors):
        opening = int(anchor["opening_index"])
        anchor_ply = int(anchor["ply"])
        for delta in range(-radius, radius + 1):
            source = by_game_ply.get((opening, anchor_ply + delta))
            if source is None:
                continue
            sid = source["id"]
            if sid in selected:
                continue
            metadata = dict(source.get("metadata", {}))
            metadata.update(
                blunder_anchor_id=anchor["id"],
                blunder_anchor_ply=anchor_ply,
                blunder_offset=delta,
                blunder_regret=float(anchor["regret"]),
                blunder_method=anchor["method"],
                blunder_anchor_rank=anchor_rank,
            )
            selected[sid] = make_position(
                source["history"],
                split=str(source.get("split", "train")),
                source="blunder-window",
                opening_index=opening,
                ply=int(source.get("ply", len(source["history"]))),
                metadata=metadata,
            )
    rows = sorted(selected.values(), key=lambda row: (row["opening_index"], row["ply"]))
    return rows, anchors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnostics-out", type=Path)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-anchors", type=int, default=0)
    args = parser.parse_args(argv)
    if args.threshold < 0.0 or args.radius < 0 or args.max_anchors < 0:
        raise SystemExit("threshold, radius, and max-anchors must be non-negative")

    try:
        positions = load_positions(args.positions)
        targets = align_targets(positions, args.targets)
        played = map_played_actions(positions, args.encoder)
        diagnostics = compute_signals(positions, targets, played)
        windows, anchors = select_windows(
            positions, diagnostics, args.threshold, args.radius, args.max_anchors
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(dumps(row) + "\n" for row in windows), encoding="utf-8")
    diag_path = args.diagnostics_out or Path(str(args.out) + ".diagnostics.jsonl")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics),
        encoding="utf-8",
    )
    manifest = {
        "schema": "zquoridor.teacher.blunder_mining.v1",
        "positions": str(args.positions),
        "targets": str(args.targets),
        "out": str(args.out),
        "threshold": args.threshold,
        "radius": args.radius,
        "positions_scored": len(diagnostics),
        "anchors": len(anchors),
        "window_positions": len(windows),
        "methods": {
            method: sum(row["method"] == method for row in diagnostics)
            for method in sorted({row["method"] for row in diagnostics})
        },
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
