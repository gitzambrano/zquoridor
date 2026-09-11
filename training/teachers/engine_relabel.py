#!/usr/bin/env python3
"""Relabel an arbitrary position corpus with a UCI/Titanium teacher.

Unlike ``collect.py``, this tool never lets the teacher choose the trajectory.
It queries exactly the histories supplied in a ``zquoridor.position.v1`` JSONL
file, which makes it suitable for DAgger/active-learning subsets. Repeated
multi-budget probes are preserved in JSONL; an optional NPZ emits a conservative
hard policy target (teacher selected move) for ensemble voting.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from collect import make_engine
from common import deepest_rich_snapshot, parse_budgets, summarize_probes
from targets import POLICY_DIM, zq_move_to_policy_index


@dataclass(frozen=True)
class Config:
    teacher: str
    protocol: str
    command: tuple[str, ...]
    budgets: tuple[int, ...]
    repeats: int


def load_positions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "zquoridor.position.v1":
                raise ValueError(f"{path}:{lineno}: unsupported position schema")
            history = row.get("history")
            if not isinstance(history, list) or not all(isinstance(x, str) for x in history):
                raise ValueError(f"{path}:{lineno}: invalid history")
            side = int(row.get("side_to_move", len(history) & 1))
            if side != (len(history) & 1):
                raise ValueError(f"{path}:{lineno}: side/history parity mismatch")
            if not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{lineno}: missing id")
            rows.append(row)
    if not rows:
        raise ValueError("position corpus is empty")
    return rows


def make_streams(config: Config):
    streams = []
    try:
        for budget in config.budgets:
            for repeat in range(config.repeats):
                streams.append((budget, repeat, make_engine(config.teacher, config.protocol, config.command)))
    except Exception:
        for _, _, engine in streams:
            engine.close()
        raise
    return streams


def probe_one(row: dict, streams, config: Config) -> dict:
    history = row["history"]
    probes = []
    for budget, repeat, engine in streams:
        move, think_s, info = engine.bestmove(history, budget)
        probes.append({
            "movetime_ms": budget,
            "repeat": repeat,
            "bestmove": move,
            "think_s": think_s,
            "info": info,
            "rich": deepest_rich_snapshot(info),
        })
    diagnostics = summarize_probes(probes, config.budgets)
    selected_move = diagnostics["selected_move"]
    if selected_move == "(none)":
        raise ValueError(f"{row['id']}: teacher returned no move on a supplied nonterminal position")
    high_budget = int(diagnostics["highest_budget_ms"])
    selected = next(
        (probe for probe in probes
         if int(probe["movetime_ms"]) == high_budget and probe["bestmove"] == selected_move),
        probes[-1],
    )
    return {
        "schema": "zquoridor.teacher.probe.v1",
        "id": row["id"],
        "teacher": config.teacher,
        "protocol": config.protocol,
        "history": history,
        "side_to_move": int(row["side_to_move"]),
        "split": row.get("split", "train"),
        "opening_index": int(row.get("opening_index", -1)),
        "ply": int(row.get("ply", len(history))),
        "bestmove": selected_move,
        "movetime_ms": selected["movetime_ms"],
        "think_s": selected["think_s"],
        "rich": selected["rich"],
        "diagnostics": diagnostics,
        "probes": probes,
        "position_metadata": row.get("metadata", {}),
    }


def relabel_chunk(indexed_rows: list[tuple[int, dict]], config: Config) -> list[tuple[int, dict]]:
    streams = make_streams(config)
    try:
        return [(index, probe_one(row, streams, config)) for index, row in indexed_rows]
    finally:
        for _, _, engine in streams:
            engine.close()


def split_chunks(rows: list[dict], workers: int) -> list[list[tuple[int, dict]]]:
    chunks: list[list[tuple[int, dict]]] = [[] for _ in range(min(workers, len(rows)))]
    for index, row in enumerate(rows):
        chunks[index % len(chunks)].append((index, row))
    return chunks


def build_hard_targets(records: Sequence[dict], out: Path) -> dict:
    n = len(records)
    policy = np.zeros((n, POLICY_DIM), dtype=np.float16)
    confidence = np.empty(n, dtype=np.float32)
    indices = np.empty(n, dtype=np.uint16)
    for i, row in enumerate(records):
        idx = zq_move_to_policy_index(row["bestmove"], int(row["side_to_move"]))
        indices[i] = idx
        policy[i, idx] = np.float16(1.0)
        diagnostics = row["diagnostics"]
        vote = float(diagnostics.get("highest_budget_vote_fraction", 0.0))
        stable = bool(diagnostics.get("budget_winners_agree", False))
        confidence[i] = np.float32(vote * (1.0 if stable else 0.5))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        id=np.asarray([row["id"] for row in records], dtype="S24"),
        policy=policy,
        policy_idx=indices,
        confidence=confidence,
    )
    manifest = {
        "schema": "zquoridor.teacher.hard_targets.v1",
        "out": str(out),
        "samples": n,
        "teacher": records[0]["teacher"],
        "mode": "hard-policy-from-selected-bestmove",
        "confidence_mean": float(confidence.mean()),
        "stable_fraction": float(np.mean([
            bool(row["diagnostics"].get("budget_winners_agree", False)) for row in records
        ])),
        "unique_actions": int(len(np.unique(indices))),
        "has_value": False,
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_command(engine: str, engine_args: Sequence[str], protocol: str) -> tuple[str, ...]:
    command = [engine, *engine_args]
    if protocol == "titanium":
        command.append("uci")
    return tuple(command)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--protocol", choices=("uci", "titanium"), default="uci")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--budgets", default="200")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hard-targets-out", type=Path)
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.workers <= 0:
        raise SystemExit("repeats and workers must be positive")

    try:
        budgets = tuple(parse_budgets(args.budgets))
        positions = load_positions(args.positions)
        config = Config(
            teacher=args.teacher,
            protocol=args.protocol,
            command=build_command(args.engine, args.engine_arg, args.protocol),
            budgets=budgets,
            repeats=args.repeats,
        )
        chunks = split_chunks(positions, args.workers)
        indexed_records: list[tuple[int, dict]] = []
        if len(chunks) == 1:
            indexed_records.extend(relabel_chunk(chunks[0], config))
        else:
            with cf.ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                futures = [pool.submit(relabel_chunk, chunk, config) for chunk in chunks]
                for future in cf.as_completed(futures):
                    indexed_records.extend(future.result())
        indexed_records.sort(key=lambda item: item[0])
        records = [row for _, row in indexed_records]
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "zquoridor.teacher.probe_manifest.v1",
        "positions": str(args.positions),
        "out": str(args.out),
        "teacher": args.teacher,
        "protocol": args.protocol,
        "command": list(config.command),
        "budgets_ms": list(budgets),
        "repeats": args.repeats,
        "workers": len(chunks),
        "samples": len(records),
        "stable_fraction": float(np.mean([
            bool(row["diagnostics"].get("budget_winners_agree", False)) for row in records
        ])),
        "highest_budget_vote_mean": float(np.mean([
            float(row["diagnostics"].get("highest_budget_vote_fraction", 0.0)) for row in records
        ])),
    }
    if args.hard_targets_out is not None:
        manifest["hard_targets"] = build_hard_targets(records, args.hard_targets_out)
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
