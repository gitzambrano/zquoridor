#!/usr/bin/env python3
"""Compare frozen zQuoridor with a genuinely independent pinned Titanium search.

Produces raw diagnostic disagreements; does not train any model or assert truth.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.engine_lab.tactical_consensus_audit import (
    CONFIG as AUDIT_CONFIG, detect_benchmark_overlap, load_book,
    wall_properties,
)
from tools.external.local_arena import Referee, TitaniumPlayer, UciPlayer

CONFIG = {
    "titanium": "titanium-bin/titanium",
    "zquoridor": "bin/zq_teacher_compare",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "source": "tools/external/openings_targeted_weakness_bank.jsonl",
    "samples": 8,
    "seed": 20261106,
    "zq_movetime_ms": 200,
    "titanium_movetime_ms": 1000,
    "output": "results/titanium-tactical-probe",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("titanium", "zquoridor", "weights", "source", "output"):
        parser.add_argument("--" + key.replace("_", "-"), default=CONFIG[key])
    for key in ("samples", "seed", "zq_movetime_ms", "titanium_movetime_ms"):
        parser.add_argument("--" + key.replace("_", "-"), default=CONFIG[key], type=int)
    cfg = {**CONFIG, **vars(parser.parse_args())}
    if cfg["samples"] <= 0 or cfg["zq_movetime_ms"] <= 0 or cfg["titanium_movetime_ms"] <= 0:
        parser.error("sample count and time budgets must be positive")

    frozen = [
        tuple(item["moves"])
        for path in AUDIT_CONFIG["frozen_evaluation_paths"]
        for item in load_book(ROOT / path)
    ]
    candidates = load_book(ROOT / cfg["source"])
    indices = list(range(len(candidates)))
    random.Random(cfg["seed"] + 2705).shuffle(indices)
    selection = [
        i for i in indices
        if not detect_benchmark_overlap(candidates[i]["moves"], frozen)
        and any(len(move) == 3 for move in candidates[i]["moves"][-4:])
    ][:cfg["samples"]]
    if len(selection) < cfg["samples"]:
        raise RuntimeError("Insufficient nonbenchmark tactical seeds")

    rows = []
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    with UciPlayer([str(ROOT / cfg["zquoridor"]), "--nnue", str(ROOT / cfg["weights"])],
                   "frozen-zquoridor") as zq:
        with TitaniumPlayer(ROOT / cfg["titanium"]) as titanium:
            for index in selection:
                moves = candidates[index]["moves"]
                ref = Referee()
                for m in moves:
                    ref.apply(m)
                zq_move, zq_time, _ = zq.bestmove(
                    moves, budget=cfg["zq_movetime_ms"], timeout_s=20.0
                )
                ti_move, ti_time, ti_info = titanium.bestmove(
                    moves, budget=cfg["titanium_movetime_ms"], timeout_s=20.0
                )
                zq_move, ti_move = zq_move.lower(), ti_move.lower()
                if not ref.is_legal_move(zq_move) or not ref.is_legal_move(ti_move):
                    raise RuntimeError("Engine returned illegal move at source " + str(index))
                row = {
                    "source_index": index,
                    "history": moves,
                    "benchmark_overlap": False,
                    "zq_move": zq_move,
                    "titanium_move": ti_move,
                    "disagree": zq_move != ti_move,
                    "zq_wall": wall_properties(moves, zq_move),
                    "titanium_wall": wall_properties(moves, ti_move),
                    "zq_elapsed_s": zq_time,
                    "titanium_elapsed_s": ti_time,
                    "titanium_info": ti_info[-8:],
                    "label_quality": "independent_reference_unverified",
                }
                rows.append(row)
                (output / "positions.json").write_text(json.dumps(rows, indent=2) + "\n")
                print("POSITION " + json.dumps({
                    "index": index, "zq": zq_move, "titanium": ti_move,
                    "disagree": row["disagree"],
                    "titanium_neutral_wall": row["titanium_wall"]["path_distance_neutral"],
                }), flush=True)
    summary = {
        "samples": len(rows),
        "disagreements": sum(r["disagree"] for r in rows),
        "titanium_wall_choices": sum(r["titanium_wall"]["is_wall"] for r in rows),
        "titanium_neutral_walls": sum(r["titanium_wall"].get("path_distance_neutral") is True for r in rows),
        "zq_wall_choices": sum(r["zq_wall"]["is_wall"] for r in rows),
        "benchmark_overlaps": sum(r["benchmark_overlap"] for r in rows),
        "caveat": "Independent Titanium best moves are not proven correct; budgets differ and no Elo is measured.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
