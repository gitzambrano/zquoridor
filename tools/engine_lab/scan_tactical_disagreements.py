#!/usr/bin/env python3
"""Identify time-sensitive wall choices without treating deep search as ground truth."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "exe": "bin/zq_tactical",
    "weights": "data/nnue/nnue_weights_int8.bin",
    "books": {
        "normal": "tools/external/openings_normal_confirm_400.jsonl",
        "center": "tools/external/openings_center_rush_sound_5k.jsonl",
        "targeted": "tools/external/openings_targeted_weakness_bank.jsonl",
    },
    "category": "center",
    "count": 12,
    "seed": 20261107,
    "shallow_ms": 200,
    "deep_ms": 1200,
    "output": "results/tactical-scans",
}


def evaluate(exe: str, weights: str, moves: list[str], limit_ms: int) -> dict:
    script = (
        "uci\n"
        + "position startpos" + (" moves " + " ".join(moves) if moves else "") + "\n"
        + f"go movetime {limit_ms}\n"
        + "quit\n"
    )
    run = subprocess.run(
        [exe, "--nnue", weights, "--dump-root"],
        input=script, text=True, capture_output=True,
        timeout=45, check=True,
    )
    roots = []
    for line in run.stdout.splitlines():
        if "root_json " in line:
            roots.append(json.loads(line.split("root_json ", 1)[1]))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one root output, found {len(roots)}: {run.stdout[-700:]}")
    root = roots[0]
    if root.get("target_status") != "ok":
        raise RuntimeError(f"Root has no visits: {root.get('target_status')}")
    return root


def inspect(shallow: dict, deep: dict, moves: list[str], index: int, category: str) -> dict:
    best_s = shallow["bestmove"]
    best_d = deep["bestmove"]
    before = shallow["visit_counts"]
    after = deep["visit_counts"]
    best_idx = int(deep["best_action"])
    prior_rank = sorted(range(len(deep["priors"])), key=lambda i: -deep["priors"][i])
    return {
        "source_index": index,
        "book": category,
        "history": moves,
        "shallow_move": best_s,
        "deep_move": best_d,
        "different": best_s != best_d,
        "deep_wall": len(best_d) == 3,
        "shallow_wall": len(best_s) == 3,
        "shallow_simulations": shallow["simulations"],
        "deep_simulations": deep["simulations"],
        "deep_best_prior_rank": prior_rank.index(best_idx) + 1,
        "deep_best_visits": after[best_idx],
        "shallow_best_visits": before[int(shallow["best_action"])],
        "deep_root_value": deep.get("root_value"),
        "shallow_root_value": shallow.get("root_value"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default=CONFIG["exe"])
    parser.add_argument("--weights", default=CONFIG["weights"])
    parser.add_argument("--category", choices=CONFIG["books"], default=CONFIG["category"])
    parser.add_argument("--count", type=int, default=CONFIG["count"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--shallow-ms", type=int, default=CONFIG["shallow_ms"])
    parser.add_argument("--deep-ms", type=int, default=CONFIG["deep_ms"])
    parser.add_argument("--output", default=CONFIG["output"])
    args = parser.parse_args()
    cfg = {**CONFIG, **vars(args)}
    if cfg["count"] <= 0 or cfg["shallow_ms"] <= 0 or cfg["deep_ms"] <= cfg["shallow_ms"]:
        parser.error("Expected positive count and a longer deep search")
    path = ROOT / cfg["books"][cfg["category"]]
    book = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = list(range(len(book)))
    random.Random(cfg["seed"] + {"normal": 1, "center": 2, "targeted": 3}[cfg["category"]]).shuffle(ids)
    rows = []
    for i in ids[:cfg["count"]]:
        moves = book[i]["moves"]
        shallow = evaluate(cfg["exe"], cfg["weights"], moves, cfg["shallow_ms"])
        deep = evaluate(cfg["exe"], cfg["weights"], moves, cfg["deep_ms"])
        result = inspect(shallow, deep, moves, i, cfg["category"])
        rows.append(result)
        print("ROW " + json.dumps({key: result[key] for key in (
            "source_index", "shallow_move", "deep_move", "different",
            "deep_wall", "deep_best_prior_rank"
        )}), flush=True)

    differences = [row for row in rows if row["different"]]
    wall_shifts = [row for row in differences if row["deep_wall"] and not row["shallow_wall"]]
    summary = {
        "book": cfg["category"],
        "count": len(rows),
        "different": len(differences),
        "different_pct": 100 * len(differences) / len(rows),
        "pawn_to_wall": len(wall_shifts),
        "deep_wall_changes": sum(row["deep_wall"] for row in differences),
        "mean_deep_best_prior_rank": sum(row["deep_best_prior_rank"] for row in rows) / len(rows),
        "caveat": "Deeper search is a reference for disagreements, not an oracle or established winning move.",
    }
    output = ROOT / cfg["output"] / cfg["category"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "positions.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
