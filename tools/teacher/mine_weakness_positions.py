#!/usr/bin/env python3
"""Mine and export weakness positions from benchmark games.

Extracts critical positions from losses, draws, swept openings (0-2), and
wall-depletion crises against external opponents (Claustrophobia and Titanium).
Outputs architecture-neutral zquoridor.position.v1 JSONL.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]


def sample_id(history: Sequence[str]) -> str:
    payload = " ".join(move.strip().lower() for move in history).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def mine_positions(
    benchmarks_dir: Path,
    out_file: Path,
    max_positions: int = 35000,
    val_fraction: float = 0.15,
    seed: int = 20260919,
) -> dict:
    rng = random.Random(seed)
    files = list(benchmarks_dir.glob("**/games.jsonl"))
    if not files:
        raise FileNotFoundError(f"no games.jsonl found in {benchmarks_dir}")

    all_games = []
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    g = json.loads(line)
                    g["_source_file"] = str(f)
                    all_games.append(g)
                except Exception:
                    continue

    groups = {}
    for g in all_games:
        key = (g["_source_file"], g.get("opponent"), g.get("opening_index"))
        groups.setdefault(key, []).append(g)

    swept_openings = set()
    for key, games in groups.items():
        if len(games) == 2 and all(g.get("result", 1.0) == 0.0 for g in games):
            swept_openings.add(key)

    claustro_candidates: dict[str, dict] = {}
    titanium_candidates: dict[str, dict] = {}

    for g in all_games:
        opp = str(g.get("opponent", "")).lower()
        if opp not in ("claustrophobia", "titanium"):
            continue
        moves = g.get("moves", [])
        if len(moves) < 2:
            continue
        zq_player = int(g.get("zq_player", 0))
        result = float(g.get("result", 1.0))
        key = (g["_source_file"], opp, g.get("opening_index"))
        is_swept = key in swept_openings

        target_pool = claustro_candidates if opp == "claustrophobia" else titanium_candidates

        # Exclude terminal position (moves[:len(moves)])
        for ply in range(1, len(moves)):
            hist = moves[:ply]
            side = ply % 2
            is_zq_turn = (side == zq_player)

            reasons = []
            if result <= 0.5:
                if is_zq_turn:
                    reasons.append("zq_turn_in_loss")
                elif ply >= 20:
                    reasons.append("late_game_in_loss")
            if is_swept:
                reasons.append("swept_opening")

            if not reasons:
                continue

            sid = sample_id(hist)
            if sid not in target_pool:
                target_pool[sid] = {
                    "history": [m.strip().lower() for m in hist],
                    "side_to_move": side,
                    "opening_index": int(g.get("opening_index", -1)),
                    "ply": ply,
                    "opponent": opp,
                    "reasons": reasons,
                }
            else:
                for r in reasons:
                    if r not in target_pool[sid]["reasons"]:
                        target_pool[sid]["reasons"].append(r)

    # Balance sample selection between Claustrophobia and Titanium
    quota_per_opp = max_positions // 2
    claustro_items = list(claustro_candidates.values())
    titanium_items = list(titanium_candidates.values())

    rng.shuffle(claustro_items)
    rng.shuffle(titanium_items)

    selected_claustro = claustro_items[:quota_per_opp]
    selected_titanium = titanium_items[:quota_per_opp]

    combined = selected_claustro + selected_titanium
    rng.shuffle(combined)

    # Assign train / val splits deterministically
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_val = int(len(combined) * val_fraction)
    val_set = set(range(n_val))

    rows = []
    with out_file.open("w", encoding="utf-8") as fh:
        for idx, item in enumerate(combined):
            split = "val" if idx in val_set else "train"
            sid = sample_id(item["history"])
            rec = {
                "schema": "zquoridor.position.v1",
                "id": sid,
                "history": item["history"],
                "side_to_move": item["side_to_move"],
                "split": split,
                "source": f"benchmark-weakness-{item['opponent']}",
                "opening_index": item["opening_index"],
                "ply": item["ply"],
                "metadata": {
                    "opponent": item["opponent"],
                    "reasons": item["reasons"],
                },
            }
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            rows.append(rec)

    manifest = {
        "schema": "zquoridor.position_manifest.v1",
        "out": str(out_file),
        "total_positions": len(rows),
        "train_positions": len(rows) - n_val,
        "val_positions": n_val,
        "claustrophobia_positions": len(selected_claustro),
        "titanium_positions": len(selected_titanium),
        "swept_opening_pairs_found": len(swept_openings),
        "seed": seed,
    }
    manifest_path = out_file.with_suffix(out_file.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--out", type=Path, default=ROOT / "data/teaching/weakness-mining-35k/positions.jsonl")
    parser.add_argument("--max-positions", type=int, default=35000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)

    manifest = mine_positions(
        args.benchmarks, args.out, args.max_positions, args.val_fraction, args.seed
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
