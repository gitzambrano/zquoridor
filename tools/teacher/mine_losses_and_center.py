#!/usr/bin/env python3
"""Mine loss positions and center pawn collision states from benchmark matches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]


def hash_history(moves: List[str]) -> str:
    h = hashlib.sha256()
    h.update(" ".join(moves).encode("utf-8"))
    return h.hexdigest()[:24]


def is_center_square(move: str) -> bool:
    if len(move) == 2:
        col = move[0]
        row = move[1]
        return col in "cdefg" and row in "34567"
    return False


def mine_losses(
    games_file: Path,
    opponent_name: str,
    min_ply: int = 6,
    max_ply: int = 44,
) -> List[Dict]:
    positions = []
    if not games_file.exists():
        print(f"Warning: {games_file} does not exist", flush=True)
        return positions

    with open(games_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            winner = game.get("winner")
            zq_player = game.get("zq_player")
            if winner is None or winner == zq_player:
                continue

            moves = game.get("moves", [])
            opening_idx = game.get("opening_index", -1)

            # Extract positions where ZQuoridor was to move in the lost game
            for ply in range(min_ply, min(len(moves), max_ply) + 1):
                side_to_move = ply % 2
                if side_to_move != zq_player:
                    continue

                hist = moves[:ply]
                pos_id = hash_history(hist)

                # Tag critical characteristics
                tags = [f"loss_vs_{opponent_name}"]
                if ply <= 16:
                    tags.append("loss_opening")
                elif ply <= 30:
                    tags.append("loss_middlegame")
                else:
                    tags.append("loss_lategame")

                # Check for center pawn engagement
                has_center = any(is_center_square(m) for m in hist[-4:])
                if has_center:
                    tags.append("center_activity")

                positions.append({
                    "schema": "zquoridor.position.v1",
                    "id": pos_id,
                    "history": hist,
                    "side_to_move": side_to_move,
                    "split": "train" if (int(pos_id[:4], 16) % 10) != 0 else "val",
                    "source": f"benchmark-loss-{opponent_name}",
                    "opening_index": opening_idx,
                    "ply": ply,
                    "tags": tags,
                    "metadata": {
                        "opponent": opponent_name,
                        "reasons": tags,
                    },
                })
    return positions


def mine_center_rush_openings(openings_file: Path) -> List[Dict]:
    positions = []
    if not openings_file.exists():
        print(f"Warning: {openings_file} does not exist", flush=True)
        return positions

    with open(openings_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            moves = item.get("moves", [])
            category = item.get("category", "center_rush")
            opening_idx = item.get("opening_index", 0)

            # Generate states from prefix 4 to full length
            for ply in range(4, len(moves) + 1):
                hist = moves[:ply]
                pos_id = hash_history(hist)
                side_to_move = ply % 2
                tags = ["center_rush", category]
                if ply == len(moves):
                    tags.append("opening_branch_leaf")

                positions.append({
                    "schema": "zquoridor.position.v1",
                    "id": pos_id,
                    "history": hist,
                    "side_to_move": side_to_move,
                    "split": "train" if (int(pos_id[:4], 16) % 10) != 0 else "val",
                    "source": "center_rush_catalog",
                    "opening_index": opening_idx,
                    "ply": ply,
                    "tags": tags,
                    "metadata": {
                        "category": category,
                        "reasons": tags,
                    },
                })
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claustro-games",
        type=Path,
        default=ROOT / "results/benchmarks/race512-cr200k-champion-claustrophobia-600g/games.jsonl",
    )
    parser.add_argument(
        "--titanium-games",
        type=Path,
        default=ROOT / "results/benchmarks/race512-cr200k-champion-titanium-600g/games.jsonl",
    )
    parser.add_argument(
        "--center-openings",
        type=Path,
        default=ROOT / "tools/external/openings_center_rush_v1.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/teaching/loss-center-seeds/positions.jsonl",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    claustro_losses = mine_losses(args.claustro_games, "claustrophobia")
    titanium_losses = mine_losses(args.titanium_games, "titanium")
    center_rush_pos = mine_center_rush_openings(args.center_openings)

    all_positions = claustro_losses + titanium_losses + center_rush_pos

    # Deduplicate by history
    seen_histories: Set[str] = set()
    unique_positions = []
    for p in all_positions:
        key = " ".join(p["history"])
        if key not in seen_histories:
            seen_histories.add(key)
            unique_positions.append(p)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in unique_positions:
            f.write(json.dumps(p) + "\n")

    manifest = {
        "schema": "zquoridor.positions_manifest.v1",
        "total_mined": len(all_positions),
        "unique_positions": len(unique_positions),
        "claustrophobia_losses": len(claustro_losses),
        "titanium_losses": len(titanium_losses),
        "center_rush_catalog": len(center_rush_pos),
        "out": str(args.out),
    }
    manifest_file = args.out.with_suffix(".jsonl.manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
