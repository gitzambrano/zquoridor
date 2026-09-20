#!/usr/bin/env python3
"""Analyze benchmark games breaking down central vs flank/wall openings."""
import argparse
import json
import sys
from pathlib import Path


def is_central_opening(moves: list[str]) -> bool:
    """Check if opening (first 6 plies) involves pawn advances toward the center."""
    op = moves[:6]
    # Check if any pawn moves occur, especially along center columns (c, d, e, f, g)
    for m in op:
        if len(m) == 2:  # pawn move like 'e2', 'e8', 'd3', 'f7'
            return True
    return False


def is_pure_center_rush(moves: list[str]) -> bool:
    """Check if opening starts with the classic e2/e8 pawn rush."""
    if len(moves) >= 4:
        return moves[0] == "e2" and moves[1] == "e8"
    return False


def analyze_games(games_path: Path):
    if not games_path.exists():
        print(f"Error: file not found: {games_path}", file=sys.stderr)
        return

    lines = [json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        print("No games found in file.", file=sys.stderr)
        return

    opponent = lines[0].get("opponent", "opponent")
    total_games = len(lines)
    total_score = sum(g["result"] for g in lines)
    total_pct = (total_score / total_games) * 100.0

    print("=" * 60)
    print(f"BENCHMARK BREAKDOWN: vs {opponent.upper()} ({total_games} games)")
    print(f"Overall Score: {total_score:.1f}/{total_games} ({total_pct:.2f}%)")
    print("-" * 60)

    # Categories
    categories = {
        "All Games": lines,
        "Central/Pawn Openings": [g for g in lines if is_central_opening(g.get("moves", []))],
        "Pure Center Rush (e2-e8)": [g for g in lines if is_pure_center_rush(g.get("moves", []))],
        "Wall/Flank Openings": [g for g in lines if not is_central_opening(g.get("moves", []))],
    }

    print(f"{'Category':<28} | {'Total':<10} | {'Score %':<10} | {'White (P0)':<12} | {'Black (P1)':<12}")
    print("-" * 80)

    for name, subset in categories.items():
        if not subset:
            continue
        n = len(subset)
        score = sum(g["result"] for g in subset)
        pct = (score / n) * 100.0

        white = [g for g in subset if g.get("zq_player") == 0]
        black = [g for g in subset if g.get("zq_player") == 1]

        w_score = sum(g["result"] for g in white) if white else 0.0
        w_pct = (w_score / len(white) * 100.0) if white else 0.0
        b_score = sum(g["result"] for g in black) if black else 0.0
        b_pct = (b_score / len(black) * 100.0) if black else 0.0

        w_str = f"{w_score:.1f}/{len(white)} ({w_pct:.1f}%)"
        b_str = f"{b_score:.1f}/{len(black)} ({b_pct:.1f}%)"

        print(f"{name:<28} | {f'{score:.1f}/{n}':<10} | {f'{pct:.2f}%':<10} | {w_str:<12} | {b_str:<12}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark games by opening type.")
    parser.add_argument("games", type=Path, help="Path to games.jsonl")
    args = parser.parse_args()
    analyze_games(args.games)


if __name__ == "__main__":
    main()
