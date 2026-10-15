#!/usr/bin/env python3
"""Deterministic legal openings that trigger ONLY production AB endgame gates.

Every row starts from the standard initial position and follows the real
local_arena referee; no wall stocks or pawn locations are artificially edited.
For gate=mover_empty the side to move (P0) has 0 walls and P1 has 8: MCAB
evaluates leaves with endgameLeafDepth=2 (production default).
For gate=both_empty both players have 0 walls: MCAB delegates to pure
Negamax/endgame-race (production default).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.external.local_arena import Referee


def build_gate(index: int, gate: str, seed: int):
    rng = random.Random(seed + index * 1000003)
    for attempt in range(60):
        ref = Referee()
        moves = []
        # All P0 turns place a wall (10 in total); P1 places either 2 or 10.
        # After 20 plies, the side to move is always P0.
        try:
            for ply in range(20):
                current = ref.side_to_move
                place_wall = current == 0 or gate == "both_empty" or ply in (1, 13)
                all_legal = ref.legal_moves()
                if place_wall:
                    candidates = [m for m in all_legal if len(m) == 3]
                else:
                    # Maintain an actual pawn position away from the goal,
                    # with varied lateral/retreat moves (important for TT).
                    candidates = [m for m in all_legal if len(m) == 2 and int(m[1]) >= 3]
                if not candidates:
                    raise ValueError("no legal candidate")
                rng.shuffle(candidates)
                # Prefer dispersed walls so that the 20-wall final remains
                # a valid (and variable) maze, not an impossible placement.
                choice = candidates[0]
                ref.apply(choice)
                moves.append(choice)
            expected = (0, 0) if gate == "both_empty" else (0, 8)
            if ref.winner is not None or ref.side_to_move != 0 or tuple(ref.walls_left) != expected:
                raise ValueError("bad gate or premature terminal")
            return {"moves": moves, "category": gate, "wall_stocks": list(ref.walls_left)}
        except ValueError:
            rng.seed(seed + index * 1000003 + attempt + 1)
    raise RuntimeError(f"unable to generate legal {gate} fixture at index {index}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["mover_empty", "both_empty"], required=True)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    rows, unique = [], set()
    for i in range(args.count):
        r = build_gate(i, args.gate, args.seed)
        k = tuple(r["moves"])
        if k in unique:
            raise AssertionError(f"duplicate gate opening {i}")
        unique.add(k)
        rows.append(r)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Generated {len(rows)} verified {args.gate} positions -> {target}", flush=True)


if __name__ == "__main__":
    main()
