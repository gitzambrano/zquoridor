#!/usr/bin/env python3
"""Materialize a cheap race-aware MCAB leaf-value experiment.

The production MCAB uses leafDepth=0, so almost every newly expanded leaf is
scored by the static NNUE value head.  In low-wall races that head can be very
flat with respect to pawn progress.  This experiment blends the NNUE Q with a
very cheap tempo proxy only once a race becomes relevant.

This is intentionally an experiment patcher, not production code.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", type=float, required=True)
    ap.add_argument("--tempo-scale", type=float, default=1.5)
    ap.add_argument(
        "--activation",
        choices=("either-empty", "mover-empty", "total-le2"),
        default="either-empty",
    )
    args = ap.parse_args()
    if not (0.0 <= args.blend <= 1.0):
        raise SystemExit("--blend must be in [0,1]")
    if args.tempo_scale <= 0.0:
        raise SystemExit("--tempo-scale must be > 0")

    path = Path("src/mcab.hpp")
    text = path.read_text()

    old = """                int score = nnueEvalInt(ap, node.side);\n                mstats.leafSearches++;\n                mstats.leafDepthSum += 0;\n                return scoreToQ(score, params.scoreScale);\n"""

    if args.activation == "either-empty":
        activation = "node.state.wallsLeft[0] == 0 || node.state.wallsLeft[1] == 0"
    elif args.activation == "mover-empty":
        activation = "node.state.wallsLeft[node.side] == 0"
    else:
        activation = "(int)node.state.wallsLeft[0] + (int)node.state.wallsLeft[1] <= 2"

    new = f"""                int score = nnueEvalInt(ap, node.side);\n                mstats.leafSearches++;\n                mstats.leafDepthSum += 0;\n                double q = scoreToQ(score, params.scoreScale);\n\n                // exp/vnext: the Gen8 WL head is known to become nearly flat\n                // in low-wall pawn races.  Add a zero-BFS tempo signal instead\n                // of paying for an AB leaf.  The proxy is deliberately simple:\n                // remaining goal-row distance for each pawn, from the leaf\n                // mover's perspective.  Quoridor is 9x9, hence goal rows 0/8.\n                // This is gated to race-like positions so midgame value remains\n                // bit-identical to production.\n                if ({activation}) {{\n                    const int self = node.side;\n                    const int opp = 1 - self;\n                    const int selfRow = (int)node.state.pawn[self] / 9;\n                    const int oppRow = (int)node.state.pawn[opp] / 9;\n                    const int selfDist = self == 0 ? 8 - selfRow : selfRow;\n                    const int oppDist = opp == 0 ? 8 - oppRow : oppRow;\n                    const int tempoAdv = oppDist - selfDist;\n                    const float z = (float)(-(double)tempoAdv / {args.tempo_scale:.9g});\n                    const double raceQ = 1.0 / (1.0 + (double)mcabFastExp(z));\n                    constexpr double raceBlend = {args.blend:.9g};\n                    q = (1.0 - raceBlend) * q + raceBlend * raceQ;\n                }}\n                return q;\n"""

    if old not in text:
        raise SystemExit("target leafDepth=0 block not found; source drifted")
    if text.count(old) != 1:
        raise SystemExit("target block is not unique")
    path.write_text(text.replace(old, new))
    print(
        f"patched src/mcab.hpp: race-value blend={args.blend:g} "
        f"scale={args.tempo_scale:g} activation={args.activation}"
    )


if __name__ == "__main__":
    main()
