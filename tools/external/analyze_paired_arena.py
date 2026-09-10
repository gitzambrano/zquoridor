#!/usr/bin/env python3
"""Pair-aware statistics for Zquoridor/Titanium arena JSONL.

The frozen arena reports its historical game-level interval.  This companion
keeps that result untouched and adds uncertainty where the independent unit is
the paired opening (same opening, colors swapped).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def elo_from_score(score: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, score))
    return 400.0 * math.log10(p / (1.0 - p))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    x = q * (len(sorted_values) - 1)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return sorted_values[lo]
    w = x - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def analyze(games_path: Path, bootstrap: int, seed: int) -> dict:
    if bootstrap < 1000:
        raise ValueError("bootstrap must be >= 1000")
    games = []
    for lineno, line in enumerate(games_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            opening = int(row["opening_index"])
            zq_player = int(row["zq_player"])
            result = float(row["result"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{games_path}:{lineno}: invalid arena row") from exc
        if zq_player not in (0, 1) or result not in (0.0, 0.5, 1.0):
            raise ValueError(f"{games_path}:{lineno}: invalid zq_player/result")
        games.append((opening, zq_player, result))
    if not games:
        raise ValueError("no games found")

    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for opening, zq_player, result in games:
        grouped[opening].append((zq_player, result))

    pair_points: list[float] = []
    pair_rows = []
    for opening in sorted(grouped):
        rows = grouped[opening]
        colors = sorted(player for player, _ in rows)
        if len(rows) != 2 or colors != [0, 1]:
            raise ValueError(
                f"opening {opening}: expected exactly two games with ZQ as P0/P1, got {rows}"
            )
        points = sum(result for _, result in rows)
        pair_points.append(points)
        pair_rows.append({
            "opening_index": opening,
            "points_0_to_2": points,
            "zq_p0": next(result for player, result in rows if player == 0),
            "zq_p1": next(result for player, result in rows if player == 1),
        })

    n = len(pair_points)
    score = sum(pair_points) / (2.0 * n)
    rng = random.Random(seed)
    boot_scores = []
    for _ in range(bootstrap):
        points = sum(pair_points[rng.randrange(n)] for _ in range(n))
        boot_scores.append(points / (2.0 * n))
    boot_scores.sort()
    score_lo = percentile(boot_scores, 0.025)
    score_hi = percentile(boot_scores, 0.975)

    by_color = {}
    for color in (0, 1):
        vals = [result for _, player, result in games if player == color]
        by_color[str(color)] = {
            "games": len(vals),
            "score_pct": 100.0 * sum(vals) / len(vals),
        }

    return {
        "schema": "zquoridor.paired_arena.v1",
        "games": len(games),
        "paired_openings": n,
        "score_pct": 100.0 * score,
        "elo": elo_from_score(score),
        "paired_bootstrap": {
            "iterations": bootstrap,
            "seed": seed,
            "score_95_low_pct": 100.0 * score_lo,
            "score_95_high_pct": 100.0 * score_hi,
            "elo_95_low": elo_from_score(score_lo),
            "elo_95_high": elo_from_score(score_hi),
        },
        "pair_outcomes": {
            "above_1_point": sum(x > 1.0 for x in pair_points),
            "exactly_1_point": sum(x == 1.0 for x in pair_points),
            "below_1_point": sum(x < 1.0 for x in pair_points),
        },
        "by_zq_color": by_color,
        "pairs": pair_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    report = analyze(args.games, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
