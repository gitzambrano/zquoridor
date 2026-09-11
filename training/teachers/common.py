#!/usr/bin/env python3
"""Common helpers for architecture-neutral teacher data collection."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

INFO_JSON_PREFIX = "info json "


def read_openings(path: Path) -> list[list[str]]:
    """Read one JSON object with a move list from each non-empty line."""
    openings: list[list[str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        moves = obj.get("moves")
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise ValueError(f"{path}:{lineno}: expected a string list in 'moves'")
        openings.append(moves)
    if not openings:
        raise ValueError(f"no openings found in {path}")
    return openings


def load_independent_openings(path: Path, frozen_path: Path) -> list[list[str]]:
    """Load teacher openings and reject overlap with the frozen benchmark."""
    if path.resolve() == frozen_path.resolve():
        raise ValueError("refusing to use the frozen benchmark openings as teacher data")
    openings = read_openings(path)
    frozen = {tuple(moves) for moves in read_openings(frozen_path)}
    overlap = [index for index, moves in enumerate(openings) if tuple(moves) in frozen]
    if overlap:
        preview = ",".join(map(str, overlap[:10]))
        raise ValueError(
            f"teacher corpus overlaps the frozen benchmark at {len(overlap)} opening(s) "
            f"(indices {preview})"
        )
    return openings


def parse_budgets(text: str) -> list[int]:
    """Parse a comma-separated list of positive search budgets."""
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("teacher budgets must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one teacher budget is required")
    return values


def reaches_goal(pre_move_ply: int, move: str) -> bool:
    """Return true when a pawn move reaches the mover's goal rank."""
    if len(move) != 2 or move[0] not in "abcdefghi" or move[1] not in "123456789":
        return False
    side = pre_move_ply & 1
    return (side == 0 and move[1] == "9") or (side == 1 and move[1] == "1")


def rich_info_snapshots(info_lines: Sequence[str]) -> list[dict]:
    """Parse valid `info json` objects from one engine search."""
    snapshots: list[dict] = []
    for line in info_lines:
        if not line.startswith(INFO_JSON_PREFIX):
            continue
        try:
            obj = json.loads(line[len(INFO_JSON_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            snapshots.append(obj)
    return snapshots


def deepest_rich_snapshot(info_lines: Sequence[str]) -> dict | None:
    """Return the richest snapshot at the deepest reported search depth."""
    snapshots = rich_info_snapshots(info_lines)
    if not snapshots:
        return None

    def key(obj: dict) -> tuple[int, int, int]:
        depth = int(obj.get("searchDepth", obj.get("mainCompletedDepth", -1)) or -1)
        nodes = int(obj.get("totalNodes", obj.get("nodes", 0)) or 0)
        root_moves = obj.get("rootMoves")
        move_count = len(root_moves) if isinstance(root_moves, list) else 0
        return depth, nodes, move_count

    return max(snapshots, key=key)


def _winner(moves: Sequence[str]) -> tuple[str, float]:
    """Return the majority move and its vote fraction."""
    if not moves:
        return "(none)", 0.0
    counts = Counter(moves)
    first_index = {move: moves.index(move) for move in counts}
    move = min(counts, key=lambda item: (-counts[item], first_index[item], item))
    return move, counts[move] / len(moves)


def summarize_probes(probes: Sequence[dict], budgets: Sequence[int]) -> dict:
    """Summarize move stability across budgets and repeated searches."""
    by_budget: dict[str, dict] = {}
    budget_winners: list[str] = []
    for budget in budgets:
        moves = [
            str(probe["bestmove"])
            for probe in probes
            if int(probe["movetime_ms"]) == budget and probe.get("bestmove") != "(none)"
        ]
        winner, vote_fraction = _winner(moves)
        by_budget[str(budget)] = {
            "bestmove": winner,
            "vote_fraction": vote_fraction,
            "unique_moves": len(set(moves)),
            "samples": len(moves),
        }
        if winner != "(none)":
            budget_winners.append(winner)

    highest_budget = max(budgets)
    high = by_budget[str(highest_budget)]
    valid_moves = [
        str(probe["bestmove"])
        for probe in probes
        if probe.get("bestmove") != "(none)"
    ]
    all_move, all_vote_fraction = _winner(valid_moves)
    unique_budget_winners = set(budget_winners)
    return {
        "selected_move": str(high["bestmove"]),
        "highest_budget_ms": highest_budget,
        "highest_budget_vote_fraction": float(high["vote_fraction"]),
        "all_probe_majority_move": all_move,
        "all_probe_vote_fraction": all_vote_fraction,
        "budget_winners_agree": len(unique_budget_winners) <= 1,
        "budget_winner_count": len(unique_budget_winners),
        "by_budget": by_budget,
    }
