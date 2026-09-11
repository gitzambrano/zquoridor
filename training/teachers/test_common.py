#!/usr/bin/env python3
"""Unit tests for teacher collection helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common import (
    deepest_rich_snapshot,
    load_independent_openings,
    parse_budgets,
    reaches_goal,
    summarize_probes,
)


class TeacherCommonTests(unittest.TestCase):
    def test_parse_budgets_deduplicates_in_order(self) -> None:
        self.assertEqual(parse_budgets("50,200,50,800"), [50, 200, 800])

    def test_deepest_rich_snapshot_prefers_depth_then_nodes(self) -> None:
        lines = [
            'info json {"searchDepth":5,"totalNodes":90,"rootMoves":[1]}',
            'info json {"searchDepth":7,"totalNodes":80,"rootMoves":[1,2]}',
            'info json {"searchDepth":7,"totalNodes":120,"rootMoves":[1]}',
            "info plain text",
        ]
        self.assertEqual(deepest_rich_snapshot(lines)["totalNodes"], 120)

    def test_goal_detection_uses_side_to_move(self) -> None:
        self.assertTrue(reaches_goal(0, "e9"))
        self.assertFalse(reaches_goal(0, "e1"))
        self.assertTrue(reaches_goal(1, "e1"))
        self.assertFalse(reaches_goal(1, "e9"))
        self.assertFalse(reaches_goal(0, "e8h"))

    def test_probe_summary_uses_highest_budget_majority(self) -> None:
        probes = [
            {"movetime_ms": 50, "bestmove": "e2"},
            {"movetime_ms": 50, "bestmove": "e2"},
            {"movetime_ms": 200, "bestmove": "e3"},
            {"movetime_ms": 200, "bestmove": "e3"},
            {"movetime_ms": 200, "bestmove": "e2"},
        ]
        summary = summarize_probes(probes, [50, 200])
        self.assertEqual(summary["selected_move"], "e3")
        self.assertAlmostEqual(summary["highest_budget_vote_fraction"], 2.0 / 3.0)
        self.assertFalse(summary["budget_winners_agree"])

    def test_frozen_opening_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen.jsonl"
            teacher = root / "teacher.jsonl"
            row = json.dumps({"moves": ["e2", "e8"]}) + "\n"
            frozen.write_text(row, encoding="utf-8")
            teacher.write_text(row, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_independent_openings(teacher, frozen)


if __name__ == "__main__":
    unittest.main()
