#!/usr/bin/env python3
"""Unit tests for active teacher collection and sample selection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import collect
from select_samples import should_select


class FakeEngine:
    """Return deterministic moves for an isolated collector test."""

    def __init__(self, role: str, budget: int | None = None):
        self.role = role
        self.budget = budget

    def bestmove(self, history, movetime_ms):
        if len(history) >= 2:
            return "(none)", 0.001, []
        if self.role == "student":
            return "e2", 0.001, ["info student"]
        move = "e3" if movetime_ms >= 200 else "e2"
        info = [
            f'info json {{"searchDepth":7,"totalNodes":{movetime_ms},'
            f'"rootMoves":[{{"move":"{move}","score":1}}]}}'
        ]
        return move, 0.001, info

    def close(self):
        return None


class TeacherCollectTests(unittest.TestCase):
    def test_collection_records_student_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            config = collect.CollectorConfig(
                teacher="Teacher",
                protocol="uci",
                command=("teacher",),
                budgets=(50, 200),
                repeats=1,
                max_plies=8,
                out_dir=out,
                val_mod=2,
                student="Student",
                student_protocol="uci",
                student_command=("student",),
                student_movetime_ms=200,
            )

            teacher_streams = [
                (50, 0, FakeEngine("teacher", 50)),
                (200, 0, FakeEngine("teacher", 200)),
            ]
            with mock.patch.object(collect, "make_teacher_streams", return_value=teacher_streams), \
                 mock.patch.object(collect, "make_engine", return_value=FakeEngine("student")):
                manifest = collect.run_collection(config, [[]], workers=1)

            self.assertEqual(manifest["samples"], 2)
            self.assertEqual(manifest["student"]["disagreements"], 2)
            shard = out / "val" / "opening_000000.jsonl"
            record = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["bestmove"], "e3")
            self.assertFalse(record["diagnostics"]["student_agrees"])
            self.assertFalse(record["diagnostics"]["budget_winners_agree"])
            self.assertEqual(record["rich"]["searchDepth"], 7)

    def test_disagreement_selector_requires_confident_teacher(self) -> None:
        record = {
            "diagnostics": {
                "highest_budget_vote_fraction": 1.0,
                "budget_winners_agree": True,
                "student_agrees": False,
            }
        }
        self.assertTrue(should_select(record, "disagreement", 1.0, True))
        record["diagnostics"]["highest_budget_vote_fraction"] = 0.5
        self.assertFalse(should_select(record, "disagreement", 1.0, True))

    def test_unstable_selector_marks_budget_disagreement(self) -> None:
        record = {
            "diagnostics": {
                "highest_budget_vote_fraction": 1.0,
                "budget_winners_agree": False,
            }
        }
        self.assertTrue(should_select(record, "unstable", 1.0, True))


if __name__ == "__main__":
    unittest.main()
