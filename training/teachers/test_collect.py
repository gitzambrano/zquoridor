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


def make_config(out: Path, trajectory_source: str = "teacher") -> collect.CollectorConfig:
    return collect.CollectorConfig(
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
        trajectory_source=trajectory_source,
    )


class TeacherCollectTests(unittest.TestCase):
    def run_fake_collection(self, trajectory_source: str):
        temp = tempfile.TemporaryDirectory()
        out = Path(temp.name) / "out"
        config = make_config(out, trajectory_source)
        teacher_streams = [
            (50, 0, FakeEngine("teacher", 50)),
            (200, 0, FakeEngine("teacher", 200)),
        ]
        with mock.patch.object(collect, "make_teacher_streams", return_value=teacher_streams), \
             mock.patch.object(collect, "make_engine", return_value=FakeEngine("student")):
            manifest = collect.run_collection(config, [[]], workers=1)
        shard = out / "val" / "opening_000000.jsonl"
        records = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
        return temp, manifest, records

    def test_collection_records_student_disagreement(self) -> None:
        temp, manifest, records = self.run_fake_collection("teacher")
        try:
            self.assertEqual(manifest["samples"], 2)
            self.assertEqual(manifest["student"]["disagreements"], 2)
            record = records[0]
            self.assertEqual(record["bestmove"], "e3")
            self.assertEqual(record["played_move"], "e3")
            self.assertEqual(record["trajectory_source"], "teacher")
            self.assertFalse(record["diagnostics"]["student_agrees"])
            self.assertFalse(record["diagnostics"]["budget_winners_agree"])
            self.assertEqual(record["rich"]["searchDepth"], 7)
            self.assertEqual(record["side_to_move"], 0)
            self.assertTrue(record["id"])
        finally:
            temp.cleanup()

    def test_dagger_advances_by_student_move_but_keeps_teacher_label(self) -> None:
        temp, manifest, records = self.run_fake_collection("student")
        try:
            self.assertEqual(manifest["trajectory_source"], "student")
            self.assertEqual(len(records), 2)
            first, second = records
            self.assertEqual(first["bestmove"], "e3")
            self.assertEqual(first["played_move"], "e2")
            self.assertEqual(first["history"], [])
            # This is the DAgger invariant: the next labeled state is the state
            # the student actually reached, not the teacher's hypothetical line.
            self.assertEqual(second["history"], ["e2"])
            self.assertEqual(second["bestmove"], "e3")
            self.assertEqual(second["played_move"], "e2")
        finally:
            temp.cleanup()

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
