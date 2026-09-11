#!/usr/bin/env python3
import unittest

import numpy as np

from blunder_mine import compute_signals, select_windows
from targets import make_position


class BlunderMineTests(unittest.TestCase):
    def make_positions(self):
        histories = [[], ["e2"], ["e2", "e8"], ["e2", "e8", "e3"]]
        rows = []
        for ply, history in enumerate(histories):
            row = make_position(
                history,
                split="train",
                source="dagger",
                opening_index=7,
                ply=ply,
                metadata={"played_move": "e2"},
            )
            rows.append(row)
        return rows

    def test_action_q_is_preferred_and_window_is_centered(self):
        positions = self.make_positions()
        policy = np.zeros((4, 209), dtype=np.float32)
        policy[:, 10] = 0.8
        policy[:, 11] = 0.2
        value = np.zeros(4, dtype=np.float32)
        action_q = np.full((4, 209), np.nan, dtype=np.float32)
        action_q[:, 10] = 0.7
        action_q[:, 11] = 0.6
        # At ply 2 the student's played action 11 is a real blunder.
        action_q[2, 11] = -0.4
        targets = {"policy": policy, "value": value, "action_q": action_q}
        played = np.full(4, 11, dtype=np.int16)
        diagnostics = compute_signals(positions, targets, played)
        anchor = next(row for row in diagnostics if row["ply"] == 2)
        self.assertEqual(anchor["method"], "action-q")
        self.assertAlmostEqual(anchor["regret"], 1.1, places=5)

        windows, anchors = select_windows(
            positions, diagnostics, threshold=0.5, radius=1, max_anchors=0
        )
        self.assertEqual(len(anchors), 1)
        self.assertEqual([row["ply"] for row in windows], [1, 2, 3])
        self.assertEqual(windows[1]["metadata"]["blunder_offset"], 0)

    def test_value_drop_fallback_respects_side_to_move_flip(self):
        positions = self.make_positions()
        policy = np.zeros((4, 209), dtype=np.float32)
        policy[:, 10] = 1.0
        # V(s1)=+0.8 for mover; after that mover plays, next side-to-move
        # value is -0.1, so the previous mover still has +0.1 and lost 0.7.
        value = np.asarray([0.0, 0.8, -0.1, 0.0], dtype=np.float32)
        targets = {"policy": policy, "value": value}
        played = np.full(4, 10, dtype=np.int16)
        diagnostics = compute_signals(positions, targets, played)
        row = next(item for item in diagnostics if item["ply"] == 1)
        self.assertEqual(row["method"], "value-drop")
        self.assertAlmostEqual(row["regret"], 0.7, places=5)
        self.assertAlmostEqual(row["value_after_for_mover"], 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
