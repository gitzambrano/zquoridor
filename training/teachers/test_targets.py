#!/usr/bin/env python3
import math
import unittest

from targets import (
    POLICY_DIM,
    claustrophobia_policy_to_zq,
    js_divergence,
    make_label,
    make_position,
    mirror_action_lr,
    normalize_policy,
    sample_id,
)


class TargetSchemaTests(unittest.TestCase):
    def test_sample_id_is_stable(self):
        self.assertEqual(sample_id(["E2", " e8 "]), sample_id(["e2", "e8"]))
        self.assertNotEqual(sample_id(["e2"]), sample_id(["e3"]))

    def test_left_right_action_mirror_is_self_inverse(self):
        for idx in range(POLICY_DIM):
            self.assertEqual(mirror_action_lr(mirror_action_lr(idx)), idx)

    def test_claustrophobia_mapping_side_zero_is_identity(self):
        p = [0.0] * POLICY_DIM
        p[17] = 0.2
        p[93] = 0.3
        p[177] = 0.5
        self.assertEqual(claustrophobia_policy_to_zq(p, 0), p)

    def test_claustrophobia_mapping_side_one_flips_columns_only(self):
        p = [0.0] * POLICY_DIM
        # Pawn canonical e3 -> row 2, col 4: center file stays fixed.
        p[2 * 9 + 4] = 0.1
        # Pawn a3 -> i3.
        p[2 * 9 + 0] = 0.2
        # H slot (row 3,col 1) -> (row 3,col 6).
        p[81 + 3 * 8 + 1] = 0.3
        # V slot same transform.
        p[145 + 5 * 8 + 7] = 0.4
        zq = claustrophobia_policy_to_zq(p, 1)
        self.assertAlmostEqual(zq[2 * 9 + 4], 0.1)
        self.assertAlmostEqual(zq[2 * 9 + 8], 0.2)
        self.assertAlmostEqual(zq[81 + 3 * 8 + 6], 0.3)
        self.assertAlmostEqual(zq[145 + 5 * 8 + 0], 0.4)
        self.assertAlmostEqual(sum(zq), 1.0)

    def test_normalize_and_js(self):
        p = normalize_policy([1.0] + [0.0] * (POLICY_DIM - 1))
        self.assertEqual(sum(p), 1.0)
        self.assertAlmostEqual(js_divergence(p, p), 0.0)
        q = normalize_policy([0.0, 1.0] + [0.0] * (POLICY_DIM - 2))
        self.assertGreater(js_divergence(p, q), 0.6)

    def test_position_and_label_schema(self):
        position = make_position(["e2", "e8"], split="train", source="dagger")
        policy = [0.0] * POLICY_DIM
        policy[22] = 1.0
        label = make_label(
            position,
            teacher="claustrophobia-v1.3.1",
            mode="network",
            policy=policy,
            value=0.25,
            budget={"forward": 1},
        )
        self.assertEqual(label["id"], position["id"])
        self.assertEqual(label["target"]["policy_top1"], 22)
        self.assertTrue(math.isfinite(label["target"]["policy_entropy"]))
        self.assertEqual(label["target"]["value"], 0.25)


if __name__ == "__main__":
    unittest.main()
