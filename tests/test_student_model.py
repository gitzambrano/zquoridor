"""Regression checks for portable student architectures."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


class StudentTests(unittest.TestCase):
    def test_race_features_encode_resource_asymmetry(self):
        import student_model as s
        data = dict(own_pawn=np.array([4, 4]), opp_pawn=np.array([76, 76]),
                    walls_h=np.zeros(2, dtype=np.uint64), walls_v=np.zeros(2, dtype=np.uint64),
                    own_dist=np.array([3, 3]), opp_dist=np.array([5, 5]),
                    walls_left_own=np.array([0, 2]), walls_left_opp=np.array([4, 4]))
        features = s.encode_features(data, np.arange(2), "race")
        self.assertEqual(features.shape, (2, 456))
        self.assertEqual(features[0, 354 + 14], 1)
        self.assertEqual(features[0, 387 + 6], 1)
        self.assertEqual(features[0, 408 + 3], 1)
        self.assertEqual(features[1, 408 + 11], 1)
        self.assertEqual(features[:, 354:].sum(), 6)

    def test_zero_expansion_preserves_production_output(self):
        import student_model as s
        baseline = s.Student("base", 256)
        baseline.load_float(ROOT / "checkpoints/gen9-teacher-head/nnue_weights.bin")
        candidate = s.Student("race", 256)
        candidate.warm_start(baseline)
        baseline.double()
        candidate.double()
        x = torch.rand(8, 354, dtype=torch.float64)
        expanded = torch.cat((x, torch.rand(8, 102, dtype=torch.float64)), dim=1)
        for old, new in zip(baseline(x), candidate(expanded)):
            torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)

    def test_width_expansion_and_export_roundtrip(self):
        import student_model as s
        old = s.Student("base", 256)
        new = s.Student("race", 384)
        new.warm_start(old)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.bin"
            s.export(new, path)
            loaded = s.Student("race", 384)
            loaded.load_float(path)
            for key, value in new.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(path.with_name("weights_int8.bin").exists())

    def test_incompatible_shrink_rejected(self):
        import student_model as s
        with self.assertRaises(ValueError):
            s.Student("base", 128).warm_start(s.Student("base", 256))


if __name__ == "__main__":
    unittest.main()
