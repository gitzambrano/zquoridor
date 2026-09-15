"""Validate training input boundaries before starting expensive work."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))


class ExperimentTests(unittest.TestCase):
    def test_cli_only_overrides_supplied_settings(self):
        import run_experiment as r
        config = r.parse_config(["--hidden", "384", "--no-qat"])
        self.assertEqual(config["hidden"], 384)
        self.assertFalse(config["qat"])
        self.assertEqual(config["epochs"], r.CONFIG["epochs"])

    def test_validation_split_is_required(self):
        import run_experiment as r
        with self.assertRaisesRegex(ValueError, "validation"):
            r.split_indices({"is_val": np.array([False, False])})

    def test_group_leak_is_rejected(self):
        import run_experiment as r
        with self.assertRaisesRegex(ValueError, "group"):
            r.split_indices({"is_val": np.array([False, True]), "group_id": np.array(["x", "x"])})


if __name__ == "__main__":
    unittest.main()
