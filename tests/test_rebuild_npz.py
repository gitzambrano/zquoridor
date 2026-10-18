import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from tools.rebuild_npz_from_jsonl import rebuild_targets_npz
from training.teachers.targets import dumps, make_label, make_position, POLICY_DIM


class TestRebuildTargetsNpz(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rebuild_valid_records(self):
        pos_file = self.dataset_dir / "positions.jsonl"
        tgt_file = self.dataset_dir / "targets.jsonl"

        # Create 15 mock records
        policy_template = [0.0] * POLICY_DIM
        policy_template[0] = 0.6
        policy_template[1] = 0.4

        with pos_file.open("w", encoding="utf-8") as f_p, tgt_file.open("w", encoding="utf-8") as f_t:
            for i in range(15):
                pos = make_position(["e2", "e8"], split="train", source="test")
                pos["id"] = f"test_{i}"
                lbl = make_label(
                    pos,
                    teacher="zquoridor-4.01",
                    mode="search-root",
                    policy=policy_template,
                    value=0.25,
                    metadata={"aux": [4, 4, 10, 10, 2, 0]},
                )
                f_p.write(dumps(pos) + "\n")
                f_t.write(dumps(lbl) + "\n")

        # Initial NPZ with corrupted zero-policy on first 5 rows
        corrupted_npz = self.dataset_dir / "targets.npz"
        bad_policy = np.zeros((15, POLICY_DIM), dtype=np.float32)
        bad_policy[5:] = np.asarray(policy_template, dtype=np.float32)
        np.savez_compressed(
            corrupted_npz,
            policy=bad_policy,
            value=np.zeros(15, dtype=np.float32),
            aux=np.zeros((15, 6), dtype=np.int32),
            sample_ids=np.array([f"test_{i}" for i in range(15)]),
        )

        # Run rebuild
        stats = rebuild_targets_npz(self.dataset_dir, backup=True)
        self.assertEqual(stats["total_records"], 15)
        self.assertEqual(stats["zero_policy_rows"], 0)
        self.assertIn("backup_file", stats)
        self.assertTrue(Path(stats["backup_file"]).exists())

        # Verify rebuilt NPZ
        with np.load(corrupted_npz) as data:
            self.assertEqual(data["policy"].shape, (15, POLICY_DIM))
            self.assertEqual(data["value"].shape, (15,))
            self.assertEqual(data["aux"].shape, (15, 6))
            self.assertEqual(len(data["sample_ids"]), 15)
            # Ensure every single row has positive policy mass sum
            sums = data["policy"].sum(axis=1)
            self.assertTrue(np.all(sums > 0.99))
            self.assertTrue(np.allclose(data["value"], 0.25))


if __name__ == "__main__":
    unittest.main()
