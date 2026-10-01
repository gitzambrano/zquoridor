"""Validate training input boundaries before starting expensive work."""
import sys
import unittest
import tempfile
import hashlib
import json
from unittest import mock
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

    def test_cosine_schedule_warms_up_then_anneals_to_minimum(self):
        import run_experiment as r
        config = dict(r.CONFIG, epochs=10, lr=1e-3, min_lr=1e-5,
                      warmup_epochs=2, schedule="cosine")
        self.assertAlmostEqual(r.learning_rate(config, 0), 5e-4)
        self.assertAlmostEqual(r.learning_rate(config, 1), 1e-3)
        self.assertLess(r.learning_rate(config, 8), r.learning_rate(config, 2))
        self.assertAlmostEqual(r.learning_rate(config, 9), 1e-5)

    def test_cosine_schedule_supports_zero_warmup(self):
        import run_experiment as r
        config = dict(r.CONFIG, epochs=5, lr=1e-3, min_lr=1e-5,
                      warmup_epochs=0, schedule="cosine")
        self.assertAlmostEqual(r.learning_rate(config, 0), 1e-3)
        self.assertAlmostEqual(r.learning_rate(config, 4), 1e-5)

    def test_weight_decay_anneals_to_configured_minimum(self):
        import run_experiment as r
        config = dict(r.CONFIG, epochs=10, weight_decay=1e-5,
                      min_weight_decay=1e-7, weight_decay_schedule="cosine")
        self.assertAlmostEqual(r.weight_decay(config, 0), 1e-5)
        self.assertLess(r.weight_decay(config, 5), r.weight_decay(config, 0))
        self.assertAlmostEqual(r.weight_decay(config, 9), 1e-7)

    def test_source_weight_boosts_scale_selected_ranges_and_clip(self):
        import run_experiment as r
        weights = np.array([1.0, 2.0, 10.0, 20.0, 30.0], dtype=np.float32)
        boosted = r.apply_weight_boosts(
            weights,
            [[1, 3, 1.5], [3, 5, 1.25]],
            max_weight=30.0,
        )
        np.testing.assert_array_equal(boosted, [1.0, 3.0, 15.0, 25.0, 30.0])

    def test_checkpoints_resume_and_report_best_epoch(self):
        import run_experiment as r
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.npz"
            policy = np.zeros((2, 209), dtype=np.float32)
            policy[:, 0] = 1
            np.savez(dataset,
                     own_pawn=np.array([4, 4]), opp_pawn=np.array([76, 76]),
                     walls_h=np.zeros(2, dtype=np.uint64), walls_v=np.zeros(2, dtype=np.uint64),
                     own_dist=np.array([8, 8]), opp_dist=np.array([8, 8]),
                     walls_left_own=np.array([10, 10]), walls_left_opp=np.array([10, 10]),
                     policy=policy, value=np.array([-1, 1], dtype=np.float32),
                     weight=np.ones(2, dtype=np.float32), is_val=np.array([False, True]))
            config = dict(r.CONFIG, data=str(dataset), out_dir=str(root / "run"),
                          architecture="base", hidden=128, from_scratch=True, qat=False,
                          mirror_h=False, epochs=2, batch_size=2, patience=0,
                          checkpoint_every=1, warmup_epochs=0, device="cpu", cpu_threads=1,
                          resume=True)
            losses = iter([0.5, 0.4, 0.5, 0.3])

            def fake_epoch(model, data, indices, config, device, optimizer=None, rng=None):
                loss = next(losses) if optimizer is None else 1.0
                return {"loss": loss, "policy_kl": 0.0, "value_mae": 0.0}

            with mock.patch.object(r, "_epoch", side_effect=fake_epoch), \
                 mock.patch.object(r, "export", return_value={"architecture": "base"}):
                save = r._atomic_torch_save

                def interrupt_after_epoch(state, path):
                    save(state, path)
                    if Path(path).name == "epoch_0001.pt":
                        raise RuntimeError("simulated interruption")

                with mock.patch.object(r, "_atomic_torch_save", side_effect=interrupt_after_epoch):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        r.train(config)
                run_dir = root / "run"
                initial_bytes = (run_dir / "initial.pt").read_bytes()
                second = r.train(config)

            self.assertEqual(second["training_status"], "complete")
            self.assertEqual(second["configured_epochs"], 2)
            self.assertEqual(second["best_epoch"], 2)
            self.assertEqual(second["epochs"], 2)
            self.assertEqual((run_dir / "initial.pt").read_bytes(), initial_bytes)
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "epoch_0001.pt").is_file())
            self.assertTrue((run_dir / "epoch_0002.pt").is_file())
            with self.assertRaisesRegex(ValueError, "resume.pt exists"):
                r.train(dict(config, resume=False))

    def test_memmap_directory_loads_verified_manifest_arrays(self):
        import run_experiment as r
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = np.zeros((2, 209), dtype=np.float32)
            policy[:, 0] = 1
            arrays = {
                "own_pawn": np.array([4, 4]), "opp_pawn": np.array([76, 76]),
                "walls_h": np.zeros(2, dtype=np.uint64), "walls_v": np.zeros(2, dtype=np.uint64),
                "own_dist": np.array([8, 8]), "opp_dist": np.array([8, 8]),
                "walls_left_own": np.array([10, 10]), "walls_left_opp": np.array([10, 10]),
                "policy": policy, "value": np.array([-1, 1], dtype=np.float32),
                "weight": np.ones(2, dtype=np.float32), "is_val": np.array([False, True]),
                "source_mass": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            }
            records = {}
            for key, values in arrays.items():
                filename = f"fields/{key}.npy"
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, values)
                records[key] = {"filename": filename,
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            (root / "dataset.manifest.json").write_text(json.dumps({
                "complete": True, "arrays": records,
                "source_names": ["old", "new"],
                "group_separation_verified": True,
            }), encoding="utf-8")
            data = r.load_dataset(root)
            self.assertIsInstance(data["value"], np.memmap)
            self.assertEqual(data["source_names"], ["old", "new"])
            train_idx, val_idx = r.split_indices(data)
            self.assertEqual(train_idx.tolist(), [0])
            self.assertEqual(val_idx.tolist(), [1])

            records["value"]["sha256"] = "invalid"
            (root / "dataset.manifest.json").write_text(json.dumps({
                "complete": True, "arrays": records,
                "source_names": ["old", "new"],
                "group_separation_verified": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                r.load_dataset(root)
            for array in data.values():
                if isinstance(array, np.memmap):
                    array._mmap.close()

    def test_mixed_memmap_dataset_trains_with_mirroring(self):
        import run_experiment as r
        from training.mix_experimental_datasets import mix_datasets

        def write_source(path, base):
            n = 2
            policy = np.zeros((n, 209), dtype=np.float32)
            policy[:, 0] = 1
            np.savez(path,
                     own_pawn=np.array([base, base + 1]),
                     opp_pawn=np.array([80 - base, 79 - base]),
                     walls_h=np.zeros(n, dtype=np.uint64), walls_v=np.zeros(n, dtype=np.uint64),
                     own_dist=np.array([8, 9]), opp_dist=np.array([9, 8]),
                     walls_left_own=np.array([10, 9]), walls_left_opp=np.array([9, 10]),
                     policy=policy, value=np.array([-0.5, 0.5], dtype=np.float32),
                     weight=np.ones(n, dtype=np.float32), is_val=np.array([False, True]),
                     group_id=np.array([f"{base}-train", f"{base}-val"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path, new_path = root / "old.npz", root / "new.npz"
            write_source(old_path, 4)
            write_source(new_path, 14)
            mixed_path = root / "mixed"
            mix_datasets([
                {"name": "historical", "path": str(old_path), "fraction": 0.25,
                 "weight_cap": None},
                {"name": "new", "path": str(new_path), "fraction": 0.75,
                 "weight_cap": None},
            ], mixed_path, chunk_size=2)
            config = dict(r.CONFIG, data=str(mixed_path), out_dir=str(root / "run"),
                          architecture="base", hidden=128, from_scratch=True, qat=False,
                          mirror_h=True, epochs=1, batch_size=2, patience=0,
                          checkpoint_every=1, warmup_epochs=0, device="cpu", cpu_threads=1,
                          resume=True, build=False)
            with mock.patch.object(r, "export", return_value={"architecture": "base"}):
                report = r.train(config)

            self.assertEqual(report["training_status"], "complete")
            self.assertEqual(report["epochs"], 1)
            validation = report["history"][0]["val"]
            self.assertEqual(set(validation["by_source"]), {"historical", "new"})
            self.assertEqual(validation["by_source_target_basis"], "blended_global_targets")


if __name__ == "__main__":
    unittest.main()
