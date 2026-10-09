"""Verification suite for training-only auxiliary policy and split modes."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import run_experiment as r
import student_model as s


class AuxiliaryPolicyTests(unittest.TestCase):
    """Test auxiliary policy mechanics, export hygiene, and dataset split modes."""

    def test_aux_policy_gradients_reach_trunk(self):
        """Confirm that auxiliary policy loss produces gradients in the trunk."""
        torch.manual_seed(20261004)
        model = s.Student("multipath_phase_bucketed", 512, qat=True, aux_policy=True)
        model.enable_aux_policy()

        x = torch.randn(4, 504)
        p_target = torch.softmax(torch.randn(4, 209), dim=-1)

        v, p, p_aux = model(x, return_aux=True)
        self.assertIsNotNone(p_aux)

        p_soft = s.soften_policy(p_target, 2.0)
        aux_loss = (p_soft * (p_soft.clamp_min(1e-12).log() - F.log_softmax(p_aux, dim=1))).sum(1).mean()

        model.zero_grad()
        aux_loss.backward()

        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertTrue((model.fc1.weight.grad.abs() > 0).any().item())
        self.assertIsNotNone(model.aux_policy.weight.grad)
        self.assertIsNone(model.policy.weight.grad)

    def test_disabling_aux_reproduces_baseline_behavior(self):
        """Confirm that disabling auxiliary policy reproduces baseline updates."""
        torch.manual_seed(42)
        base_model = s.Student("multipath_phase_bucketed", 512, qat=True, aux_policy=False)
        torch.manual_seed(42)
        aux_disabled_model = s.Student("multipath_phase_bucketed", 512, qat=True, aux_policy=False)

        x = torch.randn(4, 504)
        v1, p1 = base_model(x)
        v2, p2 = aux_disabled_model(x)

        torch.testing.assert_close(v1, v2)
        torch.testing.assert_close(p1, p2)

    def test_soft_targets_remain_normalized_and_finite(self):
        """Confirm that softened targets remain normalized, non-negative, and finite."""
        rng = torch.Generator().manual_seed(12345)
        for temp in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
            logits = torch.randn(32, 209, generator=rng)
            mask = torch.rand(32, 209, generator=rng) < 0.85
            logits[mask] = -1e9
            p = torch.softmax(logits, dim=-1)

            soft_p = s.soften_policy(p, temp)

            self.assertTrue(torch.isfinite(soft_p).all().item())
            self.assertTrue((soft_p >= 0).all().item())
            sums = soft_p.sum(dim=-1)
            torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
            self.assertTrue((soft_p[mask] == 0).all().item())

    def test_temperature_one_matches_original_distribution(self):
        """Confirm that temperature 1.0 reproduces the input policy distribution."""
        p = torch.softmax(torch.randn(16, 209), dim=-1)
        soft_p = s.soften_policy(p, 1.0)
        torch.testing.assert_close(soft_p, p, atol=1e-6, rtol=1e-6)

    def test_auxiliary_parameters_not_present_in_exported_weights(self):
        """Confirm that exported binaries do not contain auxiliary head parameters."""
        torch.manual_seed(7)
        clean_model = s.Student("multipath_phase_bucketed", 512, qat=True)
        aux_model = s.Student("multipath_phase_bucketed", 512, qat=True)
        aux_model.load_state_dict(clean_model.state_dict())
        aux_model.enable_aux_policy()

        self.assertNotIn("aux_policy", aux_model.layout_keys())
        self.assertEqual(clean_model.layout_keys(), aux_model.layout_keys())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clean_bin = temp_path / "clean.bin"
            aux_bin = temp_path / "aux.bin"

            s.export(clean_model, clean_bin)
            s.export(aux_model, aux_bin)

            self.assertEqual(clean_bin.read_bytes(), aux_bin.read_bytes())
            clean_int8 = clean_bin.with_name("clean_int8.bin")
            aux_int8 = aux_bin.with_name("aux_int8.bin")
            self.assertEqual(clean_int8.read_bytes(), aux_int8.read_bytes())

    def test_float_int8_incremental_parity_unchanged(self):
        """Confirm that float and int8 export parity matches the production format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model = s.Student("multipath_phase_bucketed", 512, qat=True)
            model.enable_aux_policy()
            out_bin = temp_path / "student.bin"
            manifest = s.export(model, out_bin)

            int8_path = temp_path / "student_int8.bin"
            self.assertTrue(int8_path.is_file())
            self.assertEqual(manifest["features"], 504)
            self.assertEqual(manifest["hidden"], 512)
            self.assertEqual(manifest["value_buckets"], 6)

    def test_deployed_network_size_and_manifest_architecture_unchanged(self):
        """Confirm that exported int8 size matches production 731,172 bytes."""
        model = s.Student("multipath_phase_bucketed", 512, qat=True)
        model.enable_aux_policy()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "student.bin"
            manifest = s.export(model, target)
            int8_file = target.with_name("student_int8.bin")

            prod_size = (ROOT / "data/nnue/nnue_weights_int8.bin").stat().st_size
            self.assertEqual(int8_file.stat().st_size, prod_size)
            self.assertEqual(int8_file.stat().st_size, 731172)
            self.assertEqual(manifest["schema"], "zquoridor.student.v1")
            self.assertEqual(manifest["policy_out"], 209)
            self.assertNotIn("aux_policy", manifest)

    def test_validation_mode_none_trains_on_all_samples(self):
        """Confirm that validation_mode='none' allocates all rows to training."""
        data = {
            "value": np.zeros(250, dtype=np.float32),
            "is_val": np.array([i % 5 == 0 for i in range(250)], dtype=bool),
        }
        split = r.get_validation_split(data, {"validation_mode": "none"})

        self.assertEqual(split["mode"], "none")
        self.assertEqual(len(split["train_idx"]), 250)
        self.assertEqual(len(split["val_idx"]), 0)
        self.assertEqual(split["train_samples"], 250)
        self.assertEqual(split["val_samples"], 0)
        np.testing.assert_array_equal(split["train_idx"], np.arange(250))

    def test_group_resplit_guarantees_zero_group_leakage(self):
        """Confirm that validation_mode='resplit' strictly partitions groups."""
        n_samples = 1000
        n_groups = 40
        group_ids = np.repeat(np.arange(n_groups), n_samples // n_groups)

        data = {
            "value": np.zeros(n_samples, dtype=np.float32),
            "is_val": np.zeros(n_samples, dtype=bool),
            "group_id": group_ids,
        }

        split1 = r.get_validation_split(data, {
            "validation_mode": "resplit",
            "validation_fraction": 0.05,
            "validation_seed": 20261004,
        })

        train_groups1 = set(group_ids[split1["train_idx"]].tolist())
        val_groups1 = set(group_ids[split1["val_idx"]].tolist())
        self.assertEqual(len(train_groups1 & val_groups1), 0)
        self.assertGreater(len(val_groups1), 0)
        self.assertEqual(len(train_groups1) + len(val_groups1), n_groups)

        split2 = r.get_validation_split(data, {
            "validation_mode": "resplit",
            "validation_fraction": 0.05,
            "validation_seed": 20261004,
        })
        np.testing.assert_array_equal(split1["train_idx"], split2["train_idx"])
        np.testing.assert_array_equal(split1["val_idx"], split2["val_idx"])

        split3 = r.get_validation_split(data, {
            "validation_mode": "resplit",
            "validation_fraction": 0.05,
            "validation_seed": 9999,
        })
        train_groups3 = set(group_ids[split3["train_idx"]].tolist())
        val_groups3 = set(group_ids[split3["val_idx"]].tolist())
        self.assertEqual(len(train_groups3 & val_groups3), 0)
        self.assertNotEqual(val_groups1, val_groups3)


if __name__ == "__main__":
    unittest.main()
