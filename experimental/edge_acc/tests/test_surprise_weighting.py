#!/usr/bin/env python3
"""Unit tests for policy surprise weighting calculation.

Validates mathematical properties:
1. Non-negativity of Kullback-Leibler divergence.
2. Exact zero divergence when target equals predicted distribution.
3. Strict monotonicity of surprise multipliers with respect to divergence.
4. Exact mass conservation under normalized multipliers.
5. Strict upper bound clipping at s_max.
"""
from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn.functional as F

from training.compute_surprise_weights import (
    compute_normalized_multipliers,
)
from training.student_model import Student


class DummyModel(torch.nn.Module):
    """Simple model fixture for policy output tests."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, x: torch.Tensor):
        return torch.zeros((len(x), 1)), self.logits[:len(x)]


class SurpriseWeightingTests(unittest.TestCase):
    """Verify numerical and structural invariants of policy surprise weighting."""

    def test_kl_divergence_zero_when_identical(self):
        """Verify that KL divergence is zero when search policy matches prior."""
        batch_size = 8
        num_actions = 209
        # Random logits
        torch.manual_seed(42)
        logits = torch.randn(batch_size, num_actions)
        p = F.softmax(logits, dim=1)

        log_prior = F.log_softmax(logits, dim=1)
        p_safe = p.clamp_min(1e-12)
        kl = (p * (p_safe.log() - log_prior)).sum(dim=1)

        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl), atol=1e-5))

    def test_kl_divergence_nonnegative(self):
        """Verify that KL divergence is strictly non-negative for differing distributions."""
        torch.manual_seed(42)
        batch_size = 16
        num_actions = 209

        logits = torch.randn(batch_size, num_actions)
        target_logits = torch.randn(batch_size, num_actions)
        p = F.softmax(target_logits, dim=1)

        log_prior = F.log_softmax(logits, dim=1)
        p_safe = p.clamp_min(1e-12)
        kl = (p * (p_safe.log() - log_prior)).sum(dim=1)

        self.assertTrue(torch.all(kl >= -1e-6))
        self.assertTrue(torch.any(kl > 0.01))

    def test_multipliers_preserve_mass(self):
        """Verify that normalized multipliers preserve the exact total base mass."""
        rng = np.random.default_rng(20261004)
        n = 10000
        kl = rng.uniform(0.0, 5.0, size=n).astype(np.float32)
        base_weights = rng.uniform(0.5, 3.0, size=n).astype(np.float32)

        multipliers = compute_normalized_multipliers(
            kl, base_weights, alpha=0.5, s_max=4.0, normalize_mass=True
        )

        initial_mass = float(np.sum(base_weights, dtype=np.float64))
        boosted_mass = float(np.sum(base_weights * multipliers, dtype=np.float64))

        np.testing.assert_allclose(initial_mass, boosted_mass, rtol=1e-5)

    def test_higher_surprise_yields_higher_multiplier(self):
        """Verify that higher KL divergence produces strictly higher multiplier."""
        kl = np.array([0.1, 0.5, 1.0, 2.5, 3.5], dtype=np.float32)
        base_weights = np.ones_like(kl)

        multipliers = compute_normalized_multipliers(
            kl, base_weights, alpha=0.5, s_max=4.0, normalize_mass=True
        )

        for i in range(len(multipliers) - 1):
            self.assertGreater(multipliers[i + 1], multipliers[i])

    def test_multiplier_bounded_by_s_max(self):
        """Verify that surprise above s_max is clipped and does not increase multiplier."""
        s_max = 3.0
        kl = np.array([2.0, 3.0, 4.0, 10.0, 100.0], dtype=np.float32)
        base_weights = np.ones_like(kl)

        multipliers = compute_normalized_multipliers(
            kl, base_weights, alpha=0.5, s_max=s_max, normalize_mass=False
        )

        # Values at 3.0, 4.0, 10.0, 100.0 must all have the same capped multiplier
        expected_cap = 1.0 + 0.5 * s_max
        self.assertAlmostEqual(multipliers[1], expected_cap, places=5)
        self.assertAlmostEqual(multipliers[2], expected_cap, places=5)
        self.assertAlmostEqual(multipliers[3], expected_cap, places=5)
        self.assertAlmostEqual(multipliers[4], expected_cap, places=5)

    def test_create_dataset_variant_loads_cleanly(self):
        """Verify that create_dataset_variant produces a valid manifest loaded by run_experiment."""
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from training.compute_surprise_weights import create_dataset_variant
        from training.run_experiment import load_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = Path(tmp_dir) / "src"
            tgt_dir = Path(tmp_dir) / "tgt"
            src_fields = src_dir / "fields"
            src_fields.mkdir(parents=True)

            n = 50
            # Create minimal valid fields
            fields = {
                "own_pawn": np.zeros(n, dtype=np.uint8),
                "opp_pawn": np.ones(n, dtype=np.uint8),
                "walls_h": np.zeros(n, dtype=np.uint64),
                "walls_v": np.zeros(n, dtype=np.uint64),
                "own_dist": np.full(n, 5, dtype=np.uint8),
                "opp_dist": np.full(n, 5, dtype=np.uint8),
                "walls_left_own": np.full(n, 10, dtype=np.int8),
                "walls_left_opp": np.full(n, 10, dtype=np.int8),
                "is_val": np.array([i % 5 == 0 for i in range(n)], dtype=bool),
                "policy": np.full((n, 209), 1.0 / 209.0, dtype=np.float32),
                "value": np.zeros(n, dtype=np.float32),
                "weight": np.ones(n, dtype=np.float32),
                "source_mass": np.ones((n, 1), dtype=np.float32),
            }
            manifest_arrays = {}
            for name, arr in fields.items():
                p = src_fields / f"{name}.npy"
                np.save(p, arr)
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                manifest_arrays[name] = {
                    "filename": f"fields\\{name}.npy",
                    "sha256": h,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }

            manifest = {
                "complete": True,
                "group_separation_verified": True,
                "arrays": manifest_arrays,
                "source_names": ["src0"],
            }
            (src_dir / "dataset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            multipliers = np.full(n, 1.5, dtype=np.float32)
            new_weights = fields["weight"] * multipliers

            create_dataset_variant(src_dir, tgt_dir, new_weights, multipliers)

            # Validate that load_dataset can load and verify the target dataset
            loaded = load_dataset(tgt_dir)
            self.assertEqual(len(loaded["weight"]), n)
            np.testing.assert_allclose(loaded["weight"], 1.5, rtol=1e-5)
            for val in loaded.values():
                if isinstance(val, np.memmap):
                    val._mmap.close()


if __name__ == "__main__":
    unittest.main()

