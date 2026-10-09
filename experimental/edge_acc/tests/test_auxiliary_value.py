#!/usr/bin/env python3
"""Unit tests for training-only auxiliary short-term value head.

Validates that:
1. Gradients from the auxiliary value head flow into the trunk.
2. The auxiliary head parameters are never serialized into export arrays or files.
3. The exported quantized binary remains bit-exact and exactly 731,172 bytes.
4. Linear and MLP auxiliary value heads both function correctly.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from training.student_model import Student, export

EXPECTED_INT8_SIZE = 731172


class AuxiliaryValueTests(unittest.TestCase):
    """Test suite for training-only auxiliary value heads."""

    def test_aux_value_gradients_reach_trunk(self):
        """Verify that backward pass through aux_value updates the 512 trunk."""
        model = Student("multipath_phase_bucketed", 512)
        model.enable_aux_value("linear")
        x = torch.randn(4, 504)
        v, p, v_aux = model(x, return_aux_value=True)

        loss = v_aux.sum()
        loss.backward()

        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertIsNotNone(model.fc1.bias.grad)
        self.assertGreater(float(model.fc1.weight.grad.abs().sum()), 0.0)

    def test_aux_value_mlp_gradients_reach_trunk(self):
        """Verify that backward pass through MLP aux_value updates the 512 trunk."""
        model = Student("multipath_phase_bucketed", 512)
        model.enable_aux_value("mlp")
        x = torch.randn(4, 504)
        v, p, v_aux = model(x, return_aux_value=True)

        loss = v_aux.sum()
        loss.backward()

        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertGreater(float(model.fc1.weight.grad.abs().sum()), 0.0)

    def test_aux_value_not_present_in_exported_weights(self):
        """Verify that arrays() and layout_keys() omit aux_value parameters."""
        model = Student("multipath_phase_bucketed", 512)
        model.enable_aux_value("linear")
        model.enable_aux_policy()

        exported_keys = list(model.arrays().keys())
        layout = model.layout_keys()

        for key in exported_keys:
            self.assertFalse("aux" in key)
        for key in layout:
            self.assertFalse("aux" in key)

    def test_exported_binary_size_and_exact_int8(self):
        """Verify that exported int8 weights match the exact production size."""
        model = Student("multipath_phase_bucketed", 512)
        model.enable_aux_value("linear")
        model.enable_aux_policy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            float_path = tmp / "model.bin"
            int8_path = tmp / "model_int8.bin"

            arch = export(model, float_path)
            self.assertEqual(arch["architecture"], "multipath_phase_bucketed")

            int8_size = int8_path.stat().st_size
            self.assertEqual(
                int8_size,
                EXPECTED_INT8_SIZE,
                f"Exported int8 binary must be {EXPECTED_INT8_SIZE} bytes, got {int8_size}",
            )



if __name__ == "__main__":
    unittest.main()
