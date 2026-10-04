"""Verify model soup blending, normalization, and export correctness."""
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from training.model_soup import create_soup, load_model_weights
from training.student_model import Student, export

ROOT = Path(__file__).resolve().parents[1]


def test_model_soup_convex_combination(tmp_path):
    arch = "multipath_phase_bucketed"
    hidden = 512

    # Create two synthetic models with known weights
    m1 = Student(arch, hidden, qat=True)
    m2 = Student(arch, hidden, qat=True)

    with torch.no_grad():
        m1.fc1.weight.fill_(1.0)
        m1.fc1.bias.fill_(0.5)
        m2.fc1.weight.fill_(3.0)
        m2.fc1.bias.fill_(1.5)

    p1 = tmp_path / "m1.bin"
    p2 = tmp_path / "m2.bin"
    export(m1, p1)
    export(m2, p2)

    # Blend 50/50
    out_dir = tmp_path / "soup_out"
    int8_bin, exe_bin = create_soup(
        model_specs=[(p1, 0.5), (p2, 0.5)],
        out_dir=out_dir,
        architecture=arch,
        hidden=hidden,
        qat=True,
        build=False,
    )

    assert int8_bin.exists()
    assert (out_dir / "student.bin").exists()
    assert (out_dir / "soup_manifest.json").exists()

    # Load resulting soup and verify weights are exact average
    soup = Student(arch, hidden, qat=True)
    soup.load_float(out_dir / "student.bin")

    torch.testing.assert_close(soup.fc1.weight, torch.full_like(soup.fc1.weight, 2.0))
    torch.testing.assert_close(soup.fc1.bias, torch.full_like(soup.fc1.bias, 1.0))
