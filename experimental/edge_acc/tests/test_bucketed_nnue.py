from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from student_model import Student, export, encode_features, FEATURES, ARCH_CONFIGS


def test_architectures_registered():
    assert "multipath_phase_bucketed" in FEATURES
    assert "multipath_phase_deep" in FEATURES
    assert "multipath_phase_contact_bucketed" in FEATURES
    assert ARCH_CONFIGS["multipath_phase_bucketed"]["buckets"] == 6
    assert ARCH_CONFIGS["multipath_phase_bucketed"]["depth"] == 2
    assert ARCH_CONFIGS["multipath_phase_deep"]["buckets"] == 1
    assert ARCH_CONFIGS["multipath_phase_deep"]["depth"] == 2
    assert ARCH_CONFIGS["multipath_phase_contact_bucketed"]["buckets"] == 6
    assert ARCH_CONFIGS["multipath_phase_contact_bucketed"]["depth"] == 2


def test_student_forward_and_qat_bucketed():
    # 6 buckets, depth 2
    model = Student("multipath_phase_bucketed", hidden=128, qat=False)
    assert model.value_buckets == 6
    assert model.value_depth == 2

    # Synthetic input with 504 features
    x = torch.zeros(12, 504)
    # Set one-hot walls left:
    # row 0: total 0 walls -> bucket 0
    # row 1: total 2 walls -> bucket 1
    # row 2: total 4 walls -> bucket 2
    # row 3: total 8 walls -> bucket 3
    # row 4: total 12 walls -> bucket 4
    # row 5: total 18 walls -> bucket 5
    walls_test = [(0, 0), (1, 1), (2, 2), (4, 4), (6, 6), (9, 9), (0, 0), (1, 1), (2, 2), (4, 4), (6, 6), (9, 9)]
    for i, (w_own, w_opp) in enumerate(walls_test):
        x[i, 332 + w_own] = 1.0
        x[i, 343 + w_opp] = 1.0

    v, p = model(x)
    assert v.shape == (12,)
    assert p.shape == (12, 209)
    assert torch.isfinite(v).all()
    assert torch.isfinite(p).all()

    # Test QAT forward pass
    model_qat = Student("multipath_phase_bucketed", hidden=128, qat=True)
    v_q, p_q = model_qat(x)
    assert v_q.shape == (12,)
    assert p_q.shape == (12, 209)
    assert torch.isfinite(v_q).all()
    assert torch.isfinite(p_q).all()


def test_student_forward_and_qat_deep_single():
    model = Student("multipath_phase_deep", hidden=128, qat=False)
    assert model.value_buckets == 1
    assert model.value_depth == 2

    x = torch.zeros(4, 504)
    for i in range(4):
        x[i, 332 + i] = 1.0
        x[i, 343 + i] = 1.0

    v, p = model(x)
    assert v.shape == (4,)
    assert p.shape == (4, 209)


def test_warm_start_preserves_function():
    # Warm start 6-bucket 2-layer model from single-head model
    old = Student("multipath_phase", hidden=128, qat=False)
    model = Student("multipath_phase_bucketed", hidden=128, qat=False)
    model.warm_start(old)

    x = torch.zeros(5, 504)
    for i in range(5):
        x[i, 332 + i] = 1.0
        x[i, 343 + i] = 1.0

    v_old, p_old = old(x)
    v_new, p_new = model(x)
    # Because value2 was initialized to identity, epoch 0 evaluation is identical!
    assert torch.allclose(v_old, v_new, atol=1e-5)
    assert torch.allclose(p_old, p_new, atol=1e-5)


def test_export_and_load_roundtrip():
    model = Student("multipath_phase_bucketed", hidden=128, qat=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_model.bin"
        manifest = export(model, out_path)
        assert manifest["value_buckets"] == 6
        assert manifest["value_depth"] == 2
        assert out_path.exists()
        assert out_path.with_name("test_model_int8.bin").exists()

        # Load float into fresh model
        loaded = Student("multipath_phase_bucketed", hidden=128, qat=True)
        loaded.load_float(out_path)

        x = torch.zeros(4, 504)
        for i in range(4):
            x[i, 332 + i] = 1.0
            x[i, 343 + i] = 1.0

        v1, p1 = model(x)
        v2, p2 = loaded(x)
        assert torch.allclose(v1, v2, atol=1e-5)
        assert torch.allclose(p1, p2, atol=1e-5)


@pytest.mark.parametrize("architecture", ["multipath_phase_bucketed", "multipath_phase_deep"])
def test_qat_matches_int8_reference(architecture):
    from quantize_nnue import quantize
    from parity_check import forward_quant

    torch.manual_seed(7)
    model = Student(architecture, hidden=512, qat=True)
    with torch.no_grad():
        model.fc1.weight.mul_(8)
        model.fc1.bias.uniform_(0, 0.6)
        for heads in (model.value1_heads, model.value2_heads, model.value3_heads):
            for head in heads:
                head.weight.uniform_(-1.5, 1.5)
                head.bias.uniform_(-0.3, 0.3)
    model.clip_weights()
    weights = quantize(model.arrays())

    rng = np.random.default_rng(3)
    rows, walls = [], []
    for _ in range(300):
        own, opp = rng.integers(0, 11, size=2)
        active = set(rng.choice(np.r_[0:332, 354:504], size=30, replace=False).tolist())
        active |= {332 + int(own), 343 + int(opp)}
        rows.append(sorted(active))
        walls.append(int(own + opp))
    x = torch.zeros(len(rows), 504)
    for i, active in enumerate(rows):
        x[i, active] = 1.0
    with torch.no_grad():
        values, _ = model(x)
    for i, active in enumerate(rows):
        reference, _ = forward_quant(weights, active, total_walls=walls[i])
        assert abs(float(values[i]) - reference) < 1e-5


def test_qat_value_heads_receive_gradients():
    model = Student("multipath_phase_bucketed", hidden=128, qat=True)
    x = torch.zeros(12, 504)
    for i in range(12):
        x[i, 332 + i % 11] = 1.0
        x[i, 343 + (i * 3) % 11] = 1.0
        x[i, i] = 1.0
    values, _ = model(x)
    values.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(h.weight.grad.abs().sum() > 0 for h in model.value3_heads)
