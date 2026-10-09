from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from student_model import Student, export
from quantize_nnue import quantize
from parity_check import forward as py_forward, forward_quant as py_forward_quant


def test_float_warm_start_identity():
    # 1. Create a trained or random 1-layer single-bucket student
    single = Student("multipath_phase", hidden=512, qat=False)
    # 2. Create a 2-layer 6-bucket student and warm start it
    bucketed = Student("multipath_phase_bucketed", hidden=512, qat=False)
    bucketed.warm_start(single)

    # 3. Test across various feature vectors with different wall counts (different buckets)
    torch.manual_seed(42)
    x = torch.randn(20, 504)
    # Set realistic wall counts
    for i in range(20):
        w_own = (i * 3) % 11
        w_opp = (i * 2) % 11
        x[i, 332:354] = 0.0
        x[i, 332 + w_own] = 1.0
        x[i, 343 + w_opp] = 1.0

    v_single, p_single = single(x)
    v_bucketed, p_bucketed = bucketed(x)

    # In float, layer 2 is exact identity: clipped_relu(I * h1 + 0) == h1
    # because h1 is already in [0, 1]
    assert torch.allclose(v_single, v_bucketed, atol=1e-5), f"Max diff: {(v_single - v_bucketed).abs().max()}"
    assert torch.allclose(p_single, p_bucketed, atol=1e-5)


def test_quantized_warm_start_parity():
    # Verify that quantizing the warm-started 6-bucket model matches the single-head model closely (< 0.005 max error)
    single = Student("multipath_phase", hidden=512, qat=True)
    bucketed = Student("multipath_phase_bucketed", hidden=512, qat=True)
    bucketed.warm_start(single)

    q_single = quantize(single.arrays())
    q_bucketed = quantize(bucketed.arrays())

    # Pick some active features
    np.random.seed(42)
    for total_walls in [0, 2, 6, 10, 15, 20]:
        active_feats = np.random.choice(504, size=35, replace=False).tolist()
        val_single, pol_single = py_forward_quant(q_single, active_feats, total_walls=total_walls)
        val_bucketed, pol_bucketed = py_forward_quant(q_bucketed, active_feats, total_walls=total_walls)

        # Policy should be bit-exact identical
        np.testing.assert_allclose(pol_single, pol_bucketed, atol=1e-5)
        # Value diff due to 6-bit intermediate quantization should be tiny (< 0.005 in WL units)
        diff = abs(val_single - val_bucketed)
        assert diff < 0.005, f"Value diff at total_walls={total_walls} was {diff}"


def test_production_warm_start_identity():
    # Test loading actual production checkpoint if it exists
    prod_bin = ROOT / "results" / "experiments" / "multipath-phase512-searchboost-100ep" / "student.bin"
    if not prod_bin.exists():
        prod_bin = ROOT / "data" / "nnue" / "nnue_weights.bin"
    if not prod_bin.exists():
        pytest.skip("Production weights not found")

    single = Student("multipath_phase", hidden=512, qat=False)
    single.load_float(prod_bin)

    bucketed = Student("multipath_phase_bucketed", hidden=512, qat=False)
    bucketed.warm_start(single)

    torch.manual_seed(123)
    x = torch.randn(10, 504)
    for i in range(10):
        x[i, 332:354] = 0.0
        x[i, 332 + (i % 11)] = 1.0
        x[i, 343 + ((i * 2) % 11)] = 1.0

    v_single, p_single = single(x)
    v_bucketed, p_bucketed = bucketed(x)
    assert torch.allclose(v_single, v_bucketed, atol=1e-5)
    assert torch.allclose(p_single, p_bucketed, atol=1e-5)
