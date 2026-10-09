"""Large-width warm starts retain the production function and export cleanly."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from student_model import Student, encode_features, export


PRODUCTION_FLOAT = ROOT / "data" / "nnue" / "nnue_weights.bin"
NATIVE_PROBE = ROOT / "tools" / "teacher" / "student_probe.cpp"


@pytest.fixture(autouse=True)
def _use_single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _six_bucket_states():
    totals = np.array([0, 1, 3, 6, 10, 15], dtype=np.int64)
    own_walls = np.array([0, 1, 2, 3, 5, 8], dtype=np.int64)
    return {
        "own_pawn": np.array([4, 13, 22, 31, 40, 49], dtype=np.int64),
        "opp_pawn": np.array([76, 67, 58, 49, 40, 31], dtype=np.int64),
        "walls_h": np.array([0, 1 << 4, 1 << 13, 1 << 22, 1 << 31, 1 << 40], dtype=np.uint64),
        "walls_v": np.array([0, 0, 1 << 3, 1 << 12, 1 << 21, 1 << 30], dtype=np.uint64),
        "own_dist": np.array([8, 7, 6, 5, 4, 3], dtype=np.int64),
        "opp_dist": np.array([8, 8, 7, 7, 6, 5], dtype=np.int64),
        "walls_left_own": own_walls,
        "walls_left_opp": totals - own_walls,
    }


def _native_rows(binary, weights):
    proc = subprocess.run([str(binary), str(weights)], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def _compile_from_manifest(tmp_path, manifest):
    compiler = shutil.which("g++") or "C:/mingw64/bin/g++.exe"
    if not Path(compiler).is_file():
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path / f"student-{manifest['hidden']}.exe"
    defines = ["-D" + flag.removeprefix("-D") for flag in manifest["cpp_flags"]]
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I" + str(ROOT / "src"), *defines,
         str(NATIVE_PROBE), "-o", str(binary)],
        check=True,
    )
    return binary


@pytest.mark.parametrize("hidden", [512, 768, 1024])
def test_production_bucketed_warm_start_contact_width_export_and_native_parity(tmp_path, hidden):
    if not PRODUCTION_FLOAT.exists():
        pytest.skip("production float weights unavailable")

    manifest_path = (
        ROOT / "results" / "experiments" / "aux_policy_stage_a" /
        "Soup_Tri_Equal" / "student.architecture.json"
    )
    if not manifest_path.exists():
        manifest_path = (
            ROOT / "results" / "experiments" / "production-central-finetune-20261002" /
            "production_bucketed512" / "train" / "student.architecture.json"
        )
    if not manifest_path.exists():
        manifest_path = (
            ROOT / "results" / "experiments" / "multipath_unified_champion" /
            "student.architecture.json"
        )
    assert hashlib.sha256(PRODUCTION_FLOAT.read_bytes()).hexdigest() == json.loads(
        manifest_path.read_text(encoding="utf-8")
    )["float_sha256"]

    data = _six_bucket_states()
    indices = np.arange(6)
    old_x = torch.from_numpy(encode_features(data, indices, "multipath_phase_bucketed"))
    new_x = torch.from_numpy(encode_features(data, indices, "multipath_phase_contact_bucketed"))
    assert old_x.shape == (6, 504)
    assert new_x.shape == (6, 858)

    for qat in (False, True):
        baseline = Student("multipath_phase_bucketed", 512, qat=qat)
        baseline.load_float(PRODUCTION_FLOAT)
        candidate = Student("multipath_phase_contact_bucketed", hidden, qat=qat)
        candidate.warm_start(baseline)
        with torch.no_grad():
            old_value, old_policy = baseline(old_x)
            new_value, new_policy = candidate(new_x)
        torch.testing.assert_close(new_value, old_value, rtol=0, atol=1e-5)
        torch.testing.assert_close(new_policy, old_policy, rtol=0, atol=1e-5)
        assert torch.count_nonzero(candidate.fc1.weight[:512, 504:]) == 0
        if hidden > 512:
            assert torch.count_nonzero(candidate.fc1.weight[512:]) > 0
            assert torch.count_nonzero(candidate.policy.weight[:, 512:]) == 0
            assert all(torch.count_nonzero(head.weight[:, 512:]) == 0 for head in candidate.value1_heads)

        if not qat:
            # Zero outgoing columns preserve the old function, while active new
            # units and nonzero gradients on those columns let them learn.
            candidate.zero_grad(set_to_none=True)
            policy_coeff = torch.linspace(-1.0, 1.0, 209).expand(6, -1)
            value, policy = candidate(new_x)
            ((policy * policy_coeff).sum() + value.square().sum()).backward()
            assert candidate.fc1.weight.grad[:, 504:].abs().sum() > 0
            if hidden > 512:
                assert candidate.policy.weight.grad[:, 512:].abs().sum() > 0

    output = tmp_path / f"contact-{hidden}.bin"
    manifest = export(candidate, output)
    assert manifest["architecture"] == "multipath_phase_contact_bucketed"
    assert manifest["hidden"] == hidden
    assert f"-DZQ_NNUE_HIDDEN={hidden}" in manifest["cpp_flags"]
    assert ("-DZQ_NNUE_EXPERIMENTAL_WIDTHS=1" in manifest["cpp_flags"]) == (hidden > 512)
    assert output.with_name(f"contact-{hidden}_int8.bin").exists()

    loaded = Student("multipath_phase_contact_bucketed", hidden, qat=True)
    loaded.load_float(output)
    for key, value in candidate.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    # Exercise nonzero contact and expanded-neuron connections through the
    # actual deployed C++ evaluator as well as the serialized layout.
    with torch.no_grad():
        candidate.fc1.weight[:512, 504:].normal_(0.0, 0.01)
        if hidden > 512:
            candidate.policy.weight[:, 512:].normal_(0.0, 0.01)
            for head in candidate.value1_heads:
                head.weight[:, 512:].normal_(0.0, 0.01)
    export(candidate, output)
    exported_copy = Student("multipath_phase_contact_bucketed", hidden, qat=True)
    exported_copy.load_float(output)
    for key, value in candidate.state_dict().items():
        torch.testing.assert_close(value, exported_copy.state_dict()[key], rtol=0, atol=0)
    binary = _compile_from_manifest(tmp_path, manifest)
    rows = _native_rows(binary, output.with_name(f"contact-{hidden}_int8.bin"))
    native_data = {
        key: np.asarray([row[key] for row in rows],
                        dtype=np.uint64 if key.startswith("walls_") else np.int64)
        for key in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                    "walls_left_own", "walls_left_opp")
    }
    with torch.no_grad():
        native_x = torch.from_numpy(encode_features(native_data, np.arange(len(rows)),
                                                    "multipath_phase_contact_bucketed"))
        value, policy = exported_copy(native_x)
    np.testing.assert_allclose(value.numpy(), [row["value"] for row in rows], atol=0.002, rtol=0)
    expected_policy = np.asarray([row["policy"] for row in rows])
    valid = np.ones_like(expected_policy, dtype=bool)
    valid[native_data["walls_left_own"] == 0, 81:] = False
    np.testing.assert_allclose(policy.numpy()[valid], expected_policy[valid], atol=0.002, rtol=0)
