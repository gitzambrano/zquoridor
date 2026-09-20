"""Cross-language checks with the real deployed integer evaluator."""
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


def compile_probe(tmp_path, architecture):
    compiler = shutil.which("g++") or "C:/mingw64/bin/g++.exe"
    if not Path(compiler).is_file():
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path / (architecture + ".exe")
    race = architecture in ("race", "multipath", "multipath_phase")
    multipath = architecture in ("multipath", "multipath_phase")
    phase = architecture == "multipath_phase"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(ROOT / "src"),
                    f"-DZQ_NNUE_RACE_FEATURES={int(race)}",
                    f"-DZQ_NNUE_MULTIPATH_FEATURES={int(multipath)}",
                    f"-DZQ_NNUE_PHASE_FEATURES={int(phase)}",
                    "-DZQ_NNUE_HIDDEN=512" if architecture == "multipath_phase" else "-DZQ_NNUE_HIDDEN=256",
                    str(ROOT / "tools/teacher/student_probe.cpp"), "-o", str(binary)], check=True)
    return binary


def probe(binary, weights):
    proc = subprocess.run([str(binary), str(weights)], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def test_zero_expansion_integer_exact_and_python_qat_parity(tmp_path):
    baseline = Student("base", 256, qat=True)
    baseline.load_float(ROOT / "checkpoints/gen9-teacher-head/nnue_weights.bin")
    candidate = Student("race", 256, qat=True)
    candidate.warm_start(baseline)
    export(baseline, tmp_path / "base.bin")
    export(candidate, tmp_path / "race.bin")
    base_rows = probe(compile_probe(tmp_path, "base"), tmp_path / "base_int8.bin")
    race_binary = compile_probe(tmp_path, "race")
    race_rows = probe(race_binary, tmp_path / "race_int8.bin")
    assert base_rows == race_rows
    # Nonzero new rows verify Python and C++ agree about the feature indices.
    with torch.no_grad():
        torch.manual_seed(19)
        candidate.fc1.weight[:, 354:].normal_(0, 0.01)
    export(candidate, tmp_path / "race.bin")
    rows = probe(race_binary, tmp_path / "race_int8.bin")
    data = {key: np.asarray([r[key] for r in rows], dtype=np.uint64 if key.startswith("walls_") else np.int64)
            for key in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                        "walls_left_own", "walls_left_opp")}
    with torch.no_grad():
        value, policy = candidate(torch.from_numpy(encode_features(data, np.arange(len(rows)), "race")))
    np.testing.assert_allclose(value.numpy(), [r["value"] for r in rows], atol=0.002, rtol=0)
    # The native evaluator deliberately omits unreachable wall logits at zero walls.
    expected = np.asarray([r["policy"] for r in rows])
    valid = np.ones_like(expected, dtype=bool)
    valid[data["walls_left_own"] == 0, 81:] = False
    np.testing.assert_allclose(policy.numpy()[valid], expected[valid], atol=0.002, rtol=0)


def test_multipath_phase_native_layout_matches_python(tmp_path):
    torch.manual_seed(29)
    baseline = Student("multipath", 512, qat=True)
    candidate = Student("multipath_phase", 512, qat=True)
    candidate.warm_start(baseline)
    export(candidate, tmp_path / "multipath_phase.bin")

    binary = compile_probe(tmp_path, "multipath_phase")
    rows = probe(binary, tmp_path / "multipath_phase_int8.bin")
    data = {key: np.asarray([r[key] for r in rows], dtype=np.uint64 if key.startswith("walls_") else np.int64)
            for key in ("own_pawn", "opp_pawn", "walls_h", "walls_v", "own_dist", "opp_dist",
                        "walls_left_own", "walls_left_opp")}
    with torch.no_grad():
        value, policy = candidate(torch.from_numpy(
            encode_features(data, np.arange(len(rows)), "multipath_phase")))
    np.testing.assert_allclose(value.numpy(), [r["value"] for r in rows], atol=0.002, rtol=0)
    expected = np.asarray([r["policy"] for r in rows])
    valid = np.ones_like(expected, dtype=bool)
    valid[data["walls_left_own"] == 0, 81:] = False
    np.testing.assert_allclose(policy.numpy()[valid], expected[valid], atol=0.002, rtol=0)
