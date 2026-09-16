"""The teacher must export labels when production bypasses the MCTS tree."""
import json
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_empty_handed_teacher_returns_solver_target(tmp_path):
    compiler = shutil.which("g++") or "C:/mingw64/bin/g++.exe"
    if not Path(compiler).is_file():
        pytest.skip("g++ is required")
    exe = tmp_path / "teacher.exe"
    subprocess.run([compiler, "-O2", "-std=c++17", "-I"+str(ROOT/"src"),
                    str(ROOT/"tools/teacher/zq_deep_relabel.cpp"), "-o", str(exe)], check=True)
    history = "d1v c4h h8h a1h e2 e8 e3 e7 e4 e6 d3h d7h f3h f7h b3h b7h h3h d6 e5 c6 a4h d6 e4h e6 d5 f6 c5 g6 b5 g5 b6 a6v b5v a5h g4h h5 b7 i5 c7 i4 d7 h4 e7 g4 f7 g6v f6 f4 g6 g5h f6 e5h"
    output = subprocess.check_output([str(exe), "--nnue", str(ROOT/"data/nnue/nnue_weights_int8.bin"),
        "--nodes", "32"], input="endgame\t"+history+"\n", text=True)
    row = json.loads(output)
    assert row["source"] == "empty_handed_solver"
    assert row["side_to_move"] == 0
    assert row["root_value_prob"] in (0, .5, 1)
    assert row["edges"] == [[row["best_action"], 1, row["root_value_prob"]]]
    assert 0 <= row["best_action"] < 81


def test_teacher_accepts_v3_canonical_state_snapshot(tmp_path):
    compiler = shutil.which("g++") or "C:/mingw64/bin/g++.exe"
    if not Path(compiler).is_file():
        pytest.skip("g++ is required")
    exe = tmp_path / "teacher.exe"
    subprocess.run([compiler, "-O2", "-std=c++17", "-I"+str(ROOT/"src"),
                    str(ROOT/"tools/teacher/zq_deep_relabel.cpp"), "-o", str(exe)], check=True)
    output = subprocess.check_output(
        [str(exe), "--nnue", str(ROOT/"data/nnue/nnue_weights_int8.bin"), "--nodes", "32"],
        input="snapshot\t@state 4 76 0 0 10 10\n", text=True,
    )
    row = json.loads(output)
    assert row["id"] == "snapshot"
    assert row["side_to_move"] == 0
    assert 0 <= row["best_action"] < 209
    assert row["edges"]
