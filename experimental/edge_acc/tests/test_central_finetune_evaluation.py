from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import run_central_finetune_evaluation as evaluation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    executable = frozen / "zquoridor.exe"
    baseline_nnue = frozen / "production_nnue_int8.bin"
    candidate_nnue = tmp_path / "candidate" / "student_int8.bin"
    candidate_nnue.parent.mkdir()
    book = tmp_path / "center.jsonl"
    executable.write_bytes(b"frozen executable")
    baseline_nnue.write_bytes(b"production weights")
    candidate_nnue.write_bytes(b"candidate weights")
    book.write_text('{"moves": ["e2"]}\n', encoding="utf-8")

    architecture = {
        "schema": "zquoridor.student.v1",
        "architecture": "multipath_phase_bucketed",
        "features": 504,
        "hidden": 512,
        "value_buckets": 6,
        "value_depth": 2,
        "int8_sha256": _sha256(candidate_nnue),
    }
    candidate_arch = candidate_nnue.with_name("student.architecture.json")
    candidate_arch.write_text(json.dumps(architecture), encoding="utf-8")
    manifest = {
        "schema": "zquoridor.frozen_finetune_baseline.v1",
        "architecture": {key: value for key, value in architecture.items()
                          if key != "int8_sha256"} | {
                              "int8_sha256": _sha256(baseline_nnue),
                          },
        "artifacts": {
            "zquoridor.exe": {
                "path": executable.name,
                "size_bytes": executable.stat().st_size,
                "sha256": _sha256(executable),
            },
            "production_nnue_int8.bin": {
                "path": baseline_nnue.name,
                "size_bytes": baseline_nnue.stat().st_size,
                "sha256": _sha256(baseline_nnue),
            },
        },
    }
    (frozen / "frozen_baseline_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = {
        "candidate_name": "candidate",
        "candidate_nnue": str(candidate_nnue),
        "candidate_architecture": str(candidate_arch),
        "frozen_dir": str(frozen),
        "pairs": 200,
        "move_time_ms": 200,
        "workers": 4,
        "seed": 20261002,
        "openings": str(book),
        "output": str(tmp_path / "benchmarks"),
        "claustrophobia_device": "cpu",
        "claustrophobia_max_sims": 4096,
        "bootstrap": 20000,
    }
    return config, executable, baseline_nnue, candidate_nnue


def test_two_mocked_arenas_share_book_clock_seed_and_resume_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, executable, baseline_nnue, candidate_nnue = _fixture(tmp_path)
    main_calls: list[dict] = []
    claustro_calls: list[dict] = []

    def main_arena(options: dict) -> dict:
        main_calls.append(options.copy())
        return {"complete_pairs": 200, "failed_games": 0, "score_pct": 51.0}

    def claustro_arena(options: dict) -> dict:
        claustro_calls.append(options.copy())
        return {"summaries": {"claustrophobia": {
            "complete_pairs": 200, "failed_games": 0, "score_pct": 52.0,
        }}}

    monkeypatch.setattr(evaluation.match_finalists, "run", main_arena)
    monkeypatch.setattr(evaluation.run_benchmark, "run", claustro_arena)

    first = evaluation.run(config)
    second = evaluation.run(config)
    capsys.readouterr()

    assert len(main_calls) == len(claustro_calls) == 2
    for main, claustro in zip(main_calls, claustro_calls):
        assert main["pairs"] == claustro["pairs"] == 200
        assert main["move_time_ms"] == 200
        assert claustro["zq_move_time_ms"] == claustro["claustrophobia_move_time_ms"] == 200
        assert main["seed"] == claustro["seed"] == 20261002
        assert main["openings"] == claustro["openings"] == str(Path(config["openings"]).resolve())
        assert main["engine1_executable"] == main["engine2_executable"] == str(executable.resolve())
        assert main["engine1_nnue"] == str(candidate_nnue.resolve())
        assert main["engine2_nnue"] == str(baseline_nnue.resolve())
        assert claustro["zq_executable"] == str(executable.resolve())
        assert claustro["nnue"] == str(candidate_nnue.resolve())
        assert claustro["resume"] is True
    assert _sha256(candidate_nnue) != _sha256(baseline_nnue)
    assert first["games_per_opponent"] == second["games_per_opponent"] == 400
    assert main_calls[0]["output"] == main_calls[1]["output"]
    assert claustro_calls[0]["output"] == claustro_calls[1]["output"]


def test_rejects_corrupt_frozen_artifact_before_starting_arenas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, executable, _, _ = _fixture(tmp_path)
    executable.write_bytes(b"changed after freezing")
    monkeypatch.setattr(evaluation.match_finalists, "run", lambda _: pytest.fail("arena started"))
    monkeypatch.setattr(evaluation.run_benchmark, "run", lambda _: pytest.fail("arena started"))

    with pytest.raises(ValueError, match="frozen baseline (size|SHA-256) mismatch"):
        evaluation.run(config)


def test_current_production_weights_are_accepted_via_frozen_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, executable, baseline_nnue, _ = _fixture(tmp_path)
    config["candidate_nnue"] = str(baseline_nnue)
    config["candidate_architecture"] = None
    monkeypatch.setattr(
        evaluation.match_finalists, "run",
        lambda _: {"complete_pairs": 200, "failed_games": 0, "score_pct": 50.0},
    )
    monkeypatch.setattr(
        evaluation.run_benchmark, "run",
        lambda _: {"summaries": {"claustrophobia": {
            "complete_pairs": 200, "failed_games": 0, "score_pct": 50.0,
        }}},
    )

    report = evaluation.run(config)
    capsys.readouterr()

    assert report["candidate_nnue_sha256"] == report["frozen_nnue_sha256"]
    assert report["candidate_executable"] == str(executable.resolve())


def test_rejects_incompatible_or_mismatched_candidate_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _, candidate_nnue = _fixture(tmp_path)
    architecture_path = Path(config["candidate_architecture"])
    metadata = json.loads(architecture_path.read_text(encoding="utf-8"))
    metadata["features"] = 858
    architecture_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(evaluation.match_finalists, "run", lambda _: pytest.fail("arena started"))

    with pytest.raises(ValueError, match="504/512/6/2"):
        evaluation.run(config)

    metadata["features"] = 504
    metadata["int8_sha256"] = "0" * 64
    architecture_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        evaluation.run(config)
    assert candidate_nnue.is_file()


@pytest.mark.parametrize("failed_opponent", ["main", "claustrophobia"])
def test_failed_games_never_produce_strength_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_opponent: str,
) -> None:
    config, _, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        evaluation.match_finalists,
        "run",
        lambda _: {"complete_pairs": 199 if failed_opponent == "main" else 200,
                   "failed_games": 1 if failed_opponent == "main" else 0},
    )
    monkeypatch.setattr(
        evaluation.run_benchmark,
        "run",
        lambda _: {"summaries": {"claustrophobia": {
            "complete_pairs": 199 if failed_opponent == "claustrophobia" else 200,
            "failed_games": 1 if failed_opponent == "claustrophobia" else 0,
        }}},
    )

    with pytest.raises(RuntimeError, match="evaluation incomplete"):
        evaluation.run(config)
    capsys.readouterr()
    assert not list((tmp_path / "benchmarks").glob("**/evaluation_summary.json"))
