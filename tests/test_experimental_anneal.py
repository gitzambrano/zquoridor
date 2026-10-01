"""Check the shared recipe and preparation gates for experimental training."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import run_experimental_anneal as runner
from training.student_model import Student, export


def test_dry_run_reports_missing_inputs_without_writes(tmp_path):
    config = copy.deepcopy(runner.CONFIG)
    config.update(out_dir=str(tmp_path / "outputs"))
    report = runner.run(config)
    assert report["status"] == "dry_run"
    assert report["reserve_unassigned"] == 4_000_000
    assert report["mixture"]["historical_mass"] == 0.25
    assert report["mixture"]["new_mass"] == 0.75
    assert report["mixture"]["use_all_eligible_records"]
    assert all(entry["fraction"] is None for entry in report["datasets"][1:])
    assert len(report["networks"]) == 4
    assert not (tmp_path / "outputs").exists()


def test_training_recipe_preserves_shared_low_rate_schedule(tmp_path):
    config = copy.deepcopy(runner.CONFIG)
    entry = dict(runner.NETWORKS[3], initial=str(tmp_path / "initial.bin"))
    recipe = runner.training_config(config, entry, tmp_path / "mixed")
    assert recipe["epochs"] == 120
    assert recipe["lr"] == 1e-5
    assert recipe["min_lr"] == 1e-7
    assert recipe["trunk_lr_scale"] == 0.05
    assert recipe["patience"] == recipe["warmup_epochs"] == 0
    assert recipe["checkpoint_every"] == 10
    assert recipe["policy_weight"] == recipe["value_weight"] == 1
    assert recipe["qat"] and recipe["mirror_h"]
    assert recipe["init_hidden"] == recipe["hidden"] == 1024


def test_new_replay_requires_literal_search_result_blend(tmp_path):
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"frozen dataset")
    entry = dict(name="colab3", path=str(dataset), historical=False)
    manifest = dict(complete=True, dataset_sha256=runner.sha(dataset),
                    config=dict(mode="stored_search", stored_outcome_weight=0.5, stored_gamma=1))
    file = tmp_path / "replay_manifest.json"
    file.write_text(json.dumps(manifest))
    runner.validate_replay_source(entry)
    manifest["config"]["stored_outcome_weight"] = 0
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="50/50"):
        runner.validate_replay_source(entry)


def test_initialization_exports_all_candidates_and_checks_resume_sha(tmp_path):
    torch.set_num_threads(2)
    champion = tmp_path / "champion.bin"
    export(Student("multipath_phase_bucketed", 512, qat=True), champion)
    config = copy.deepcopy(runner.CONFIG)
    config.update(champion=str(champion), production_weights=str(champion),
                  out_dir=str(tmp_path / "matrix"), dry_run=False, initialize_only=True)
    result = runner.run(config)
    assert result["status"] == "initialized_training_not_started"
    for entry in result["networks"]:
        manifest = json.loads(Path(entry["architecture_manifest"]).read_text())
        assert manifest["hidden"] == entry["hidden"]
        assert manifest["value_buckets"] == 6 and manifest["value_depth"] == 2
        assert Path(entry["initial"]).with_name("initial.pt").exists()
        assert not Path(entry["initial"]).parent.parent.joinpath("train").exists()
    assert runner.run(config)["networks"] == result["networks"]
    initial = Path(result["networks"][1]["initial"])
    initial.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.run(config)


def test_missing_colab_is_not_silently_redistributed(tmp_path):
    config = copy.deepcopy(runner.CONFIG)
    config["data_sources"][0]["fraction"] = 0.40
    with pytest.raises(ValueError, match="sum to one"):
        runner.validate_config(config)
    with pytest.raises(FileNotFoundError, match="not ready"):
        runner.validate_replay_source(dict(name="colab4", path=str(tmp_path / "absent"), historical=False))


def test_retention_uses_unchanged_historical_targets(tmp_path):
    import numpy as np
    import shutil
    torch.set_num_threads(2)
    policy = np.zeros((2, 209), dtype=np.float32)
    policy[:, 0] = 1
    dataset = tmp_path / "historical.npz"
    np.savez(dataset, own_pawn=np.array([4, 5]), opp_pawn=np.array([76, 76]),
             walls_h=np.zeros(2, dtype=np.uint64), walls_v=np.zeros(2, dtype=np.uint64),
             own_dist=np.array([8, 8]), opp_dist=np.array([8, 8]),
             walls_left_own=np.array([10, 10]), walls_left_opp=np.array([10, 10]),
             policy=policy, value=np.array([-1, 1], dtype=np.float32),
             weight=np.array([1, 34.1], dtype=np.float32), is_val=np.array([False, True]),
             group_id=np.array(["train", "val"]))
    config = dict(runner.CONFIG, out_dir=str(tmp_path / "matrix"), device="cpu",
                  data_sources=[dict(name="historical", path=str(dataset), historical=True,
                                     fraction=1, weight_cap=30)])
    initial = tmp_path / "initial.bin"
    export(Student("multipath_phase_bucketed", 512, qat=True), initial)
    entry = dict(runner.NETWORKS[0], initial=str(initial))
    recipe = runner.training_config(config, entry, tmp_path / "mixed")
    folder = Path(recipe["out_dir"])
    folder.mkdir(parents=True)
    shutil.copyfile(initial, folder / "student.bin")
    metrics = runner.frozen_retention(config, entry, recipe)["historical"]
    assert metrics["target_basis"] == "unchanged_historical_targets"
    assert metrics["validation_samples"] == 1
    assert metrics["loss_delta"] == 0
    assert (folder / "frozen_retention.json").is_file()


def test_orchestrator_runs_tiny_shared_recipe_and_resumes(tmp_path, monkeypatch):
    import numpy as np
    torch.set_num_threads(2)
    source_records = []
    for name, historical, pawn in (("historical", True, 4), ("new", False, 6)):
        folder = tmp_path / name
        folder.mkdir()
        dataset = folder / "dataset.npz"
        policy = np.zeros((2, 209), dtype=np.float32)
        policy[:, 0] = 1
        np.savez(dataset, own_pawn=np.array([pawn, pawn + 1]), opp_pawn=np.array([76, 76]),
                 walls_h=np.zeros(2, dtype=np.uint64), walls_v=np.zeros(2, dtype=np.uint64),
                 own_dist=np.array([8, 8]), opp_dist=np.array([8, 8]),
                 walls_left_own=np.array([10, 10]), walls_left_opp=np.array([10, 10]),
                 policy=policy, value=np.array([-0.2, 0.2], dtype=np.float32),
                 weight=np.array([1, 34.1 if historical else 1], dtype=np.float32),
                 is_val=np.array([False, True]), group_id=np.array(["train", "val"]))
        entry = dict(name=name, path=str(dataset), historical=historical,
                     weight_cap=30 if historical else None)
        if historical:
            entry["fraction"] = 0.25
        else:
            entry.update(fraction=None, pool="new", pool_fraction=0.75)
            (folder / "replay_manifest.json").write_text(json.dumps(dict(
                complete=True, dataset_sha256=runner.sha(dataset),
                config=dict(mode="stored_search", stored_gamma=1, stored_outcome_weight=0.5))))
        source_records.append(entry)
    initial = tmp_path / "initial.bin"
    export(Student("multipath_phase_bucketed", 512, qat=True), initial)
    entry = dict(runner.NETWORKS[0], initial=str(initial))
    monkeypatch.setattr(runner, "initialize", lambda _: [entry])
    config = copy.deepcopy(runner.CONFIG)
    config.update(out_dir=str(tmp_path / "matrix"), networks=[runner.NETWORKS[0]],
                  data_sources=source_records, epochs=1, checkpoint_every=1,
                  batch_size=2, device="cpu", build=False, dry_run=False)
    report = runner.run(config)
    assert report["status"] == "training_complete_promotion_pending"
    assert report["jobs"][0]["epochs"] == 1
    assert report["jobs"][0]["frozen_retention"]["historical"]["target_basis"] == "unchanged_historical_targets"
    assert runner.run(config)["jobs"][0]["epochs"] == 1
