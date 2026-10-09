"""Check the production-only recipe and the preparation boundary."""
import copy
from training import run_production_finetune as runner


def test_dry_run_keeps_all_sources_without_training(tmp_path, monkeypatch):
    config = copy.deepcopy(runner.CONFIG)
    config["out_dir"] = str(tmp_path / "experiment")
    monkeypatch.setattr(runner, "prepare", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dry run must not prepare data")))
    report = runner.run(config)
    assert len(report["raw_sources"]) == 4
    assert report["networks"] == [runner.anneal.NETWORKS[0]]
    assert report["evaluation"]["games_per_opponent"] == 400
    assert report["evaluation"]["enabled"] is False
    assert report["evaluation"]["pairs_per_opponent"] == 200
    assert report["local_selfplay"] == "stopped"
    assert not (tmp_path / "experiment").exists()


def test_historical_weights_cannot_exceed_the_block_budget():
    recipe = runner.training_plan(runner.CONFIG)
    old, new = recipe["data_sources"]
    assert old["fraction"] == 0.25 and old["weight_cap"] == 30
    assert new["fraction"] == 0.75 and new["weight_cap"] is None
    assert recipe["epochs"] == 120
    assert recipe["lr"] == 1e-5 and recipe["min_lr"] == 1e-7
    assert recipe["trunk_lr_scale"] == 0.05
    assert recipe["checkpoint_every"] == 10


def test_prepare_uses_all_manifests_before_training(tmp_path, monkeypatch):
    config = copy.deepcopy(runner.CONFIG)
    config.update(out_dir=str(tmp_path / "experiment"), dry_run=False, evaluate=False)
    calls = []

    def prepare(sources, output, **kwargs):
        calls.append(("prepare", sources, output, kwargs))
        return dict(raw_records=100, accepted_records=90, samples=80)

    original = runner.anneal.run

    def run(recipe):
        if recipe["dry_run"]:
            return original(recipe)
        calls.append(("train", recipe))
        return dict(status="training_complete_promotion_pending")

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner.anneal, "run", run)
    runner.run(config)
    assert [call[0] for call in calls] == ["prepare", "train"]
    assert len(calls[0][1]) == 4
    assert calls[1][1]["networks"] == [runner.anneal.NETWORKS[0]]
