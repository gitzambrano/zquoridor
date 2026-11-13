import json
from pathlib import Path
import pytest

from training.new_experiment import CONFIG, scaffold_experiment


def test_scaffold_experiment_creates_valid_structure(tmp_path, monkeypatch):
    monkeypatch.setattr("training.new_experiment.ROOT", tmp_path)
    cfg = dict(CONFIG)
    cfg["name"] = "test_experiment_scaffold"
    cfg["hypothesis"] = "Testing experiment scaffold generation."

    created_dir = scaffold_experiment(cfg)
    assert created_dir.exists()
    assert (created_dir / "config.json").is_file()
    assert (created_dir / "README.md").is_file()

    with (created_dir / "config.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "config" in loaded
    assert loaded["config"]["architecture"] == "multipath_phase_contact_bucketed"
    assert loaded["config"]["out_dir"] == "experimental/experiments/test_experiment_scaffold"

    readme_text = (created_dir / "README.md").read_text(encoding="utf-8")
    assert "Testing experiment scaffold generation." in readme_text
    assert "Strict Promotion Criteria" in readme_text


def test_scaffold_experiment_prevents_accidental_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("training.new_experiment.ROOT", tmp_path)
    cfg = dict(CONFIG)
    cfg["name"] = "duplicate_experiment"

    scaffold_experiment(cfg)

    with pytest.raises(FileExistsError):
        scaffold_experiment(cfg)

    cfg["force"] = True
    scaffold_experiment(cfg)
