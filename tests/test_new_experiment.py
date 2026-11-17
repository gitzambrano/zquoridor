import json
import re
import shlex
from pathlib import Path

import pytest

from training import new_experiment
from training.new_experiment import CONFIG, scaffold_experiment

ROOT = Path(__file__).resolve().parents[1]


def _scaffold(tmp_path, monkeypatch, name, **overrides):
    monkeypatch.setattr(new_experiment, "ROOT", tmp_path)
    cfg = dict(CONFIG, name=name, **overrides)
    return scaffold_experiment(cfg)


def _commands(readme_text):
    return [line for block in re.findall(r"```bash\n(.*?)```", readme_text, re.S)
            for line in block.splitlines() if line.strip()]


def test_scaffold_experiment_creates_valid_structure(tmp_path, monkeypatch):
    created_dir = _scaffold(tmp_path, monkeypatch, "test_experiment_scaffold",
                            hypothesis="Testing experiment scaffold generation.")
    loaded = json.loads((created_dir / "config.json").read_text(encoding="utf-8"))
    assert loaded["config"]["architecture"] == "multipath_phase_contact_bucketed"
    assert loaded["config"]["out_dir"] == "experimental/experiments/test_experiment_scaffold"
    assert loaded["config"]["init_architecture"]
    assert loaded["config"]["init_hidden"] > 0

    readme_text = (created_dir / "README.md").read_text(encoding="utf-8")
    assert "Testing experiment scaffold generation." in readme_text
    assert "Promotion criteria" in readme_text


def test_scaffold_config_keys_belong_to_runner(tmp_path, monkeypatch):
    created_dir = _scaffold(tmp_path, monkeypatch, "key_contract")
    loaded = json.loads((created_dir / "config.json").read_text(encoding="utf-8"))
    assert set(loaded["config"]) <= new_experiment.runner_keys()


def test_readme_commands_parse_with_real_runners(tmp_path, monkeypatch):
    from training import run_experiment
    from tools import run_full_candidate_suite

    created_dir = _scaffold(tmp_path, monkeypatch, "command_contract")
    train_cmd, suite_cmd = _commands((created_dir / "README.md").read_text(encoding="utf-8"))

    train_args = shlex.split(train_cmd)
    assert train_args[1] == "training/run_experiment.py"
    config_path = tmp_path / train_args[train_args.index("--config") + 1]
    config = run_experiment.parse_config(["--config", str(config_path)])
    assert config["out_dir"] == "experimental/experiments/command_contract"
    assert config["epochs"] == CONFIG["epochs"]

    suite_args = shlex.split(suite_cmd)
    assert suite_args[1] == new_experiment.CANDIDATE_SUITE
    flags = {arg for arg in suite_args[2:] if arg.startswith("--")}
    suite_config = run_full_candidate_suite.CONFIG
    assert {"--candidate-exe", "--candidate-nnue"} == flags
    assert {"candidate_exe", "candidate_nnue"} <= set(suite_config)


def test_runner_rejects_unknown_config_keys(tmp_path):
    from training import run_experiment

    recipe = tmp_path / "bad.json"
    recipe.write_text(json.dumps({"config": {"no_such_key": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_key"):
        run_experiment.parse_config(["--config", str(recipe)])


def test_cli_override_wins_over_config_file(tmp_path):
    from training import run_experiment

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"epochs": 7, "hidden": 768}), encoding="utf-8")
    config = run_experiment.parse_config(["--config", str(recipe), "--epochs", "3"])
    assert config["epochs"] == 3
    assert config["hidden"] == 768


def test_scaffold_experiment_prevents_accidental_overwrite(tmp_path, monkeypatch):
    _scaffold(tmp_path, monkeypatch, "duplicate_experiment")
    with pytest.raises(FileExistsError):
        _scaffold(tmp_path, monkeypatch, "duplicate_experiment")
    _scaffold(tmp_path, monkeypatch, "duplicate_experiment", force=True)


def test_scaffold_rejects_nested_names(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        _scaffold(tmp_path, monkeypatch, "../escape")
