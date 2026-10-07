from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from scripts.colab import config as colab_config
from scripts.colab import search_games_profile as profile
from tools import run_search_games


def test_profiles_isolate_five_accounts_without_mutating_existing_jobs():
    original = deepcopy(colab_config.WORKERS)
    workers = profile.build_worker_profiles()

    assert list(workers) == [1, 2, 3, 4, 5]
    assert len({worker["drive_dir"] for worker in workers.values()}) == 5
    assert len({worker["seed"] for worker in workers.values()}) == 5
    for worker_id, worker in workers.items():
        assert worker["account"] == original[worker_id]["account"]
        assert worker["notebook_url"] == original[worker_id]["notebook_url"]
        assert worker["drive_dir"] == (
            "/content/drive/MyDrive/zquoridor_data/claustro_search_games_v1/"
            f"worker_{worker_id}"
        )
        assert worker["seed"] == worker_id * 1000001
        assert worker["positions"] == ""
        assert worker["total_games"] == 10000
    workers[1]["target_keywords"].append("changed")
    assert colab_config.WORKERS == original
    assert "changed" not in workers[2]["target_keywords"]


def test_profile_arguments_resolve_to_match_gpu_and_all_default_books():
    worker = profile.build_worker_profile(1)
    args = run_search_games.build_parser().parse_args(shlex.split(worker["extra_args"]))
    config = run_search_games.resolve_config(args)

    assert config["mode"] == "match"
    assert config["claustrophobia_device"] == "gpu"
    assert config["pairs"] == 5000
    assert config["workers"] == 1
    assert config["schedule_origin"] == "opening"
    assert config["start_move_time_ms"] == 400
    assert config["end_move_time_ms"] == 50
    assert config["decay_start_ply"] == 14
    assert config["decay_end_ply"] == 80
    assert config["resume"] and config["export_targets"]
    assert not config["auto_setup"]
    assert config["zq_executable"].endswith("/worker_1/zquoridor_uci")
    assert config["claustrophobia_bridge"].endswith("/worker_1/zq_benchmark_bridge")
    assert config["claustrophobia_checkpoint"].endswith("/worker_1/champion.pt")
    assert config["opening_books"] == run_search_games.CONFIG["opening_books"]
    assert config["nnue"] == run_search_games.CONFIG["nnue"]
    assert config["opening_weights"] == {"center_rush": 7, "normal": 2, "weakness": 1}
    assert config["opening_temperature"] == 1.0
    assert config["temperature_plies"] == 14
    assert config["record_both_searches"]
    config["dry_run"] = True
    assert run_search_games.run(config)["pairs_by_book"] == {
        "center_rush": 3500, "normal": 1000, "weakness": 500,
    }


def test_template_supports_generic_launcher_and_valid_ipython_cell():
    worker = profile.build_worker_profile(1)
    cell = profile.BOOTLOADER_TEMPLATE.format(**worker, cmd_args=worker["extra_args"])

    assert cell == profile.render_bootloader(worker)
    assert "drive.mount('/content/drive')" in cell
    assert "torch.cuda.is_available()" in cell
    assert "git reset" not in cell
    assert "-march=native" not in cell
    assert "/content/zquoridor_search_games/" in cell
    assert "--pairs 5000" in cell
    assert "!python -u tools/run_search_games.py" in cell
    assert "%cd " in cell
    transformed = []
    for line in cell.splitlines():
        if line.startswith(("!", "%")):
            transformed.append("pass")
        else:
            transformed.append(line)
    compile("\n".join(transformed), "colab_bootloader", "exec")


def test_revision_override_requires_full_pinned_commit():
    with pytest.raises(ValueError, match="full lowercase Git commit"):
        profile.build_worker_profile(1, config={"revision": "main"})
    revision = "a" * 40
    worker = profile.build_worker_profile(2, config={"revision": revision})
    assert worker["checkout_dir"].endswith(revision)
    assert f"revision = '{revision}'" in profile.render_bootloader(worker)


def test_config_overrides_do_not_mutate_defaults():
    original = deepcopy(profile.CONFIG)
    workers = profile.build_worker_profiles(config={"worker_ids": [2], "pairs": 7})
    assert list(workers) == [2]
    assert workers[2]["total_games"] == 14
    assert "--pairs 7" in workers[2]["extra_args"]
    assert profile.CONFIG == original


def _execute_cached_bootstrap(tmp_path, monkeypatch, *, corrupt=False, missing=False):
    worker = profile.build_worker_profile(1, config={
        "checkout_root": str(tmp_path / "checkout"),
        "drive_root": str(tmp_path / "drive"),
    })
    checkout = Path(worker["checkout_dir"])
    checkout.mkdir(parents=True)
    cache = Path(ast.literal_eval(worker["runtime_cache_literal"]))
    payloads = {
        "zquoridor_uci": b"frozen main binary",
        "zq_benchmark_bridge": b"frozen bridge binary",
        "zq_inference_worker.py": b"frozen inference worker",
        "champion.pt": b"frozen champion checkpoint",
    }
    if missing:
        output = Path(worker["drive_dir"])
        output.mkdir(parents=True)
        (output / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        cache.mkdir(parents=True)
        for name, payload in payloads.items():
            (cache / name).write_bytes(payload)
        (cache / "artifacts.json").write_text(json.dumps({
            "revision": profile.CONFIG["revision"],
            "sha256": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            },
        }), encoding="utf-8")
        if corrupt:
            (cache / "zquoridor_uci").write_bytes(b"modified binary")
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        drive=SimpleNamespace(mount=lambda path: None),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    ))
    import shutil
    import subprocess
    calls = []
    monkeypatch.setattr(shutil, "which", lambda command: f"/tools/{command}")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kwargs:
                        profile.CONFIG["revision"] if command[-1] == "HEAD" else "")
    monkeypatch.setattr("os.chdir", lambda path: None)
    for name in ("PATH", "CARGO_BUILD_JOBS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "test")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : avx2 fma", encoding="utf-8")
    cell = profile.render_bootloader(worker).split("%cd ", 1)[0]
    cell = cell.replace("Path('/proc/cpuinfo')", f"Path({str(cpuinfo)!r})")
    exec(compile(cell, "colab_bootstrap", "exec"), {})
    return worker, payloads, calls


def test_restart_restores_frozen_artifacts_without_rebuild(tmp_path, monkeypatch):
    worker, payloads, calls = _execute_cached_bootstrap(tmp_path, monkeypatch)
    runtime = Path(ast.literal_eval(worker["runtime_dir_literal"]))

    for name, payload in payloads.items():
        assert (runtime / name).read_bytes() == payload
    assert calls == [["cargo", "--version"]]


def test_restart_rejects_corrupted_cache(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="modified artifact"):
        _execute_cached_bootstrap(tmp_path, monkeypatch, corrupt=True)


def test_restart_preserves_manifest_when_cache_is_missing(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="frozen runtime cache is absent"):
        _execute_cached_bootstrap(tmp_path, monkeypatch, missing=True)
