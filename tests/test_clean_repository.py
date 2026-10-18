"""Tests for tools.clean_repository."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools import clean_repository as cr


def test_clean_repository_dry_run_preserves_targets(tmp_path: Path):
    """Verify dry_run reports items without removing them."""
    pycache_dir = tmp_path / "mod" / "__pycache__"
    pycache_dir.mkdir(parents=True)
    dummy_file = pycache_dir / "mod.cpython-311.pyc"
    dummy_file.write_bytes(b"\x00" * 128)

    cfg = {
        "root": tmp_path,
        "clean_pycache": True,
        "clean_input_cache": True,
        "clean_staging_interrupted": True,
        "clean_transient_recovery": True,
        "dry_run": True,
        "verbose": False,
    }

    result = cr.clean_repository(cfg)
    assert result["dry_run"] is True
    assert result["items_found"] == 1
    assert result["items_deleted"] == 0
    assert result["bytes_reclaimed"] == 128
    assert dummy_file.exists()
    assert pycache_dir.exists()


def test_clean_repository_removes_pycache_and_pytest_cache(tmp_path: Path):
    """Verify execution removes __pycache__ and .pytest_cache."""
    pycache_dir = tmp_path / "sub" / "__pycache__"
    pycache_dir.mkdir(parents=True)
    (pycache_dir / "test.pyc").write_bytes(b"\x01" * 64)

    pytest_dir = tmp_path / ".pytest_cache" / "v"
    pytest_dir.mkdir(parents=True)
    (pytest_dir / "cache.json").write_text("{}", encoding="utf-8")

    cfg = {
        "root": tmp_path,
        "clean_pycache": True,
        "clean_input_cache": False,
        "clean_staging_interrupted": False,
        "clean_transient_recovery": False,
        "dry_run": False,
        "verbose": False,
    }

    result = cr.clean_repository(cfg)
    assert result["items_found"] == 2
    assert result["items_deleted"] == 2
    assert not pycache_dir.exists()
    assert not pytest_dir.exists()


def test_clean_repository_removes_input_cache(tmp_path: Path):
    """Verify execution cleans .input_cache under experimental directories."""
    input_cache = tmp_path / "experimental" / "experiments" / "lab" / "dataset" / ".input_cache" / "hash123"
    input_cache.mkdir(parents=True)
    (input_cache / "policy.npy").write_bytes(b"\x00" * 256)

    cfg = {
        "root": tmp_path,
        "clean_pycache": False,
        "clean_input_cache": True,
        "clean_staging_interrupted": False,
        "clean_transient_recovery": False,
        "dry_run": False,
        "verbose": False,
    }

    result = cr.clean_repository(cfg)
    assert result["items_found"] == 1
    assert result["items_deleted"] == 1
    assert not (tmp_path / "experimental" / "experiments" / "lab" / "dataset" / ".input_cache").exists()


def test_clean_repository_removes_staging_interrupted(tmp_path: Path):
    """Verify interrupted directories and tmp files in selfplay staging are deleted."""
    staging = tmp_path / "data" / "selfplay" / "campaign" / "staging"
    interrupted = staging / "interrupted_12_987654321"
    interrupted.mkdir(parents=True)
    (interrupted / "shard.bin.tmp").write_bytes(b"\x00" * 512)

    stray_tmp = staging / "shard_00001.bin.tmp"
    stray_tmp.write_bytes(b"\x00" * 1024)

    # Valid shard that must be kept
    valid_shard = staging / "shard_00002.bin"
    valid_shard.write_bytes(b"\x00" * 2048)

    cfg = {
        "root": tmp_path,
        "clean_pycache": False,
        "clean_input_cache": False,
        "clean_staging_interrupted": True,
        "clean_transient_recovery": False,
        "dry_run": False,
        "verbose": False,
    }

    result = cr.clean_repository(cfg)
    assert result["items_found"] == 2
    assert result["items_deleted"] == 2
    assert not interrupted.exists()
    assert not stray_tmp.exists()
    assert valid_shard.exists()


def test_clean_repository_removes_transient_recovery(tmp_path: Path):
    """Verify transient recovery backups are removed."""
    selfplay = tmp_path / "data" / "selfplay" / "corpus"
    selfplay.mkdir(parents=True)
    recovery_tmp = selfplay / "manifest-recovery-20261018-120000.tmp"
    recovery_tmp.write_bytes(b"\x00" * 100)
    recovery_bak = selfplay / "manifest.json.recovery.bak"
    recovery_bak.write_bytes(b"\x00" * 200)

    # Valid manifest that must be kept
    manifest = selfplay / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    cfg = {
        "root": tmp_path,
        "clean_pycache": False,
        "clean_input_cache": False,
        "clean_staging_interrupted": False,
        "clean_transient_recovery": True,
        "dry_run": False,
        "verbose": False,
    }

    result = cr.clean_repository(cfg)
    assert result["items_found"] == 2
    assert result["items_deleted"] == 2
    assert not recovery_tmp.exists()
    assert not recovery_bak.exists()
    assert manifest.exists()


def test_clean_repository_never_touches_protected_files(tmp_path: Path):
    """Ensure tracked files, private configs, and profile folders are preserved."""
    # Fake a tracked file
    tracked_file = tmp_path / "important.tmp"
    tracked_file.write_bytes(b"do not delete")
    tracked = {tracked_file}

    # Private config
    private_file = tmp_path / "colab" / "private.py"
    private_file.parent.mkdir(parents=True)
    private_file.write_text("# secret", encoding="utf-8")

    assert cr.is_protected(tracked_file, tracked) is True
    assert cr.is_protected(private_file, set()) is True
    assert cr.is_protected(tmp_path / ".git" / "HEAD", set()) is True
    assert cr.is_protected(tmp_path / "colab" / "profiles" / "worker1", set()) is True
