import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import setup_bots  # noqa: E402
from tools.external import bot_setup  # noqa: E402


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test User")
    (source / "source.txt").write_text("pinned\n", encoding="utf-8")
    (source / "Cargo.toml").write_text("# original\n", encoding="utf-8")
    _git(source, "add", "source.txt", "Cargo.toml")
    _git(source, "commit", "-m", "pinned")
    return source, _git(source, "rev-parse", "HEAD")


def test_verified_download_reuses_a_matching_cache(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"verified artifact")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()

    bot_setup._download_verified("missing://must-not-open", destination, digest)

    assert destination.read_bytes() == b"verified artifact"


def test_verified_download_rejects_a_bad_hash_without_partial_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"wrong artifact")
    destination = tmp_path / "cache" / "artifact.bin"

    with pytest.raises(RuntimeError, match="SHA-256"):
        bot_setup._download_verified(source.as_uri(), destination, "0" * 64)

    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_checkout_reuses_the_pinned_commit_and_rejects_unknown_content(
    tmp_path: Path,
) -> None:
    source, revision = _source_repository(tmp_path)
    checkout = tmp_path / "checkout"

    bot_setup._ensure_checkout(str(source), revision, checkout)
    assert _git(checkout, "rev-parse", "HEAD") == revision

    bot_setup._ensure_checkout("invalid://unused", revision, checkout)
    assert _git(checkout, "status", "--porcelain") == ""

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "keep.txt").write_text("user data\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuses to replace"):
        bot_setup._ensure_checkout(str(source), revision, unsafe)
    assert (unsafe / "keep.txt").read_text(encoding="utf-8") == "user data\n"


def test_bridge_sources_refresh_by_content_and_cargo_entries_are_idempotent(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    bridge_dir = project / "training" / "teachers"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "claustrophobia_encode_bridge.rs").write_text(
        "fn main() { println!(\"new\"); }\n", encoding="utf-8"
    )
    (bridge_dir / "claustrophobia_search_bridge.rs").write_text(
        "fn main() { println!(\"search\"); }\n", encoding="utf-8"
    )
    checkout = tmp_path / "checkout"
    (checkout / "src" / "bin").mkdir(parents=True)
    (checkout / "src" / "bin" / "zq_encode_bridge.rs").write_text(
        "fn main() { println!(\"old\"); }\n", encoding="utf-8"
    )
    (checkout / "Cargo.toml").write_text(
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
        encoding="utf-8",
    )

    changed = bot_setup._prepare_claustrophobia_sources(project, checkout)
    changed_again = bot_setup._prepare_claustrophobia_sources(project, checkout)

    assert changed is True
    assert changed_again is False
    assert "println!(\"new\")" in (
        checkout / "src" / "bin" / "zq_encode_bridge.rs"
    ).read_text(encoding="utf-8")
    cargo = (checkout / "Cargo.toml").read_text(encoding="utf-8")
    assert cargo.count('name = "zq_encode_bridge"') == 1
    assert cargo.count('name = "zq_search_bridge"') == 1


def test_managed_checkout_digest_accepts_only_unchanged_generated_files(
    tmp_path: Path,
) -> None:
    source, revision = _source_repository(tmp_path)
    checkout = tmp_path / "checkout"
    bot_setup._ensure_checkout(str(source), revision, checkout)
    (checkout / "src" / "bin").mkdir(parents=True)
    bridge = checkout / "src" / "bin" / "zq_encode_bridge.rs"
    bridge.write_text("fn main() {}\n", encoding="utf-8")
    (checkout / "Cargo.toml").write_text("# managed\n", encoding="utf-8")
    bot_setup._record_managed_checkout(checkout)

    bot_setup._ensure_checkout("invalid://unused", revision, checkout, allow_managed=True)

    bridge.write_text("fn main() { panic!(); }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="local changes"):
        bot_setup._ensure_checkout(
            "invalid://unused", revision, checkout, allow_managed=True
        )


def test_cli_values_override_each_config_option(tmp_path: Path) -> None:
    config = {
        "bots": ["titanium"],
        "root": ROOT,
        "build": True,
        "dry_run": False,
    }

    options = setup_bots.parse_args(
        [
            "--bots",
            "claustrophobia",
            "--root",
            str(tmp_path),
            "--no-build",
            "--dry-run",
        ],
        config,
    )

    assert options.bots == ["claustrophobia"]
    assert options.root == tmp_path.resolve()
    assert options.build is False
    assert options.dry_run is True


def test_cli_runs_directly_from_the_project_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/setup_bots.py",
            "--bots",
            "titanium",
            "--root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert Path(output["titanium"]["executable"]).is_relative_to(tmp_path.resolve())
