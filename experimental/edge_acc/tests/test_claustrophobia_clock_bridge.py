from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_claustrophobia_dynamic_bridge as bridge_builder  # noqa: E402
from tools.external import bot_setup  # noqa: E402


MAIN = Path(r"C:\Projetos\Zquoridor")
UPSTREAM = MAIN / "external_bots/claustrophobia/repo"


def _only_original_library() -> Path:
    libraries = list((UPSTREAM / "target/release/deps").glob("libquoridor-*.rlib"))
    if len(libraries) != 1:
        pytest.skip("the pinned std-only Claustrophobia library is unavailable")
    return libraries[0]


def test_clock_library_preserves_count_search_and_stops_on_time(tmp_path: Path) -> None:
    if not UPSTREAM.is_dir():
        pytest.skip("the local pinned Claustrophobia checkout is unavailable")

    original_library = _only_original_library()
    # Reuse the normal ignored cache; only the tiny test executable goes in tmp.
    bridge_executable = bridge_builder.ensure_bridge(
        shared_root=MAIN,
        output=bridge_builder.CONFIG["output"],
    )
    clock_library = bridge_executable.parent / "libquoridor_clock.rlib"
    assert clock_library.is_file(), "the isolated builder must place its library beside the bridge"
    assert (bridge_executable.parent / "build.json").is_file()

    rustc = Path(bot_setup._find_cargo(MAIN)).with_name("rustc.exe")
    probe_source = Path(__file__).with_name("claustrophobia_deadline_probe.rs")
    probe_executable = tmp_path / "claustrophobia_deadline_probe.exe"
    command = [
        str(rustc),
        "--edition=2021",
        "-C",
        "opt-level=2",
        "-C",
        "debuginfo=0",
        "-C",
        "panic=abort",
        "-C",
        "lto=fat",
        "-C",
        "codegen-units=1",
        "--crate-name",
        "claustrophobia_deadline_probe",
        str(probe_source),
        "--extern",
        f"quoridor_orig={original_library}",
        "--extern",
        f"quoridor_clock={clock_library}",
        "-L",
        f"dependency={original_library.parent}",
        "-L",
        f"dependency={clock_library.parent}",
        "-o",
        str(probe_executable),
    ]
    environment = bot_setup._cargo_environment(MAIN)
    environment["PATH"] = os.pathsep.join(
        [r"C:\mingw64\bin", environment.get("PATH", "")]
    )
    subprocess.run(command, cwd=tmp_path, env=environment, check=True, timeout=120)

    result = subprocess.run(
        [str(probe_executable)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "fixed_count=PASS expired_deadline=PASS timed_wave=PASS" in result.stdout
