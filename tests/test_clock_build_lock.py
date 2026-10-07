"""Verify that concurrent processes cannot build one cache at the same time."""
import subprocess
import sys
from pathlib import Path

import pytest

from tools.external.build_lock import build_lock


def test_second_process_waits_for_the_cache_lock(tmp_path: Path):
    code = """
from pathlib import Path
import sys
from tools.external.build_lock import build_lock
try:
    with build_lock(Path(sys.argv[1]), timeout_s=0.2):
        print('acquired')
except RuntimeError:
    print('locked')
"""
    root = Path(__file__).resolve().parents[1]
    with build_lock(tmp_path):
        blocked = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                                 cwd=root, check=True, capture_output=True, text=True, timeout=10)
    released = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                              cwd=root, check=True, capture_output=True, text=True, timeout=10)
    assert blocked.stdout.strip() == "locked"
    assert released.stdout.strip() == "acquired"


def test_exception_releases_the_cache_lock(tmp_path: Path):
    with pytest.raises(ValueError):
        with build_lock(tmp_path):
            raise ValueError("The build failed")
    with build_lock(tmp_path, timeout_s=0.2):
        pass
