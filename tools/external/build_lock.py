"""Serialize builds that share one artifact cache."""
from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def build_lock(directory: Path, timeout_s: float = 300.0):
    """Hold a process lock until the build finishes or its process exits."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".build.lock").open("a+b") as stream:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"The build cache remains locked: {directory}")
                time.sleep(0.1)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
