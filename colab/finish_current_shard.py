#!/usr/bin/env python3
"""Finish one active self-play shard before the Colab handover."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time


CONFIG = {
    "worker_id": 1,
    "status_path": "/content/zquoridor_search_games/handover_worker_1.json",
    "poll_seconds": 2.0,
}


def process_info(pid: int, proc_root: Path = Path("/proc")) -> dict | None:
    """Read a process identity and its command arguments."""
    try:
        directory = proc_root / str(pid)
        stat = (directory / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rindex(")") + 2:].split()
        arguments = (directory / "cmdline").read_bytes().split(b"\0")
        return {
            "pid": pid, "state": fields[0], "ppid": int(fields[1]),
            "start_ticks": int(fields[19]),
            "exit_status": int(fields[49]) if len(fields) > 49 else None,
            "args": [os.fsdecode(argument) for argument in arguments if argument],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def processes(proc_root: Path = Path("/proc")) -> list[dict]:
    """Return the current Linux process records."""
    return [record for directory in proc_root.iterdir() if directory.name.isdigit()
            if (record := process_info(int(directory.name), proc_root)) is not None]


def option(arguments: list[str], flag: str) -> str | None:
    """Read a separate or equals-form command option."""
    for index, argument in enumerate(arguments):
        if argument == flag:
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if argument.startswith(flag + "="):
            return argument[len(flag) + 1:]
    return None


def is_launcher(record: dict, worker_id: int) -> bool:
    """Identify the Python launcher for one worker."""
    args = record["args"]
    return bool(args and Path(args[0]).name.startswith("python")
                and any(Path(arg).name == "run_colab_worker.py" for arg in args[1:])
                and option(args, "--worker-id") == str(worker_id))


def shard_paths(record: dict, worker_id: int) -> tuple[Path, Path] | None:
    """Identify the native shard outputs for one worker."""
    args = record["args"]
    binary, metadata = option(args, "--out"), option(args, "--meta-out")
    if not binary or not metadata:
        return None
    binary_path, metadata_path = Path(binary), Path(metadata)
    pattern = rf"c{worker_id}_shard_\d+"
    if (not re.fullmatch(pattern + r"\.bin", binary_path.name)
            or metadata_path != binary_path.with_suffix(".meta")):
        return None
    return binary_path, metadata_path


def atomic_status(path: Path, value: dict) -> None:
    """Replace the handover status after a complete JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".handover_", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_artifacts(binary: Path, metadata: Path) -> dict:
    """Verify aligned records and synchronize both shard files."""
    sizes = [binary.stat().st_size, metadata.stat().st_size]
    if (sizes[0] <= 0 or sizes[0] % 64 or sizes[1] % 20
            or sizes[0] // 64 != sizes[1] // 20):
        raise RuntimeError("The shard files do not contain aligned binary and metadata records.")
    rows = []
    for path, size in zip((binary, metadata), sizes):
        digest = hashlib.sha256()
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({"path": str(path), "size": size, "sha256": digest.hexdigest()})
    return {"records": sizes[0] // 64, "binary": rows[0], "metadata": rows[1]}


def same_process(expected: dict, actual: dict | None) -> bool:
    """Confirm that a process identifier still names the original process."""
    return actual is not None and actual["start_ticks"] == expected["start_ticks"]


def checked_signal(record: dict, signum: int) -> bool:
    """Signal the original process if its identity still matches."""
    if not same_process(record, process_info(record["pid"])):
        return False
    os.kill(record["pid"], signum)
    return True


def finish_current_shard(config: dict) -> dict:
    """Freeze the launcher until its native child completes the shard."""
    worker_id = int(config["worker_id"])
    status_path = Path(config["status_path"])
    poll_seconds = float(config["poll_seconds"])
    if worker_id <= 0 or poll_seconds <= 0:
        raise ValueError("The worker identifier and poll interval must be positive.")
    launcher = None
    frozen = False
    base = {"worker_id": worker_id}
    try:
        records = processes()
        launchers = [record for record in records if is_launcher(record, worker_id)]
        if len(launchers) > 1:
            raise RuntimeError("More than one launcher matches this worker.")
        if not launchers:
            if any(shard_paths(record, worker_id) for record in records):
                raise RuntimeError("A native shard remains active without its launcher.")
            result = {**base, "state": "ready", "reason": "no_active_worker", "artifacts": None}
            atomic_status(status_path, result)
            return result
        launcher = launchers[0]
        previous_children = {record["pid"]: record for record in records
                             if record["ppid"] == launcher["pid"]
                             and shard_paths(record, worker_id)}
        if not checked_signal(launcher, signal.SIGSTOP):
            raise RuntimeError("The launcher exited before the freeze.")
        frozen = True
        for _ in range(20):
            current = process_info(launcher["pid"])
            if not same_process(launcher, current):
                raise RuntimeError("The launcher exited during the freeze.")
            if current["state"] in ("T", "t"):
                break
            time.sleep(min(poll_seconds, 0.1))
        else:
            raise RuntimeError("The launcher did not enter the stopped state.")
        children = []
        for record in processes():
            if record["ppid"] != launcher["pid"]:
                continue
            previous = previous_children.get(record["pid"])
            if (previous is not None and not shard_paths(record, worker_id)
                    and same_process(previous, record)):
                record = {**record, "args": previous["args"]}
            if shard_paths(record, worker_id):
                children.append(record)
        if len(children) != 1:
            raise RuntimeError("The frozen launcher does not have exactly one native shard child.")
        child = children[0]
        binary, metadata = shard_paths(child, worker_id)
        atomic_status(status_path, {**base, "state": "waiting", "launcher_pid": launcher["pid"],
                                    "child_pid": child["pid"], "binary_path": str(binary),
                                    "metadata_path": str(metadata)})
        print(f"ZQ_HANDOVER_WAITING_{worker_id}", flush=True)
        exit_status = None
        while True:
            current_launcher = process_info(launcher["pid"])
            if (not same_process(launcher, current_launcher)
                    or current_launcher["state"] not in ("T", "t")):
                raise RuntimeError("The launcher did not remain frozen during the shard.")
            current = process_info(child["pid"])
            if not same_process(child, current):
                break
            if current["state"] == "Z":
                exit_status = current["exit_status"]
                break
            time.sleep(poll_seconds)
        if exit_status is not None and exit_status != 0:
            raise RuntimeError(f"The native shard exited with wait status {exit_status}.")
        artifacts = inspect_artifacts(binary, metadata)
        result = {**base, "state": "ready", "reason": "shard_saved", "artifacts": artifacts,
                  "launcher_pid": launcher["pid"], "child_pid": child["pid"],
                  "native_exit_status": exit_status}
        atomic_status(status_path, result)
        if checked_signal(launcher, signal.SIGTERM):
            checked_signal(launcher, signal.SIGCONT)
        frozen = False
        return result
    except BaseException as error:
        if frozen and launcher is not None:
            try:
                checked_signal(launcher, signal.SIGCONT)
            except ProcessLookupError:
                pass
        atomic_status(status_path, {**base, "state": "failed", "error": str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--status-path")
    parser.add_argument("--poll-seconds", type=float)
    args = parser.parse_args()
    config = {**CONFIG, **{key: value for key, value in vars(args).items() if value is not None}}
    if args.worker_id is not None and args.status_path is None:
        config["status_path"] = str(Path(CONFIG["status_path"]).with_name(
            f"handover_worker_{args.worker_id}.json"))
    try:
        print(json.dumps(finish_current_shard(config), sort_keys=True), flush=True)
        print(f"ZQ_HANDOVER_READY_{config['worker_id']}", flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"state": "failed", "error": str(error)}), flush=True)
        print(f"ZQ_HANDOVER_FAILED_{config['worker_id']}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
