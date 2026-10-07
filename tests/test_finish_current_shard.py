from __future__ import annotations

import json
from pathlib import Path
import signal
import sys

import pytest

from scripts.colab import finish_current_shard as handover


@pytest.fixture(autouse=True)
def _linux_signal_constants(monkeypatch):
    monkeypatch.setattr(signal, "SIGSTOP", getattr(signal, "SIGSTOP", 19), raising=False)
    monkeypatch.setattr(signal, "SIGCONT", getattr(signal, "SIGCONT", 18), raising=False)


def _process(pid, args, *, ppid=1, state="S", exit_status=0):
    return {"pid": pid, "args": args, "ppid": ppid, "state": state,
            "start_ticks": pid * 10, "exit_status": exit_status}


def _fixture(tmp_path, monkeypatch, *, exit_status=0, corrupt=False):
    binary = tmp_path / "c1_shard_0004.bin"
    metadata = binary.with_suffix(".meta")
    binary.write_bytes(b"x" * 128)
    metadata.write_bytes(b"m" * (20 if corrupt else 40))
    launcher = _process(100, ["python3", "/content/zquoridor/tools/selfplay/run_colab_worker.py",
                              "--worker-id", "1"])
    child = _process(101, ["/content/zquoridor/bin/selfplay", "--out", str(binary),
                           "--meta-out", str(metadata)], ppid=100, state="Z",
                     exit_status=exit_status)
    signals = []
    monkeypatch.setattr(handover, "processes", lambda: [dict(launcher), dict(child)])
    monkeypatch.setattr(handover, "process_info", lambda pid: launcher if pid == 100 else child)

    def send(record, signum):
        signals.append((record["pid"], signum))
        if signum == signal.SIGSTOP:
            launcher["state"] = "T"
        return True

    monkeypatch.setattr(handover, "checked_signal", send)
    config = {**handover.CONFIG, "status_path": str(tmp_path / "status.json")}
    return config, signals, launcher, child


def test_saved_shard_stops_only_launcher_after_file_verification(tmp_path, monkeypatch):
    config, signals, _, _ = _fixture(tmp_path, monkeypatch)
    result = handover.finish_current_shard(config)

    assert result["state"] == "ready"
    assert result["native_exit_status"] == 0
    assert result["artifacts"]["records"] == 2
    assert result["artifacts"]["binary"]["size"] == 128
    assert len(result["artifacts"]["binary"]["sha256"]) == 64
    assert signals == [(100, signal.SIGSTOP), (100, signal.SIGTERM), (100, signal.SIGCONT)]
    assert json.loads(Path(config["status_path"]).read_text()) == result


@pytest.mark.parametrize("exit_status,corrupt", [(256, False), (0, True)])
def test_failure_resumes_launcher_and_preserves_child(tmp_path, monkeypatch, exit_status, corrupt):
    config, signals, _, _ = _fixture(tmp_path, monkeypatch,
                                    exit_status=exit_status, corrupt=corrupt)
    with pytest.raises(RuntimeError):
        handover.finish_current_shard(config)

    assert signals == [(100, signal.SIGSTOP), (100, signal.SIGCONT)]
    assert json.loads(Path(config["status_path"]).read_text())["state"] == "failed"
    assert (tmp_path / "c1_shard_0004.bin").exists()
    assert (tmp_path / "c1_shard_0004.meta").exists()


def test_missing_launcher_requires_no_orphan_native_child(tmp_path, monkeypatch):
    config, signals, _, child = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(handover, "processes", lambda: [child])
    with pytest.raises(RuntimeError, match="without its launcher"):
        handover.finish_current_shard(config)
    assert signals == []
    monkeypatch.setattr(handover, "processes", lambda: [])
    assert handover.finish_current_shard(config)["reason"] == "no_active_worker"


def test_duplicate_launchers_fail_before_signals(tmp_path, monkeypatch):
    config, signals, launcher, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(handover, "processes", lambda: [launcher, {**launcher, "pid": 102}])
    with pytest.raises(RuntimeError, match="More than one launcher"):
        handover.finish_current_shard(config)
    assert signals == []


def test_zombie_retains_shard_paths_from_initial_snapshot(tmp_path, monkeypatch):
    config, _, launcher, child = _fixture(tmp_path, monkeypatch)
    calls = 0

    def records():
        nonlocal calls
        calls += 1
        return [launcher, child if calls == 1 else {**child, "args": []}]

    monkeypatch.setattr(handover, "processes", records)
    assert handover.finish_current_shard(config)["artifacts"]["records"] == 2


def test_process_stat_handles_spaces_and_exit_status(tmp_path):
    directory = tmp_path / "123"
    directory.mkdir()
    fields = ["0"] * 50
    fields[0], fields[1], fields[19], fields[49] = "Z", "100", "456", "256"
    (directory / "stat").write_text("123 (process name) " + " ".join(fields), encoding="utf-8")
    (directory / "cmdline").write_bytes(b"python3\0run_colab_worker.py\0--worker-id=1\0")
    record = handover.process_info(123, tmp_path)

    assert record["ppid"] == 100
    assert record["start_ticks"] == 456
    assert record["exit_status"] == 256
    assert handover.is_launcher(record, 1)


def test_signal_refuses_reused_process_identifier(monkeypatch):
    original = _process(100, ["python3"])
    monkeypatch.setattr(handover, "process_info", lambda pid: {**original, "start_ticks": 999})
    monkeypatch.setattr(handover.os, "kill", lambda *args: pytest.fail("A reused PID received a signal."))
    assert not handover.checked_signal(original, signal.SIGSTOP)


@pytest.mark.parametrize("fails,exit_code,marker", [
    (False, 0, "ZQ_HANDOVER_READY_2"),
    (True, 1, "ZQ_HANDOVER_FAILED_2"),
])
def test_main_emits_controller_marker(monkeypatch, capsys, fails, exit_code, marker):
    monkeypatch.setattr(sys, "argv", ["-c", "--worker-id", "2"])

    def finish(config):
        assert config["status_path"].endswith("handover_worker_2.json")
        if fails:
            raise RuntimeError("The shard failed verification.")
        return {"worker_id": config["worker_id"], "state": "ready"}

    monkeypatch.setattr(handover, "finish_current_shard", finish)
    assert handover.main() == exit_code
    assert marker in capsys.readouterr().out.splitlines()


def test_waiting_marker_follows_verified_freeze(tmp_path, monkeypatch, capsys):
    config, _, _, _ = _fixture(tmp_path, monkeypatch)
    handover.finish_current_shard(config)
    assert "ZQ_HANDOVER_WAITING_1" in capsys.readouterr().out.splitlines()
