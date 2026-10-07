import hashlib

from scripts.colab import switch_search_games as switch


def test_terminal_command_verifies_the_committed_helper_bytes(monkeypatch):
    revision = "a" * 40
    source = b"print('example')\n"
    def output(args, **kwargs):
        return revision + "\n" if args[1] == "rev-parse" else source
    monkeypatch.setattr(switch.subprocess, "check_output", output)
    command = switch.handover_command(3)
    assert revision + "/scripts/colab/finish_current_shard.py" in command
    assert hashlib.sha256(source).hexdigest() in command
    assert "sha256sum -c - && python -u" in command
    assert command.endswith("--worker-id 3")


def test_switch_defaults_use_the_frozen_collection_revision():
    assert switch.CONFIG["revision"] == switch.PROFILE_CONFIG["revision"]
    assert switch.CONFIG["worker_ids"] == [1, 2, 3, 4, 5]
