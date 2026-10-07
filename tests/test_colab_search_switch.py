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


def test_existing_handover_does_not_submit_another_helper(monkeypatch):
    class Locator:
        def click(self):
            pass

        def wait_for(self, **kwargs):
            pass

        def all_text_contents(self):
            return ["/content ZQ_HANDOVER_WAITING_5"]

    class Page:
        def locator(self, selector):
            return Locator()

    class Expectation:
        def to_contain_text(self, *args, **kwargs):
            pass

    import playwright.sync_api
    monkeypatch.setattr(playwright.sync_api, "expect", lambda locator: Expectation())
    switch.start_handover(Page(), 5)


def test_watcher_bounds_driver_recovery(monkeypatch, tmp_path):
    from scripts.colab import watch_search_games as watch
    calls = []
    delays = []

    def fail_driver():
        calls.append(1)
        raise RuntimeError("Connection closed while reading from the driver")

    monkeypatch.setitem(watch.CONFIG, "output", str(tmp_path))
    monkeypatch.setattr(watch, "switch_main", fail_driver)
    monkeypatch.setattr(watch.time, "sleep", delays.append)
    watch.main()
    assert len(calls) == 4
    assert delays == [60, 60, 60]


def test_authentication_failure_does_not_restart_driver():
    from scripts.colab.watch_search_games import driver_failed
    assert not driver_failed({"1": {"status": "blocked", "error": "Auth Required"}})
