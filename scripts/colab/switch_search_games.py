"""Finish current self-play shards and launch paired search collection in Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.config import WORKERS
from scripts.colab.search_games_profile import BOOTLOADER_TEMPLATE, CONFIG as PROFILE_CONFIG, build_worker_profiles
from scripts.colab.browser_utils import connect_runtime_if_needed, get_notebook_dom_state, is_cdp_reachable, trigger_cell_execution
from tools.external.build_lock import build_lock

CONFIG = {
    "worker_ids": [1, 2, 3, 4, 5],
    "revision": PROFILE_CONFIG["revision"],
    "check_interval_seconds": 30,
    "headless": True,
    "auto_resume": True,
    "retry_cooldown_seconds": 900,
    "max_resume_attempts": 3,
    "output": str(ROOT / "artifacts/colab/search_games_handover"),
}


def handover_command(worker_id: int) -> str:
    """Fetch the pinned helper and verify its bytes before runtime execution."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(
        ["git", "show", f"{revision}:scripts/colab/finish_current_shard.py"], cwd=ROOT)
    digest = hashlib.sha256(source).hexdigest()
    url = (f"https://raw.githubusercontent.com/gitzambrano/zquoridor/{revision}/"
           "scripts/colab/finish_current_shard.py")
    destination = f"/tmp/zq_finish_current_shard_{worker_id}.py"
    checksum = f"{digest}  {destination}"
    return (f"curl -fsSL {shlex.quote(url)} -o {destination} && "
            f"printf '%s\\n' {shlex.quote(checksum)} | sha256sum -c - && "
            f"python -u {destination} --worker-id {worker_id}")


def start_handover(page, worker_id: int) -> None:
    """Request a batch boundary stop through the visible Colab terminal."""
    from playwright.sync_api import expect
    page.locator('md-text-button[command="show-terminal"]').click()
    terminal = page.locator('textarea[aria-label="Terminal input"]')
    terminal.wait_for(state="attached", timeout=60000)
    expect(page.locator('.xterm-rows')).to_contain_text('/content', timeout=60000)
    text = "\n".join(page.locator('.xterm-rows').all_text_contents())
    if any(f"ZQ_HANDOVER_{stage}_{worker_id}" in text for stage in ("WAITING", "READY")):
        return
    terminal.focus()
    page.keyboard.type(handover_command(worker_id))
    page.keyboard.press("Enter")


def run(config: dict) -> None:
    from playwright.sync_api import sync_playwright

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    revision = config["revision"] or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    profiles = build_worker_profiles(config={"revision": revision})
    states = {wid: {"worker_id": wid, "status": "pending", "revision": revision}
              for wid in config["worker_ids"]}
    status_path = output / "status.json"
    if status_path.is_file():
        previous_states = json.loads(status_path.read_text(encoding="utf-8"))
        for wid, state in states.items():
            previous = previous_states.get(str(wid), {})
            if previous.get("revision") == revision:
                for key in ("resume_attempts", "last_resume_at", "last_successful_game", "started_at"):
                    if key in previous:
                        state[key] = previous[key]
    pages = {}
    handles = []
    contexts = {}
    def save_status():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
        temporary.replace(output / "status.json")

    with build_lock(output / "controller_lock"), sync_playwright() as playwright:
        save_status()
        for wid, state in states.items():
            worker = WORKERS[wid]
            try:
                if is_cdp_reachable(worker["cdp_port"]):
                    browser = playwright.chromium.connect_over_cdp(
                        f"http://127.0.0.1:{worker['cdp_port']}")
                    context = browser.contexts[0]
                    handles.append(browser)
                else:
                    context = playwright.chromium.launch_persistent_context(
                        worker["profile_dir"], channel="chrome", headless=config["headless"],
                        args=[f"--remote-debugging-port={worker['cdp_port']}"])
                    handles.append(context)
                page = next((p for p in context.pages if p.url == worker["notebook_url"]), None)
                if page is None:
                    page = context.pages[0] if context.pages else context.new_page()
                    page.goto(worker["notebook_url"], wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(12000)
                contexts[wid] = context
                pages[wid] = page
                previous = get_notebook_dom_state(page, profiles[wid]["target_keywords"])
                has_collection = page.evaluate("""() => {
                    const cells = globalThis.colab?.global?.notebook?.cells || [];
                    return cells.some(c => c.getText?.().includes('run_search_games.py'));
                }""")
                if has_collection:
                    state["status"] = "collection_starting"
                    state.setdefault("started_at", time.time())
                    print(json.dumps(state), flush=True)
                    save_status()
                    continue
                start_handover(page, wid)
                state["status"] = "handover_requested"
            except Exception as error:
                state.update(status="blocked", error=str(error))
            print(json.dumps(state), flush=True)
            save_status()
        while True:
            for wid, state in states.items():
                if state["status"] != "blocked":
                    continue
                attempts = state.get("browser_recovery_attempts", 0)
                if attempts >= config.get("max_resume_attempts", 3):
                    continue
                if time.time() - state.get("last_browser_recovery", 0) < 60:
                    continue
                state.update(browser_recovery_attempts=attempts + 1, last_browser_recovery=time.time())
                try:
                    worker = WORKERS[wid]
                    context = contexts.get(wid)
                    if context is None or not is_cdp_reachable(worker["cdp_port"]):
                        if is_cdp_reachable(worker["cdp_port"]):
                            browser = playwright.chromium.connect_over_cdp(
                                f"http://127.0.0.1:{worker['cdp_port']}")
                            context = browser.contexts[0]
                            handles.append(browser)
                        else:
                            context = playwright.chromium.launch_persistent_context(
                                worker["profile_dir"], channel="chrome", headless=config["headless"],
                                args=[f"--remote-debugging-port={worker['cdp_port']}"])
                            handles.append(context)
                        contexts[wid] = context
                    page = next((p for p in context.pages if p.url == worker["notebook_url"]), None)
                    if page is None:
                        page = context.new_page()
                        page.goto(worker["notebook_url"], wait_until="domcontentloaded", timeout=60000)
                    else:
                        page.reload(wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(12000)
                    pages[wid] = page
                    has_collection = page.evaluate("""() => {
                        const cells = globalThis.colab?.global?.notebook?.cells || [];
                        return cells.some(c => c.getText?.().includes('run_search_games.py'));
                    }""")
                    if has_collection:
                        state.update(status="collection_starting", started_at=time.time())
                    else:
                        start_handover(page, wid)
                        state["status"] = "handover_requested"
                    state.pop("error", None)
                except Exception as error:
                    state["error"] = str(error)
            for wid, page in pages.items():
                state = states[wid]
                if state["status"] in ("blocked", "stopped", "completed"):
                    continue
                try:
                    text = "\n".join(page.locator(".xterm-rows").all_text_contents())
                    if state["status"] in ("handover_requested", "waiting_for_saved_shard"):
                        marker = f"ZQ_HANDOVER_READY_{wid}"
                        failed = f"ZQ_HANDOVER_FAILED_{wid}"
                        if f"ZQ_HANDOVER_WAITING_{wid}" in text:
                            state["status"] = "waiting_for_saved_shard"
                        state["terminal_preview"] = text[-2000:]
                        if failed in text:
                            state.update(status="blocked", error=text[-2500:])
                        elif marker in text:
                            old = get_notebook_dom_state(page, WORKERS[wid]["target_keywords"])
                            if not old["running"] and not old["pending"]:
                                if not trigger_cell_execution(page, profiles[wid], BOOTLOADER_TEMPLATE,
                                                              WORKERS[wid]["target_keywords"], force=True):
                                    raise RuntimeError("The notebook did not accept the collection cell.")
                                state["status"] = "collection_starting"
                                state["started_at"] = time.time()
                                print(json.dumps(state), flush=True)
                    if state["status"] in ("collection_starting", "collecting"):
                        current = get_notebook_dom_state(page, profiles[wid]["target_keywords"])
                        if not current["kConnected"]:
                            connect_runtime_if_needed(page)
                        state["output_preview"] = current["outText"]
                        state["connected"] = bool(current["kConnected"])
                        if not current["running"] and not current["pending"]:
                            finished = re.search(r'"games":\s*(\d+)', current["outText"])
                            if finished and int(finished[1]) == 2 * PROFILE_CONFIG["pairs"]:
                                state["status"] = "completed"
                            elif time.time() - state.get("started_at", 0) > 120:
                                attempts = state.get("resume_attempts", 0)
                                elapsed = time.time() - state.get("last_resume_at", 0)
                                fatal = any(word in current["outText"] for word in (
                                    "does not match", "different run identity", "modified artifact",
                                    "cache is absent", "Auth Required", "requires verification"))
                                if (config.get("auto_resume", True) and not fatal and
                                        attempts < config.get("max_resume_attempts", 3) and
                                        elapsed >= config.get("retry_cooldown_seconds", 900)):
                                    connect_runtime_if_needed(page)
                                    page.wait_for_timeout(5000)
                                    confirmed = get_notebook_dom_state(page, profiles[wid]["target_keywords"])
                                    if not confirmed["running"] and not confirmed["pending"]:
                                        if trigger_cell_execution(page, profiles[wid], BOOTLOADER_TEMPLATE,
                                                                  profiles[wid]["target_keywords"], force=True):
                                            state.update(status="collection_starting", started_at=time.time(),
                                                         last_resume_at=time.time(), resume_attempts=attempts + 1)
                                elif fatal or attempts >= config.get("max_resume_attempts", 3):
                                    state.update(status="stopped", error=current["outText"])
                        elif "pending_games" in current["outText"] or "Game " in current["outText"]:
                            state["status"] = "collecting"
                            successes = re.findall(r'Game\s+([^:]+):\s+ok', current["outText"])
                            if successes and successes[-1] != state.get("last_successful_game"):
                                state.update(last_successful_game=successes[-1], resume_attempts=0)
                except Exception as error:
                    state.update(status="blocked", error=str(error))
            save_status()
            if all(s["status"] in ("stopped", "completed") or
                   (s["status"] == "blocked" and s.get("browser_recovery_attempts", 0) >=
                    config.get("max_resume_attempts", 3)) for s in states.values()):
                break
            time.sleep(config["check_interval_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"])
    parser.add_argument("--revision", default=CONFIG["revision"])
    parser.add_argument("--check-interval-seconds", type=int, default=CONFIG["check_interval_seconds"])
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=CONFIG["headless"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=CONFIG["auto_resume"])
    parser.add_argument("--retry-cooldown-seconds", type=int, default=CONFIG["retry_cooldown_seconds"])
    parser.add_argument("--max-resume-attempts", type=int, default=CONFIG["max_resume_attempts"])
    args = vars(parser.parse_args())
    run(args)


if __name__ == "__main__":
    main()
