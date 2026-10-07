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
from scripts.colab.browser_utils import get_notebook_dom_state, is_cdp_reachable, trigger_cell_execution
from tools.external.build_lock import build_lock

CONFIG = {
    "worker_ids": [1, 2, 3, 4, 5],
    "revision": PROFILE_CONFIG["revision"],
    "check_interval_seconds": 30,
    "headless": True,
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
    page.locator('md-text-button[command="show-terminal"]').click()
    terminal = page.locator('textarea[aria-label="Terminal input"]')
    terminal.wait_for(state="attached", timeout=60000)
    page.wait_for_function(
        "() => document.querySelector('.xterm-rows')?.textContent.includes('/content')",
        timeout=60000)
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
    pages = {}
    handles = []
    with build_lock(output / "controller_lock"), sync_playwright() as playwright:
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
                page.wait_for_timeout(5000)
                pages[wid] = page
                previous = get_notebook_dom_state(page, profiles[wid]["target_keywords"])
                has_collection = page.evaluate("""() => {
                    const cells = globalThis.colab?.global?.notebook?.cells || [];
                    return cells.some(c => c.getText?.().includes('run_search_games.py'));
                }""")
                if has_collection and (previous["running"] or previous["pending"]):
                    state["status"] = "collection_starting"
                    print(json.dumps(state), flush=True)
                    continue
                start_handover(page, wid)
                state["status"] = "waiting_for_saved_shard"
            except Exception as error:
                state.update(status="blocked", error=str(error))
            print(json.dumps(state), flush=True)
        while True:
            for wid, page in pages.items():
                state = states[wid]
                try:
                    text = "\n".join(page.locator(".xterm-rows").all_text_contents())
                    if state["status"] == "waiting_for_saved_shard":
                        marker = f"ZQ_HANDOVER_READY_{wid}"
                        failed = f"ZQ_HANDOVER_FAILED_{wid}"
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
                        state["output_preview"] = current["outText"]
                        if not current["running"] and not current["pending"]:
                            finished = re.search(r'"games":\s*(\d+)', current["outText"])
                            if finished and int(finished[1]) == 2 * PROFILE_CONFIG["pairs"]:
                                state["status"] = "completed"
                            elif time.time() - state.get("started_at", 0) > 120:
                                state.update(status="stopped", error=current["outText"])
                        elif "pending_games" in current["outText"] or "Game " in current["outText"]:
                            state["status"] = "collecting"
                except Exception as error:
                    state.update(status="blocked", error=str(error))
            temporary = output / "status.tmp"
            temporary.write_text(json.dumps(states, indent=2), encoding="utf-8")
            temporary.replace(output / "status.json")
            if all(s["status"] in ("blocked", "stopped", "completed") for s in states.values()):
                break
            time.sleep(config["check_interval_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"])
    parser.add_argument("--revision", default=CONFIG["revision"])
    parser.add_argument("--check-interval-seconds", type=int, default=CONFIG["check_interval_seconds"])
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=CONFIG["headless"])
    parser.add_argument("--output", default=CONFIG["output"])
    args = vars(parser.parse_args())
    run(args)


if __name__ == "__main__":
    main()
