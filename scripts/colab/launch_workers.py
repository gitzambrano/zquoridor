"""Launch and bootstrap Google Colab self-play workers headlessly via Playwright.

Updates notebook cells with the resilient self-cloning bootloader, connects
the runtime, and triggers execution while avoiding redundant restarts of active workers.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure scripts root is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import WORKERS, BOOTLOADER_TEMPLATE
from browser_utils import is_cdp_reachable, is_profile_in_use, dismiss_modals, connect_runtime_if_needed

CONFIG: Dict[str, Any] = {
    "worker_ids": [3, 4, 5],
    "skip_if_running": True,
    "headless": True,
    "page_timeout_ms": 45000,
    "load_delay_seconds": 8,
    "wait_after_run_seconds": 25,
}


def launch_worker(worker: Dict[str, Any], skip_if_running: bool, headless: bool, timeout_ms: int, delay_s: int, wait_run_s: int) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    name = worker["name"]
    profile = worker["profile_dir"]
    url = worker["notebook_url"]
    worker_id = worker["worker_id"]
    cdp_port = worker.get("cdp_port", 9000 + worker_id)
    cell_code = BOOTLOADER_TEMPLATE.format(worker_id=worker_id)

    result: Dict[str, Any] = {
        "worker_id": worker_id,
        "name": name,
        "account": worker["account"],
        "action_taken": "none",
        "running": False,
        "pending": False,
        "error": None,
    }

    cdp_active = is_cdp_reachable(cdp_port)
    in_use, proc_pid = is_profile_in_use(profile)

    if in_use and not cdp_active:
        msg = f"Profile directory is actively locked by process PID {proc_pid}. Skipping launch to prevent corruption."
        print(f"  [WARN] {msg}")
        result["action_taken"] = "locked_by_external_process"
        result["error"] = msg
        return result

    try:
        with sync_playwright() as p:
            browser_to_close = None
            if cdp_active:
                browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                browser_to_close = browser
            else:
                ctx = p.chromium.launch_persistent_context(
                    user_data_dir=profile,
                    headless=headless,
                    channel="chrome",
                    args=[
                        f"--remote-debugging-port={cdp_port}",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--remote-allow-origins=*",
                    ],
                )
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                browser_to_close = ctx

            try:
                page.goto(url, wait_until="commit", timeout=timeout_ms)
                time.sleep(delay_s)

                # Step 1: Connect runtime if disconnected
                connect_runtime_if_needed(page)
                time.sleep(4)

                # Step 2: Check current status and update target cell
                prep_res = page.evaluate("""(data) => {
                    const { newCode, skipIfRunning } = data;
                    const nb = typeof colab !== 'undefined' && colab.global ? colab.global.notebook : null;
                    const cells = nb && nb.cells ? nb.cells : [];
                    let targetCell = null;
                    for (let i = cells.length - 1; i >= 0; i--) {
                        const txt = cells[i].getText ? cells[i].getText() : '';
                        if (txt.includes('run_colab_worker.py') || txt.includes('selfplay_15m')) {
                            targetCell = cells[i];
                            break;
                        }
                    }
                    if (!targetCell && cells.length > 0) {
                        targetCell = cells[cells.length - 1];
                    }

                    if (!targetCell) {
                        return { error: 'No cells found in notebook' };
                    }

                    const isRunning = targetCell.isRunning ? targetCell.isRunning() : false;
                    const isPending = targetCell.isPending ? targetCell.isPending() : false;

                    if (skipIfRunning && (isRunning || isPending)) {
                        return { skipped: true, isRunning, isPending };
                    }

                    // Update cell model
                    if (targetCell.model && targetCell.model.setText) {
                        targetCell.model.setText(newCode);
                    }
                    if (targetCell.model && targetCell.model.removeOutputs) {
                        targetCell.model.removeOutputs();
                    }

                    // Update Monaco editor models
                    let monacoCount = 0;
                    if (typeof monaco !== 'undefined') {
                        for (const m of monaco.editor.getModels()) {
                            if (m.getValue().includes('run_colab_worker.py') || m.getValue().includes('selfplay_15m') || m.getValue().includes('zquoridor')) {
                                m.setValue(newCode);
                                monacoCount++;
                            }
                        }
                    }

                    // Trigger execution
                    let runClicked = false;
                    const elem = targetCell.getElement ? targetCell.getElement() : (targetCell.element_ || targetCell.dom_);
                    if (elem && elem.scrollIntoView) elem.scrollIntoView();
                    if (typeof targetCell.manualExecute === 'function') {
                        targetCell.manualExecute();
                        runClicked = true;
                    } else if (elem) {
                        const btn = elem.querySelector('colab-run-button');
                        if (btn) {
                            if (btn.shadowRoot) {
                                const inner = btn.shadowRoot.querySelector('button, [role="button"]');
                                if (inner) inner.click();
                                else btn.click();
                            } else {
                                btn.click();
                            }
                            runClicked = true;
                        }
                    }

                    return {
                        skipped: false,
                        runClicked,
                        monacoCount,
                        cellId: targetCell.getCellId ? targetCell.getCellId() : null
                    };
                }""", {"newCode": cell_code, "skipIfRunning": skip_if_running})

                if prep_res.get("skipped"):
                    result["action_taken"] = "already_running_skipped"
                    result["running"] = prep_res.get("isRunning", False)
                    result["pending"] = prep_res.get("isPending", False)
                    return result

                if prep_res.get("error"):
                    result["error"] = prep_res["error"]
                    return result

                result["action_taken"] = "bootloader_injected_and_run"
                time.sleep(3)

                # Step 3: Dismiss any warning modal ("Executar de qualquer maneira" / "Run anyway" / Drive / OK)
                dismiss_modals(page)

                # Step 4: Wait for execution to start
                time.sleep(wait_run_s)
                dismiss_modals(page)

                status_after = page.evaluate("""() => {
                    const nb = typeof colab !== 'undefined' && colab.global ? colab.global.notebook : null;
                    const cells = nb && nb.cells ? nb.cells : [];
                    let targetCell = null;
                    for (let i = cells.length - 1; i >= 0; i--) {
                        const txt = cells[i].getText ? cells[i].getText() : '';
                        if (txt.includes('run_colab_worker.py') || txt.includes('selfplay_15m')) {
                            targetCell = cells[i];
                            break;
                        }
                    }
                    if (!targetCell && cells.length > 0) targetCell = cells[cells.length - 1];
                    if (!targetCell) return { running: false, pending: false, out: '' };

                    const dom = targetCell.getElement ? targetCell.getElement() : (targetCell.element_ || targetCell.dom_);
                    const outDiv = dom ? dom.querySelector('.output, colab-output, .output-stream, .output_text') : null;
                    return {
                        running: targetCell.isRunning ? targetCell.isRunning() : false,
                        pending: targetCell.isPending ? targetCell.isPending() : false,
                        out: outDiv ? outDiv.innerText.slice(-400) : ''
                    };
                }""")

                result["running"] = status_after["running"]
                result["pending"] = status_after["pending"]
                result["output_preview"] = status_after["out"]
            finally:
                if not cdp_active and browser_to_close:
                    browser_to_close.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_launch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("ZQUORIDOR COLAB WORKER LAUNCHER")
    print("=" * 70)

    results = []
    for wid in cfg["worker_ids"]:
        if wid not in WORKERS:
            print(f"Unknown worker ID {wid}, skipping.")
            continue
        w = WORKERS[wid]
        print(f"Processing {w['name']} (Worker #{wid}, Account: {w['account']})...")
        res = launch_worker(
            w,
            cfg["skip_if_running"],
            cfg["headless"],
            cfg["page_timeout_ms"],
            cfg["load_delay_seconds"],
            cfg["wait_after_run_seconds"],
        )
        results.append(res)

        if res["error"]:
            print(f"  [ERROR] {res['error']}")
        elif res["action_taken"] == "already_running_skipped":
            print(f"  [SKIPPED] Worker is already actively running. Preserving execution.")
        else:
            status = "RUNNING" if res["running"] else ("PENDING VM" if res["pending"] else "IDLE")
            print(f"  [LAUNCHED] Status: {status}")
            if res.get("output_preview"):
                lines = [line.strip() for line in res["output_preview"].splitlines() if line.strip()]
                preview = " | ".join(lines[-2:]) if lines else "no lines"
                print(f"  Output: {preview[:100]}...")
        print("-" * 70)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Colab self-play workers via Playwright.")
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"], help="Worker IDs to launch")
    parser.add_argument("--force-restart", action="store_true", help="Do not skip workers that are already running")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--timeout-ms", type=int, default=CONFIG["page_timeout_ms"], help="Page navigation timeout in ms")
    parser.add_argument("--show-config", action="store_true", help="Display effective configuration and exit.")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["worker_ids"] = args.worker_ids
    if args.force_restart:
        cfg["skip_if_running"] = False
    if args.headed:
        cfg["headless"] = False
    cfg["page_timeout_ms"] = args.timeout_ms

    if args.show_config:
        print("Effective Configuration:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        return

    run_launch(cfg)


if __name__ == "__main__":
    main()
