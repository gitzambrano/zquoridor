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

    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=profile,
                headless=headless,
                channel="chrome",
                args=["--no-sandbox"],
            )
            try:
                page = ctx.new_page()
                page.goto(url, timeout=timeout_ms)
                time.sleep(delay_s)

                # Step 1: Connect runtime if disconnected
                page.evaluate("""() => {
                    const btn = document.querySelector('colab-connect-button');
                    if (btn && btn.shadowRoot) {
                        const conn = btn.shadowRoot.querySelector('#connect');
                        if (conn && (conn.innerText.includes('Conectar') || conn.innerText.includes('Reconectar'))) {
                            conn.click();
                        }
                    }
                }""")
                time.sleep(4)

                # Step 2: Check current status and update target cell
                prep_res = page.evaluate("""(data) => {
                    const { newCode, skipIfRunning } = data;
                    const cells = typeof colab !== 'undefined' && colab.global && colab.global.notebook && colab.global.notebook.cells ? colab.global.notebook.cells : [];
                    let targetCell = null;
                    for (let i = cells.length - 1; i >= 0; i--) {
                        const txt = cells[i].getText();
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
                    targetCell.model.setText(newCode);
                    if (targetCell.model.removeOutputs) {
                        targetCell.model.removeOutputs();
                    }

                    // Update Monaco editor models
                    let monacoCount = 0;
                    if (typeof monaco !== 'undefined') {
                        for (const m of monaco.editor.getModels()) {
                            if (m.getValue().includes('run_colab_worker.py') || m.getValue().includes('selfplay_15m')) {
                                m.setValue(newCode);
                                monacoCount++;
                            }
                        }
                    }

                    // Trigger execution
                    let runClicked = false;
                    if (targetCell.runButton) {
                        targetCell.runButton.click();
                        runClicked = true;
                    } else if (targetCell.manualExecute) {
                        targetCell.manualExecute();
                        runClicked = true;
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

                # Step 3: Dismiss any warning modal ("Executar de qualquer maneira" / "Run anyway")
                page.evaluate("""() => {
                    const btns = Array.from(document.querySelectorAll('paper-button, mwc-button, button'));
                    for (const b of btns) {
                        const t = b.innerText || '';
                        if (t.includes('Executar de qualquer maneira') || t.includes('Run anyway')) {
                            b.click();
                            return;
                        }
                    }
                }""")

                # Step 4: Wait for execution to start
                time.sleep(wait_run_s)

                status_after = page.evaluate("""() => {
                    const cells = typeof colab !== 'undefined' && colab.global && colab.global.notebook && colab.global.notebook.cells ? colab.global.notebook.cells : [];
                    let targetCell = null;
                    for (let i = cells.length - 1; i >= 0; i--) {
                        const txt = cells[i].getText();
                        if (txt.includes('run_colab_worker.py') || txt.includes('selfplay_15m')) {
                            targetCell = cells[i];
                            break;
                        }
                    }
                    if (!targetCell && cells.length > 0) targetCell = cells[cells.length - 1];
                    if (!targetCell) return { running: false, pending: false, out: '' };

                    const dom = targetCell.element_ || targetCell.dom_;
                    const outDiv = dom ? dom.querySelector('.output, colab-output') : null;
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
                ctx.close()
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
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["worker_ids"] = args.worker_ids
    if args.force_restart:
        cfg["skip_if_running"] = False
    if args.headed:
        cfg["headless"] = False
    cfg["page_timeout_ms"] = args.timeout_ms

    run_launch(cfg)


if __name__ == "__main__":
    main()
