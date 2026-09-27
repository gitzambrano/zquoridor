"""Inspect Google Colab worker status headlessly via Playwright.

Checks runtime connection state, hardware accelerator type, running cells,
and recent self-play progress for all configured workers.
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

from config import WORKERS

CONFIG: Dict[str, Any] = {
    "worker_ids": [3, 4, 5],
    "headless": True,
    "page_timeout_ms": 45000,
    "load_delay_seconds": 8,
}


def inspect_worker(worker: Dict[str, Any], headless: bool, timeout_ms: int, delay_s: int) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    name = worker["name"]
    profile = worker["profile_dir"]
    url = worker["notebook_url"]
    worker_id = worker["worker_id"]

    result: Dict[str, Any] = {
        "worker_id": worker_id,
        "name": name,
        "account": worker["account"],
        "connected": False,
        "running": False,
        "pending": False,
        "status_text": "unknown",
        "output_preview": "",
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

                info = page.evaluate("""() => {
                    const btn = document.querySelector('colab-connect-button');
                    const sr = btn ? btn.shadowRoot : null;
                    const connBtn = sr ? sr.querySelector('#connect') : null;
                    const statusText = connBtn ? connBtn.innerText.trim() : (btn ? btn.innerText.trim() : 'no connect button');

                    const k = typeof colab !== 'undefined' && colab.global && colab.global.notebook ? colab.global.notebook.kernel : null;
                    const kConnected = k && k.isConnected ? k.isConnected() : false;
                    const kState = k ? k.state : 'no kernel';

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

                    let running = false;
                    let pending = false;
                    let outText = '';
                    if (targetCell) {
                        running = targetCell.isRunning ? targetCell.isRunning() : false;
                        pending = targetCell.isPending ? targetCell.isPending() : false;
                        const dom = targetCell.element_ || targetCell.dom_;
                        const outDiv = dom ? dom.querySelector('.output, colab-output') : null;
                        outText = outDiv ? outDiv.innerText.slice(-600) : '';
                    }

                    const bodyText = document.body ? document.body.innerText : '';
                    const quotaWarning = bodyText.includes('não tem unidades de computação') || bodyText.includes('não tem unidades de computa');

                    return {
                        statusText,
                        kConnected,
                        kState,
                        running,
                        pending,
                        outText,
                        quotaWarning,
                        cellCount: cells.length
                    };
                }""")

                result["connected"] = info["kConnected"] or ("RAM" in info["statusText"])
                result["status_text"] = info["statusText"]
                result["running"] = info["running"]
                result["pending"] = info["pending"]
                result["quota_warning"] = info["quotaWarning"]
                result["output_preview"] = info["outText"]
            finally:
                ctx.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_inspect(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("ZQUORIDOR COLAB WORKER INSPECTION")
    print("=" * 70)

    results = []
    for wid in cfg["worker_ids"]:
        if wid not in WORKERS:
            print(f"Unknown worker ID {wid}, skipping.")
            continue
        w = WORKERS[wid]
        print(f"Inspecting {w['name']} (Worker #{wid}, Account: {w['account']})...")
        res = inspect_worker(w, cfg["headless"], cfg["page_timeout_ms"], cfg["load_delay_seconds"])
        results.append(res)

        if res["error"]:
            print(f"  [ERROR] {res['error']}")
        else:
            conn_status = "CONNECTED" if res["connected"] else "CONNECTING/DISCONNECTED"
            run_status = "ACTIVE RUNNING" if res["running"] else ("PENDING" if res["pending"] else "IDLE")
            quota_status = " [!] QUOTA WARNING (GPU)" if res.get("quota_warning") else ""
            print(f"  Connection: {conn_status} ({res['status_text']}){quota_status}")
            print(f"  Execution:  {run_status}")
            if res["output_preview"]:
                lines = [line.strip() for line in res["output_preview"].splitlines() if line.strip()]
                preview = " | ".join(lines[-3:]) if lines else "no lines"
                print(f"  Output:     {preview[:120]}...")
        print("-" * 70)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Colab workers status via Playwright.")
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"], help="Worker IDs to inspect")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--timeout-ms", type=int, default=CONFIG["page_timeout_ms"], help="Page navigation timeout in ms")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["worker_ids"] = args.worker_ids
    if args.headed:
        cfg["headless"] = False
    cfg["page_timeout_ms"] = args.timeout_ms

    run_inspect(cfg)


if __name__ == "__main__":
    main()
