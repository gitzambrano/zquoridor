"""Continuous watchdog and keep-alive monitor for Google Colab self-play workers.

Maintains active persistent browser sessions with periodic micro-interactions to
prevent Colab idle disconnects. Detects session disconnects, automatically reconnects,
re-triggers the resilient bootloader cell, and saves periodic status screenshots.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Ensure scripts root is in path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import WORKERS, BOOTLOADER_TEMPLATE
from browser_utils import (
    is_cdp_reachable,
    is_profile_in_use,
    dismiss_modals,
    connect_runtime_if_needed,
    trigger_cell_execution,
)

CONFIG: Dict[str, Any] = {
    "worker_ids": [3, 4, 5],
    "check_interval_seconds": 60,
    "screenshot_interval_cycles": 10,
    "auto_reconnect": True,
    "headless": True,
    "page_timeout_ms": 60000,
    "artifacts_dir": str(REPO_ROOT / "artifacts" / "colab"),
}


def create_artifacts_dir(dir_path: str) -> Path:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_watchdog(cfg: Dict[str, Any]) -> None:
    from playwright.sync_api import sync_playwright

    artifacts_path = create_artifacts_dir(cfg["artifacts_dir"])
    worker_ids = [wid for wid in cfg["worker_ids"] if wid in WORKERS]
    if not worker_ids:
        print("[WATCHDOG] No valid worker IDs configured.")
        return

    print("=" * 70)
    print("ZQUORIDOR COLAB ACTIVE WATCHDOG & KEEP-ALIVE")
    print("=" * 70)
    print(f"Monitoring workers: {worker_ids}")
    print(f"Interval: {cfg['check_interval_seconds']}s | Auto-reconnect: {cfg['auto_reconnect']}")
    print(f"Artifacts output: {artifacts_path}")
    print("=" * 70)

    with sync_playwright() as p:
        sessions: Dict[int, Dict[str, Any]] = {}

        # 1. Initialize persistent contexts or attach via CDP
        for wid in worker_ids:
            w = WORKERS[wid]
            cdp_port = w.get("cdp_port", 9000 + wid)
            print(f"Initializing session for {w['name']} (Worker #{wid})...")

            cdp_active = is_cdp_reachable(cdp_port)
            in_use, proc_pid = is_profile_in_use(w["profile_dir"])

            if in_use and not cdp_active:
                print(f"  [WARN] Profile for {w['name']} is locked by external process PID {proc_pid}.")
                print(f"  Skipping direct control. Active watchdog for other workers will continue.")
                continue

            try:
                if cdp_active:
                    print(f"  Attaching over live CDP on port {cdp_port}...")
                    browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
                    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                    page = ctx.pages[0] if ctx.pages else ctx.new_page()
                    is_cdp = True
                else:
                    ctx = p.chromium.launch_persistent_context(
                        user_data_dir=w["profile_dir"],
                        headless=cfg["headless"],
                        channel="chrome",
                        args=[
                            f"--remote-debugging-port={cdp_port}",
                            "--no-sandbox",
                            "--disable-gpu",
                            "--remote-allow-origins=*",
                        ],
                    )
                    page = ctx.pages[0] if ctx.pages else ctx.new_page()
                    page.goto(w["notebook_url"], wait_until="commit", timeout=cfg["page_timeout_ms"])
                    is_cdp = False

                sessions[wid] = {
                    "worker": w,
                    "ctx": ctx,
                    "page": page,
                    "is_cdp": is_cdp,
                    "reconnect_count": 0,
                    "last_progress": "Initializing...",
                }
                print(f"  Connected to {w['name']}.")
            except Exception as exc:
                print(f"  [ERROR] Failed to initialize {w['name']}: {exc}")

        if not sessions:
            print("[WATCHDOG] No sessions could be initialized. Exiting.")
            return

        time.sleep(10)
        cycle = 0

        # 2. Continuous monitoring and keep-alive loop
        try:
            while True:
                cycle += 1
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n--- [Cycle {cycle:04d}] {now_str} ---")

                for wid, sess in sessions.items():
                    w = sess["worker"]
                    page = sess["page"]

                    try:
                        dismiss_modals(page)

                        # Extract state from Colab DOM
                        state = page.evaluate("""() => {
                            const btn = document.querySelector('colab-connect-button');
                            const sr = btn ? btn.shadowRoot : null;
                            const connBtn = sr ? sr.querySelector('#connect') : null;
                            const statusText = connBtn ? connBtn.innerText.trim() : (btn ? btn.innerText.trim() : 'no button');

                            const k = typeof colab !== 'undefined' && colab.global && colab.global.notebook ? colab.global.notebook.kernel : null;
                            const kConnected = k && k.isConnected ? k.isConnected() : false;

                            const isRunningBtn = Array.from(document.querySelectorAll(
                                'colab-run-button[title*="Interromper"], colab-run-button[aria-label*="Interromper"], colab-run-button.running'
                            ));
                            const nb = typeof colab !== 'undefined' && colab.global ? colab.global.notebook : null;
                            let running = isRunningBtn.length > 0;
                            if (nb && typeof nb.isExecuting === 'function') {
                                running = running || nb.isExecuting();
                            }

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

                            let pending = false;
                            let outText = '';
                            if (targetCell) {
                                running = running || (targetCell.isRunning ? targetCell.isRunning() : false);
                                pending = targetCell.isPending ? targetCell.isPending() : false;
                                const dom = targetCell.getElement ? targetCell.getElement() : (targetCell.element_ || targetCell.dom_);
                                const outDiv = dom ? dom.querySelector('.output, colab-output, .output-stream, .output_text') : null;
                                outText = outDiv ? outDiv.innerText.slice(-1200) : '';
                            }

                            return {
                                statusText,
                                kConnected,
                                running,
                                pending,
                                outText
                            };
                        }""")

                        # Extract clean progress line
                        lines = [line.strip() for line in state["outText"].splitlines() if line.strip()]
                        progress_lines = [l for l in lines if any(kw in l for kw in ["progresso:", "Launching Shard", "ok:", "chunk", "selfplay"])]
                        current_progress = progress_lines[-1] if progress_lines else (lines[-1] if lines else "No output")
                        sess["last_progress"] = current_progress

                        # Keep-alive micro interaction: subtle mouse movement prevents idle timeout
                        page.mouse.move(60 + (cycle % 40), 60 + (cycle % 40))

                        run_tag = "[RUNNING]" if state["running"] else ("[PENDING]" if state["pending"] else "[IDLE]")
                        print(f"[{w['name']}] {run_tag} (VM: {state['statusText']}) -> {current_progress[:90]}")

                        # 3. Auto-reconnect or bootstrap if idle and not running
                        if cfg["auto_reconnect"] and not state["running"] and not state["pending"]:
                            if "Conectando" not in state["statusText"]:
                                if "Conectar" in state["statusText"] or "Connect" in state["statusText"]:
                                    print(f"[{w['name']}] [DISCONNECTED] Connecting VM...")
                                    sess["reconnect_count"] += 1
                                    connect_runtime_if_needed(page)
                                    time.sleep(12)

                                print(f"[{w['name']}] Triggering cell execution...")
                                ok = trigger_cell_execution(page, wid, BOOTLOADER_TEMPLATE)
                                if ok:
                                    print(f"[{w['name']}] Triggered execution successfully.")
                                else:
                                    print(f"[{w['name']}] Trigger attempt complete (will re-verify next cycle).")

                        # 4. Periodic health screenshot
                        if cycle % cfg["screenshot_interval_cycles"] == 0:
                            shot_file = artifacts_path / f"{w['name'].lower().replace(' ', '_')}_watchdog.png"
                            page.screenshot(path=str(shot_file))

                    except Exception as err:
                        print(f"[{w['name']}] [ERROR in cycle {cycle}]: {err}")

                sys.stdout.flush()
                time.sleep(cfg["check_interval_seconds"])

        except KeyboardInterrupt:
            print("\n[WATCHDOG] Interrupted by user. Closing sessions cleanly...")
        finally:
            for wid, sess in sessions.items():
                try:
                    if sess.get("is_cdp"):
                        sess["ctx"].browser.disconnect()
                    else:
                        sess["ctx"].close()
                    print(f"Closed session for Worker #{wid}.")
                except Exception:
                    pass
            print("[WATCHDOG] All sessions closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Active watchdog and keep-alive for Colab workers.")
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"], help="Worker IDs to monitor")
    parser.add_argument("--interval", type=int, default=CONFIG["check_interval_seconds"], help="Seconds between health checks")
    parser.add_argument("--no-auto-reconnect", action="store_true", help="Disable automatic reconnect upon disconnect")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--show-config", action="store_true", help="Display effective configuration and exit.")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["worker_ids"] = args.worker_ids
    cfg["check_interval_seconds"] = args.interval
    if args.no_auto_reconnect:
        cfg["auto_reconnect"] = False
    if args.headed:
        cfg["headless"] = False

    if args.show_config:
        print("Effective Configuration:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        return

    run_watchdog(cfg)


if __name__ == "__main__":
    main()
