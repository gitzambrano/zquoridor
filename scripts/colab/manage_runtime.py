"""Administrative runtime management for Google Colab notebooks via Playwright.

Provides tools to switch hardware accelerators (e.g. from GPU to CPU to clear quota
exhaustion), reset hanging environments, or terminate stale ghost sessions.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure scripts root is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import WORKERS

CONFIG: Dict[str, Any] = {
    "worker_ids": [3, 4, 5],
    "action": "switch-cpu",  # Options: 'switch-cpu', 'reset', 'terminate-active'
    "headless": True,
    "page_timeout_ms": 45000,
    "load_delay_seconds": 8,
}


def manage_worker(worker: Dict[str, Any], action: str, headless: bool, timeout_ms: int, delay_s: int) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    name = worker["name"]
    profile = worker["profile_dir"]
    url = worker["notebook_url"]
    wid = worker["worker_id"]

    result: Dict[str, Any] = {
        "worker_id": wid,
        "name": name,
        "action": action,
        "success": False,
        "details": "",
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

                if action == "switch-cpu":
                    # 1. Open Runtime menu
                    page.locator('div[id="runtime-menu-button"]').click()
                    time.sleep(1)
                    page.locator('text="Alterar o tipo de ambiente de execução"').click()
                    time.sleep(2)

                    # 2. Select CPU option
                    page.get_by_text("CPU", exact=True).click()
                    time.sleep(1)

                    # 3. Save
                    page.locator('md-text-button').filter(has_text="Salvar").click()
                    time.sleep(5)
                    result["success"] = True
                    result["details"] = "Hardware accelerator switched to CPU (Standard)."

                elif action == "reset":
                    # Open menu and disconnect/delete runtime
                    page.locator('div[id="runtime-menu-button"]').click()
                    time.sleep(1)
                    page.locator('text="Desconectar e excluir ambiente de execução"').click()
                    time.sleep(2)

                    # Confirm if dialog appears
                    page.evaluate("""() => {
                        const btns = Array.from(document.querySelectorAll('mwc-button, paper-button, button'));
                        const ok = btns.find(b => b.innerText && (b.innerText.trim() === 'Sim' || b.innerText.trim() === 'Desconectar'));
                        if (ok) ok.click();
                    }""")
                    time.sleep(3)
                    result["success"] = True
                    result["details"] = "Runtime disconnected and deleted."

                elif action == "terminate-active":
                    # Open Gerenciar sessões
                    page.locator('div[id="runtime-menu-button"]').click()
                    time.sleep(1)
                    page.locator('text="Gerenciar sessões"').click()
                    time.sleep(2)

                    # Click terminate trash icon
                    trash_res = page.evaluate("""() => {
                        const dialog = Array.from(document.querySelectorAll('mwc-dialog, paper-dialog, dialog')).find(d => d.innerText && d.innerText.includes('Sessões ativas'));
                        if (!dialog) return 'No dialog found';
                        const btns = Array.from(dialog.querySelectorAll('mwc-icon-button, paper-icon-button, button'));
                        for (const b of btns) {
                            if (b.innerText.includes('delete') || (b.title && b.title.includes('Encerrar'))) {
                                b.click();
                                return 'Clicked terminate button';
                            }
                        }
                        if (btns.length > 0) {
                            btns[0].click();
                            return 'Clicked first button';
                        }
                        return 'No buttons in dialog';
                    }""")
                    time.sleep(2)

                    # Confirm
                    page.evaluate("""() => {
                        const btns = Array.from(document.querySelectorAll('mwc-button, paper-button, button'));
                        const ok = btns.find(b => b.innerText && (b.innerText.trim() === 'Encerrar' || b.innerText.trim() === 'Sim'));
                        if (ok) ok.click();
                    }""")
                    time.sleep(2)

                    # Close modal
                    page.locator('text="Fechar"').click()
                    result["success"] = True
                    result["details"] = f"Manage sessions result: {trash_res}"

                else:
                    result["error"] = f"Unknown action: {action}"

            finally:
                ctx.close()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Google Colab runtimes.")
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"], help="Worker IDs to manage")
    parser.add_argument("--action", choices=["switch-cpu", "reset", "terminate-active"], default=CONFIG["action"], help="Action to execute")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--timeout-ms", type=int, default=CONFIG["page_timeout_ms"], help="Navigation timeout in ms")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["worker_ids"] = args.worker_ids
    cfg["action"] = args.action
    if args.headed:
        cfg["headless"] = False
    cfg["page_timeout_ms"] = args.timeout_ms

    print("=" * 70)
    print(f"COLAB RUNTIME MANAGEMENT: Action '{cfg['action']}'")
    print("=" * 70)

    for wid in cfg["worker_ids"]:
        if wid not in WORKERS:
            print(f"Unknown worker ID {wid}, skipping.")
            continue
        w = WORKERS[wid]
        print(f"Applying '{cfg['action']}' on {w['name']} (Worker #{wid})...")
        res = manage_worker(w, cfg["action"], cfg["headless"], cfg["page_timeout_ms"], cfg["load_delay_seconds"])
        if res["error"]:
            print(f"  [ERROR] {res['error']}")
        else:
            print(f"  [SUCCESS] {res['details']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
