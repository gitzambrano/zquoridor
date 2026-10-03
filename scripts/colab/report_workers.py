"""Comprehensive status inspection, metric parser, and reporting tool for Colab workers.

Connects headlessly via Playwright, gathers live runtime and cell execution metrics,
captures screenshots into artifacts/colab/, and outputs both terminal and markdown reports.
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

from config import WORKERS
from browser_utils import is_cdp_reachable, is_profile_in_use, get_notebook_dom_state, dismiss_modals

CONFIG: Dict[str, Any] = {
    "worker_ids": [1, 2, 3, 4, 5, 6, 7],
    "headless": True,
    "page_timeout_ms": 60000,
    "load_delay_seconds": 8,
    "artifacts_dir": str(REPO_ROOT / "artifacts" / "colab"),
}


def create_artifacts_dir(dir_path: str) -> Path:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def report_single_worker(worker: Dict[str, Any], headless: bool, timeout_ms: int, delay_s: int, artifacts_dir: Path) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    wid = worker["worker_id"]
    name = worker["name"]
    profile = worker["profile_dir"]
    url = worker["notebook_url"]
    account = worker["account"]
    cdp_port = worker.get("cdp_port", 9000 + wid)

    record: Dict[str, Any] = {
        "worker_id": wid,
        "name": name,
        "account": account,
        "url": url,
        "connected": False,
        "running": False,
        "pending": False,
        "status_text": "Unknown",
        "current_shard": "Unknown",
        "games_completed": 0,
        "total_games": 250,
        "positions_generated": 0,
        "speed_nps": "N/A",
        "speed_pps": "N/A",
        "screenshot_path": None,
        "error": None,
    }

    cdp_active = is_cdp_reachable(cdp_port)
    in_use, proc_pid = is_profile_in_use(profile)

    if in_use and not cdp_active:
        record["status_text"] = f"Managed by external process (PID {proc_pid})"
        record["connected"] = True
        record["running"] = True

        existing_ss = artifacts_dir / f"colab_{wid}_report.png"
        if existing_ss.exists():
            record["screenshot_path"] = str(existing_ss)
        return record

    try:
        with sync_playwright() as p:
            browser_to_close = None
            if cdp_active:
                browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                browser_to_close = browser
                if "colab.research.google.com" not in (page.url or ""):
                    page.goto(url, wait_until="commit", timeout=timeout_ms)
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
                page.goto(url, wait_until="commit", timeout=timeout_ms)
                browser_to_close = ctx

            try:
                time.sleep(delay_s)
                dismiss_modals(page)

                info = page.evaluate("""() => {
                const curUrl = window.location.href || '';
                const bodyText = document.body ? (document.body.innerText || '') : '';
                if (curUrl.includes('accounts.google.com') || curUrl.includes('signin') || bodyText.includes('Confirme que') || bodyText.includes('Confirm it')) {
                    return {
                        statusText: 'Auth Required (Google Login)',
                        kConnected: false,
                        running: false,
                        pending: false,
                        outText: 'Google account requires verification (Confirm it is you).'
                    };
                }

                const btn = document.querySelector('colab-connect-button');
                const sr = btn ? btn.shadowRoot : null;
                const connBtn = sr ? sr.querySelector('#connect') : null;
                const statusText = connBtn ? connBtn.innerText.trim() : (btn ? btn.innerText.trim() : 'no button');

                    const k = typeof colab !== 'undefined' && colab.global && colab.global.notebook ? colab.global.notebook.kernel : null;
                    const kConnected = k && k.isConnected ? k.isConnected() : false;

                    const cells = typeof colab !== 'undefined' && colab.global && colab.global.notebook && colab.global.notebook.cells ? colab.global.notebook.cells : [];
                    let targetCell = null;
                    for (let i = cells.length - 1; i >= 0; i--) {
                        const txt = cells[i].getText();
                        if (txt.includes('run_colab_worker.py') || txt.includes('selfplay_targeted_weakness') || txt.includes('selfplay_15m')) {
                            targetCell = cells[i];
                            break;
                        }
                    }
                    if (!targetCell && cells.length > 0) targetCell = cells[cells.length - 1];

                    let running = false;
                    let pending = false;
                    let outText = '';
                    if (targetCell) {
                        running = targetCell.isRunning ? targetCell.isRunning() : false;
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

                record["connected"] = info["kConnected"] or ("RAM" in info["statusText"])
                record["status_text"] = info["statusText"]
                record["running"] = info["running"]
                record["pending"] = info["pending"]

                # Parse metrics from output text
                out = info["outText"]
                lines = [l.strip() for l in out.splitlines() if l.strip()]

                # Find shard
                for l in reversed(lines):
                    if "c" in l and "_shard_" in l and ".bin" in l:
                        parts = l.split("/")
                        for p_part in parts:
                            if "_shard_" in p_part and ".bin" in p_part:
                                record["current_shard"] = p_part.split()[0].replace(".bin", "")
                                break
                        if record["current_shard"] != "Unknown":
                            break
                    elif "Launching Shard" in l:
                        record["current_shard"] = l.split("Launching Shard")[-1].strip().split()[0]
                        break

                # Find progress: e.g. "progresso: 45/250 partidas | 2387 posicoes"
                for l in reversed(lines):
                    if "progresso:" in l:
                        try:
                            # Parse "progresso: 45/250 partidas | 2387 posicoes"
                            after = l.split("progresso:")[1].strip()
                            parts = [p_part.strip() for p_part in after.split("|")]
                            games_part = parts[0].split()[0]  # "45/250"
                            g_done, g_total = games_part.split("/")
                            record["games_completed"] = int(g_done)
                            record["total_games"] = int(g_total)

                            if len(parts) > 1 and "posicoes" in parts[1]:
                                record["positions_generated"] = int(parts[1].split()[0])
                            break
                        except Exception:
                            pass

                # Find speed from completion line: "ok: 705.1 s | ... | 3235 nos/s | 18.7 pos/s"
                for l in reversed(lines):
                    if "nos/s" in l and "pos/s" in l:
                        parts = [p_part.strip() for p_part in l.split("|")]
                        for p_part in parts:
                            if "nos/s" in p_part:
                                record["speed_nps"] = p_part
                            if "pos/s" in p_part:
                                record["speed_pps"] = p_part
                        break

                # Save screenshot
                shot_filename = f"{name.lower().replace(' ', '_')}_report.png"
                shot_path = artifacts_dir / shot_filename
                page.screenshot(path=str(shot_path))
                record["screenshot_path"] = str(shot_path)

            finally:
                if not cdp_active and browser_to_close:
                    browser_to_close.close()
    except Exception as exc:
        record["error"] = str(exc)

    return record


def generate_markdown_report(records: List[Dict[str, Any]], artifacts_dir: Path) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [
        f"# Colab Self-Play Worker Report",
        f"\n*Generated at: {now_str}*\n",
        "| Worker | Account | VM Status | Execution | Shard | Progress (Games) | Positions |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in records:
        if r["error"]:
            md.append(f"| **{r['name']}** | `{r['account']}` | Error | Error | N/A | Error: {r['error'][:30]}... | N/A |")
        else:
            vm_st = "[CONNECTED]" if r["connected"] else f"[DISCONNECTED: {r['status_text']}]"
            ex_st = "[ACTIVE RUNNING]" if r["running"] else ("[PENDING]" if r["pending"] else "[IDLE]")
            prog = f"{r['games_completed']}/{r['total_games']}"
            pos = f"{r['positions_generated']:,}" if r['positions_generated'] else "0"
            md.append(f"| **{r['name']}** | `{r['account']}` | {vm_st} | {ex_st} | `{r['current_shard']}` | {prog} | {pos} |")

    md_content = "\n".join(md) + "\n"
    out_file = artifacts_dir / "report.md"
    out_file.write_text(md_content, encoding="utf-8")
    return md_content


def run_report(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts_path = create_artifacts_dir(cfg["artifacts_dir"])
    print("=" * 70)
    print("ZQUORIDOR COLAB WORKER AUDIT & REPORT")
    print("=" * 70)

    records = []
    for wid in cfg["worker_ids"]:
        if wid not in WORKERS:
            continue
        w = WORKERS[wid]
        print(f"Auditing {w['name']} (Worker #{wid}, {w['account']})...")
        rec = report_single_worker(w, cfg["headless"], cfg["page_timeout_ms"], cfg["load_delay_seconds"], artifacts_path)
        records.append(rec)

        if rec["error"]:
            print(f"  [ERROR] {rec['error']}")
        else:
            conn = "CONNECTED" if rec["connected"] else f"DISCONNECTED ({rec['status_text']})"
            run_st = "ACTIVE RUNNING" if rec["running"] else ("PENDING" if rec["pending"] else "IDLE")
            print(f"  VM Status:  {conn}")
            print(f"  Execution:  {run_st}")
            print(f"  Shard:      {rec['current_shard']}")
            print(f"  Progress:   {rec['games_completed']}/{rec['total_games']} games ({rec['positions_generated']} positions)")
            if rec["screenshot_path"]:
                print(f"  Screenshot: {rec['screenshot_path']}")
        print("-" * 70)

    # Output Markdown report
    md = generate_markdown_report(records, artifacts_path)
    print("\n" + md)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and report Colab workers.")
    parser.add_argument("--worker-ids", type=int, nargs="+", default=CONFIG["worker_ids"], help="Worker IDs to report")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--show-config", action="store_true", help="Display effective configuration and exit.")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["worker_ids"] = args.worker_ids
    if args.headed:
        cfg["headless"] = False

    if args.show_config:
        print("Effective Configuration:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        return

    run_report(cfg)


if __name__ == "__main__":
    main()
