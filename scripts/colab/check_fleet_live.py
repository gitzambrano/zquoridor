"""Check live status of all 5 Colabs and ensure execution is active."""
from playwright.sync_api import sync_playwright
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.search_games_profile import build_worker_profile
from scripts.colab.browser_utils import dismiss_modals, connect_runtime_if_needed, is_cdp_reachable


def inspect_and_ensure_worker(wid: int) -> dict:
    profile = build_worker_profile(wid)
    port = profile['cdp_port']
    with sync_playwright() as p:
        if is_cdp_reachable(port):
            b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            ctx = b.contexts[0]
            owns_browser = False
        else:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=profile['profile_dir'],
                headless=True,
                channel='chrome',
                args=[f'--remote-debugging-port={port}', '--remote-allow-origins=*'],
            )
            owns_browser = True

        page = None
        for pg in ctx.pages:
            if "colab.research.google.com" in pg.url:
                page = pg
                break
        if not page:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(profile['notebook_url'], wait_until='domcontentloaded', timeout=60000)

        page.wait_for_timeout(6000)
        connect_runtime_if_needed(page)
        dismiss_modals(page)

        # Inspect cell 0 and status
        status = page.evaluate('''() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb ? nb.cells[0] : null;
            const elem = cell ? (cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_)) : null;
            let out = '';
            if (elem) {
                const d = elem.querySelector('.output, colab-output, .output-stream, .output_text');
                out = d ? d.innerText.slice(-800) : '';
            }

            const btn = elem ? elem.querySelector('colab-run-button') : null;
            const sr = btn ? btn.shadowRoot : null;
            const wrap = sr ? sr.querySelector('.cell-execution') : null;

            return {
                cellIsRunning: cell ? cell.isRunning() : false,
                cellIsPending: cell ? cell.isPending() : false,
                shadowClass: wrap ? wrap.className : '',
                outText: out,
                textSnippet: cell ? cell.getText().slice(-120).replace(/\\n/g, ' ') : ''
            };
        }''')

        # If not running and not pending, click run!
        triggered = False
        if not status.get("cellIsRunning") and not status.get("cellIsPending"):
            run_res = page.evaluate('''() => {
                const btn = document.querySelector('colab-run-button');
                if (btn && btn.shadowRoot) {
                    const inner = btn.shadowRoot.querySelector('#run-button, button');
                    if (inner) {
                        inner.click();
                        return true;
                    }
                }
                return false;
            }''')
            if run_res:
                triggered = True
                time.sleep(4)
                dismiss_modals(page)
                # Recheck
                status = page.evaluate('''() => {
                    const nb = globalThis.colab?.global?.notebook;
                    const cell = nb ? nb.cells[0] : null;
                    const elem = cell ? (cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_)) : null;
                    let out = '';
                    if (elem) {
                        const d = elem.querySelector('.output, colab-output, .output-stream, .output_text');
                        out = d ? d.innerText.slice(-800) : '';
                    }
                    return {
                        cellIsRunning: cell ? cell.isRunning() : false,
                        cellIsPending: cell ? cell.isPending() : false,
                        outText: out,
                        textSnippet: cell ? cell.getText().slice(-120).replace(/\\n/g, ' ') : ''
                    };
                }''')

        page.screenshot(path=str(ROOT / f"artifacts/colab/worker_{wid}_status.png"))
        if owns_browser:
            ctx.close()
        return {"worker_id": wid, "triggered": triggered, **status}


def main():
    report = {}
    for wid in [1, 2, 3, 4, 5]:
        try:
            print(f"Checking Worker {wid}...", flush=True)
            report[wid] = inspect_and_ensure_worker(wid)
            print(f"Worker {wid} result: running={report[wid].get('cellIsRunning')}, pending={report[wid].get('cellIsPending')}, outSnippet={report[wid].get('outText', '')[:100]!r}", flush=True)
        except Exception as e:
            print(f"Worker {wid} error: {e}", flush=True)
            report[wid] = {"worker_id": wid, "error": str(e)}

    print("\n================ FLEET LIVE REPORT ================")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
