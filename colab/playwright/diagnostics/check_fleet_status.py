
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[3]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.config import WORKERS

def inspect_all():
    print("=" * 70, flush=True)
    print("INSPECTING COLAB WORKERS STATUS", flush=True)
    print("=" * 70, flush=True)

    with sync_playwright() as p:
        for wid in [1, 2, 3, 4, 5]:
            w = WORKERS[wid]
            print(f"\n--- Checking Worker {wid} ({w['account']}) ---", flush=True)
            try:
                ctx = p.chromium.launch_persistent_context(
                    user_data_dir=w["profile_dir"],
                    headless=True,
                    channel="chrome",
                    args=[
                        f"--remote-debugging-port={w['cdp_port']}",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                )
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                if "colab" not in (page.url or ""):
                    page.goto(w["notebook_url"], wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(5000)

                data = page.evaluate("""() => {
                    const nb = globalThis.colab?.global?.notebook;
                    const cell = nb?.cells?.[0];
                    const isBusy = cell ? !!cell.busy : false;
                    const lastFailed = cell ? !!cell.lastExecutionFailed : false;

                    const btn = document.querySelector('colab-run-button');
                    const sr = btn ? btn.shadowRoot : null;
                    const wrap = sr ? sr.querySelector('.cell-execution') : null;
                    const cls = wrap ? wrap.className : '';
                    const running = isBusy || cls.includes('running') || cls.includes('animating');
                    const pending = cls.includes('pending') || cls.includes('waiting');

                    const connBtn = document.querySelector('colab-connect-button');
                    const connSr = connBtn ? connBtn.shadowRoot : null;
                    const connTxt = connSr && connSr.querySelector('#connect') ? (connSr.querySelector('#connect').innerText || '') : '';

                    const dialogs = Array.from(document.querySelectorAll('mwc-dialog, colab-dialog, paper-dialog, md-dialog, .modal-dialog'))
                        .filter(d => d.open || d.offsetParent !== null)
                        .map(d => (d.innerText || '').slice(0, 200));

                    const streams = Array.from(document.querySelectorAll('.output-stream, colab-output, .output_text'));
                    const out = streams.map(s => s.innerText || '').join('\\n');

                    return {
                        connTxt: connTxt.trim(),
                        running: running,
                        pending: pending,
                        lastFailed: lastFailed,
                        dialogs: dialogs,
                        cellTextSnippet: (cell?.getText ? cell.getText() : '').slice(0, 150),
                        outputSnippet: out.trim().slice(-600)
                    };
                }""")

                print(f"  Connect button: {data.get('connTxt')}", flush=True)
                print(f"  Cell state: running={data.get('running')}, pending={data.get('pending')}, lastFailed={data.get('lastFailed')}", flush=True)
                if data.get("dialogs"):
                    print(f"  Active dialogs: {data.get('dialogs')}", flush=True)
                print(f"  Output tail:\n{data.get('outputSnippet')}", flush=True)

                ctx.close()
            except Exception as exc:
                print(f"  Inspection error on worker {wid}: {exc}", flush=True)

if __name__ == "__main__":
    inspect_all()
