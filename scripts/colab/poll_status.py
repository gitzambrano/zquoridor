"""Poll live progress across all 5 workers."""
from playwright.sync_api import sync_playwright
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.search_games_profile import build_worker_profile
from scripts.colab.browser_utils import is_cdp_reachable

results = {}
with sync_playwright() as p:
    for wid in [1, 2, 3, 4, 5]:
        profile = build_worker_profile(wid)
        port = profile['cdp_port']
        try:
            if is_cdp_reachable(port):
                b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
                ctx = b.contexts[0]
                owns = False
            else:
                ctx = p.chromium.launch_persistent_context(
                    user_data_dir=profile['profile_dir'],
                    headless=True,
                    channel='chrome',
                    args=[f'--remote-debugging-port={port}', '--remote-allow-origins=*'],
                )
                owns = True

            page = None
            for pg in ctx.pages:
                if 'colab.research.google.com' in pg.url:
                    page = pg
                    break
            if not page:
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(profile['notebook_url'], wait_until='domcontentloaded', timeout=60000)

            page.wait_for_timeout(5000)
            info = page.evaluate('''() => {
                const cell = globalThis.colab?.global?.notebook?.cells[0];
                const elem = cell ? (cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_)) : null;
                let out = '';
                if (elem) {
                    const d = elem.querySelector('.output, colab-output, .output-stream, .output_text');
                    out = d ? d.innerText.slice(-600) : '';
                }
                const btn = elem ? elem.querySelector('colab-run-button') : null;
                const sr = btn ? btn.shadowRoot : null;
                const wrap = sr ? sr.querySelector('.cell-execution') : null;
                return {
                    isRunning: cell ? cell.isRunning() : false,
                    isPending: cell ? cell.isPending() : false,
                    statusClass: wrap ? wrap.className : '',
                    outText: out
                };
            }''')
            results[wid] = info
            if owns:
                ctx.close()
        except Exception as e:
            results[wid] = {'error': str(e)}

print("\n" + "="*50)
for wid, r in results.items():
    print(f"Colab {wid}: isRunning={r.get('isRunning')}, isPending={r.get('isPending')}, class={r.get('statusClass')}")
    out = r.get('outText', '')
    if out:
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        last_line = lines[-1] if lines else ''
        print(f"   Last output: {last_line}")
print("="*50)
