from pathlib import Path
import sys
import time
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.config import WORKERS

def inspect_worker(p, wid: int):
    w = WORKERS[wid]
    port = w["cdp_port"]
    print(f"\n--- INSPECTING WORKER {wid} ({w['account']}) on port {port} ---")
    try:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=w["profile_dir"],
            headless=True,
            channel="chrome",
            args=[f"--remote-debugging-port={port}", "--remote-allow-origins=*"],
        )
        page = next((pg for pg in ctx.pages if "colab.research.google.com" in (pg.url or "")), None)
        if not page:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(w["notebook_url"], wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)

        info = page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            let out = '';
            let code = '';
            if (cell) {
                if (typeof cell.getText === 'function') code = cell.getText().slice(0, 150);
                const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
                const d = elem ? elem.querySelector('.output, colab-output, .output-stream, .output_text') : null;
                out = d ? d.innerText.slice(-1000) : '';
            }
            
            const dialogs = Array.from(document.querySelectorAll('mwc-dialog, colab-dialog')).filter(d => d.open || d.offsetParent !== null).map(d => ({
                text: (d.innerText || '').slice(0, 200),
                buttons: Array.from(d.querySelectorAll('button, md-text-button, mwc-button, paper-button')).map(b => (b.innerText || '').trim())
            }));

            const frames = Array.from(document.querySelectorAll('iframe')).map(f => f.src);
            const hasRecaptcha = frames.some(src => src.includes('recaptcha'));

            return {
                running: cell && cell.isRunning ? cell.isRunning() : false,
                code_snippet: code,
                output: out,
                dialogs: dialogs,
                hasRecaptcha: hasRecaptcha
            };
        }""")
        print(f"Running: {info['running']}")
        print(f"Code: {info['code_snippet'][:80]}...")
        print(f"Dialogs: {info['dialogs']}")
        print(f"Has reCAPTCHA iframe: {info['hasRecaptcha']}")
        print(f"Output tail:\n{info['output']}")
        ctx.close()
    except Exception as e:
        print(f"Inspection error on Worker {wid}: {e}")

def main():
    with sync_playwright() as p:
        for wid in [1, 2, 3, 4, 5]:
            inspect_worker(p, wid)

if __name__ == "__main__":
    main()
