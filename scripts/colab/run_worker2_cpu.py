import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import dismiss_modals

def run_w2():
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp("http://127.0.0.1:9002")
        page = b.contexts[0].pages[0]
        dismiss_modals(page)

        res = page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            if (!cell) return { ok: false, err: 'no cell' };
            let text = cell.getText();
            text = text.replace("assert torch.cuda.is_available(), 'Select a GPU runtime before collection.'",
                                "# CPU fallback: assert torch.cuda.is_available()");
            text = text.replace("--claustrophobia-device gpu", "--claustrophobia-device cpu");

            if (typeof cell.setText === 'function') cell.setText(text);
            if (cell.model && typeof cell.model.setText === 'function') cell.model.setText(text);
            if (typeof monaco !== 'undefined') {
                for (const m of monaco.editor.getModels()) {
                    m.setValue(text);
                }
            }
            return {
                ok: true,
                hasCpu: text.includes("--claustrophobia-device cpu")
            };
        }""")
        print("Updated Worker 2 code:", res, flush=True)
        time.sleep(2)

        # Click run
        btn = page.locator("colab-run-button").first
        btn.click()
        print("Clicked run button on Worker 2", flush=True)
        time.sleep(5)
        dismiss_modals(page)

        info = page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            let out = '';
            if (cell) {
                const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
                const d = elem ? elem.querySelector('.output, colab-output, .output-stream, .output_text') : null;
                out = d ? d.innerText.slice(-200) : '';
            }
            return {
                running: cell && cell.isRunning ? cell.isRunning() : false,
                out: out.replace(/\\n/g, ' ')
            };
        }""")
        print("Worker 2 status:", info, flush=True)

if __name__ == "__main__":
    run_w2()
