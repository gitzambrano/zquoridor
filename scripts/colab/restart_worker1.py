import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import dismiss_modals

def restart_w1():
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp("http://127.0.0.1:9001")
        page = b.contexts[0].pages[0]

        # Dismiss modal if open
        dismiss_modals(page)
        time.sleep(1)

        # Open runtime menu
        btn = page.locator("#runtime-menu-button")
        if btn.count() > 0:
            btn.click()
            time.sleep(1)

            # Find menu items
            items = page.locator(".goog-menuitem")
            print("Menu items count:", items.count())
            for i in range(items.count()):
                txt = items.nth(i).inner_text()
                if "Reiniciar sessão" in txt or "Reiniciar ambiente" in txt or "Restart session" in txt:
                    print(f"Clicking menu item: {txt}")
                    items.nth(i).click()
                    time.sleep(2)
                    dismiss_modals(page)
                    break

        time.sleep(5)
        # Re-trigger Cell 0
        res = page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            if (!cell) return false;
            let clicked = false;
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            if (elem) {
                const b = elem.querySelector('colab-run-button');
                if (b && b.shadowRoot) {
                    const inner = b.shadowRoot.querySelector('#run-button, button');
                    if (inner) {
                        inner.click();
                        clicked = true;
                    }
                }
            }
            if (!clicked && typeof cell.manualExecute === 'function') {
                cell.manualExecute();
                clicked = true;
            }
            return clicked;
        }""")
        print("Triggered Cell 0:", res)
        time.sleep(4)
        dismiss_modals(page)

if __name__ == "__main__":
    restart_w1()
