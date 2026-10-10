from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9004")
    ctx = browser.contexts[0]
    page = next(pg for pg in ctx.pages if "colab" in (pg.url or ""))
    
    # Click Conectar sem GPU
    btn = page.locator("button:has-text('Conectar sem GPU'), md-text-button:has-text('Conectar sem GPU')")
    if btn.count() > 0:
        print("Clicking Conectar sem GPU...")
        btn.first.click()
        time.sleep(3)
        
    # Click Run
    print("Clicking Run...")
    page.evaluate("""() => {
        const nb = globalThis.colab?.global?.notebook;
        const cell = nb?.cells ? nb.cells[0] : null;
        if (cell) {
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            const btn = elem ? elem.querySelector('colab-run-button') : null;
            const inner = btn && btn.shadowRoot ? btn.shadowRoot.querySelector('#run-button, button') : null;
            if (inner) inner.click();
            else if (typeof cell.manualExecute === 'function') cell.manualExecute();
        }
    }""")
    time.sleep(4)
    page.screenshot(path="worker_4_running.png")
    print("Captured worker_4_running.png")
