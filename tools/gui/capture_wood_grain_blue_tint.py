from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path('wood_grain_blue_tint_shots')
out.mkdir(exist_ok=True)
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    page.goto('http://127.0.0.1:8123/style.html', wait_until='domcontentloaded')
    page.wait_for_function('() => window.__qb', timeout=15000)
    page.evaluate("setOpt('board','wood'); setOpt('boardTexture','subtle'); setOpt('goalRows','subtle'); setOpt('boardContrast','standard')")
    page.wait_for_timeout(300)
    page.locator('#board').screenshot(path=str(out / 'wood_grain_blue_tint.png'))
    browser.close()
