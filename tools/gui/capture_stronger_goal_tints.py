from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path('gui_web/goal_tint_stronger_shots')
out.mkdir(exist_ok=True)
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    page.goto('http://127.0.0.1:8123/style.html', wait_until='domcontentloaded')
    page.wait_for_function('() => window.__qb && window.__w', timeout=15000)
    page.wait_for_timeout(400)
    page.locator('#board').screenshot(path=str(out / 'goal_tints_stronger.png'))
    browser.close()
