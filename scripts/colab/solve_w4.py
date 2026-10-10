import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright
import time
from scripts.colab.browser_utils import solve_recaptcha_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9004")
    ctx = browser.contexts[0]
    page = next(pg for pg in ctx.pages if "colab" in (pg.url or ""))
    res = solve_recaptcha_playwright(page)
    print("solve_recaptcha_playwright result:", res)
    time.sleep(2)
    page.screenshot(path="worker_4_after.png")
