"""Smoke test for the standalone file:// bundle (zquoridor.html): boot,
a UI pawn move, engine reply, analysis toggle with worker-less slicing
fallback, and Text I/O round trip. No network, no server."""
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

HERE = Path(__file__).parent

def main():
    failures = []
    url = (HERE / "zquoridor.html").as_uri()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        reqs = []
        page.on("request", lambda r: reqs.append(r.url) if not r.url.startswith("file://") else None)
        page.goto(url)
        # The standalone file:// bundle boots the embedded WASM asynchronously.
        # A fixed delay is racy on CI; wait for the actual engine binding that
        # every interaction below depends on.
        try:
            page.wait_for_function(
                "() => window.__w && typeof window.__w.applyPawn === 'function'",
                timeout=15000)
        except PlaywrightTimeoutError:
            diag = page.evaluate("""() => ({
              title: document.title,
              status: document.getElementById('status')?.textContent || '',
              moduleType: typeof ZquoridorModule,
              wasmBytes: typeof __QR_WASM_BYTES__ !== 'undefined' ? __QR_WASM_BYTES__.byteLength : -1,
              dataBytes: typeof __QR_DATA_BYTES__ !== 'undefined' ? __QR_DATA_BYTES__.byteLength : -1,
              engineType: typeof window.__w
            })""")
            print("STANDALONE BOOT DIAG:", diag)
            print("STANDALONE PAGE ERRORS:", errors[:10])
            raise

        def check(name, cond):
            if not cond:
                failures.append(name)
                print("FAIL:", name)
            else:
                print("ok:", name)

        check("boot", "BOOTERR" not in (page.title() or ""))
        check("no network requests", len(reqs) == 0)
        if reqs:
            print("   external:", reqs[:5])
        # play a move through the engine surface + GUI scheduling
        page.evaluate("window.__w.applyPawn(13); afterHumanMove();")
        page.wait_for_timeout(3000)
        check("engine replied", page.evaluate("window.__w.plyCount()") >= 2)
        # analysis on -> worker file missing under file:// -> slicing fallback
        page.click("#panelTabs .tab[data-pane='anPane']")
        page.click("#anEngBtn")
        page.wait_for_timeout(2500)
        rows = page.locator(".pvRow").count()
        check("analysis lines via fallback", rows >= 1)
        # text io round trip
        page.evaluate("openTextIO('qgn')")
        body = page.input_value("#ioArea")
        check("io export works", "[Event" in body)
        page.fill("#ioArea", body)
        page.click("#ioLoad")
        page.wait_for_timeout(400)
        check("io reload closes", not page.is_visible("#overlay"))
        check("zero page errors", len(errors) == 0)
        if errors:
            print("   errors:", errors[:5])
        browser.close()
    print("RESULT:", "PASS" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
