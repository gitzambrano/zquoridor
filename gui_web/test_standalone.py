"""Smoke test for the standalone file:// bundle (zquoridor.html): boot,
a UI pawn move, engine reply, analysis toggle with worker-less slicing
fallback, and Text I/O round trip. No network, no server."""
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

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
        page.wait_for_timeout(3000)

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
