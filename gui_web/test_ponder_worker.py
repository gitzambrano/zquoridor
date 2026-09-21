"""Browser regression for persistent worker pondering.

The test verifies that:
1. the WASM ponder export exists;
2. two complete human/engine cycles work with the persistent worker;
3. the main UI thread keeps ticking while the worker ponders;
4. no worker/page error is emitted.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main() -> int:
    server = subprocess.Popen(
        [sys.executable, "dev_server.py", "8217"],
        cwd=HERE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    failures: list[str] = []
    try:
        time.sleep(1.0)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
            page.on(
                "console",
                lambda m: errors.append("console: " + m.text)
                if m.type == "error"
                else None,
            )
            page.goto("http://127.0.0.1:8217/style.html", wait_until="domcontentloaded")
            page.wait_for_function(
                "() => window.__w && typeof ANW !== 'undefined' && ANW.ready",
                timeout=15000,
            )

            def check(name: str, condition: bool) -> None:
                if condition:
                    print("ok:", name)
                else:
                    print("FAIL:", name)
                    failures.append(name)

            check(
                "WASM ponder export",
                page.evaluate("typeof window.__w.ponder === 'function'"),
            )
            check("worker ready", page.evaluate("ANW.ready && !ANW.failed"))

            # Keep the test short. Pondering starts after each engine reply and
            # continues in the worker while the human side is to move.
            page.evaluate("S.level='pawn'; saveSettings();")

            def play_human_pawn() -> bool:
                data = page.evaluate(
                    """() => {
                      const W=window.__w;
                      for (let i=0;i<W.moveCount();i++) {
                        if (!W.mvIsWall(i)) return W.mvA(i);
                      }
                      return -1;
                    }"""
                )
                if data < 0:
                    return False
                old = page.evaluate("window.__w.plyCount()")
                ok = page.evaluate("(d) => window.__w.applyPawn(d)", data)
                if not ok:
                    return False
                page.evaluate("afterHumanMove()")
                try:
                    page.wait_for_function(
                        """old => gameOver ||
                          (window.__w.plyCount() >= old + 2 &&
                           window.__w.turn() === humanSide &&
                           !engineThinking)""",
                        arg=old,
                        timeout=12000,
                    )
                except Exception:
                    return False
                return True

            check("first human/engine cycle", play_human_pawn())
            check("worker still healthy after first reply",
                  page.evaluate("ANW.ready && !ANW.failed"))

            # The worker should now be pondering. A fast main-thread heartbeat
            # proves that background search does not freeze browser interaction.
            page.evaluate(
                """() => {
                  window.__ponderTicks = 0;
                  window.__ponderTimer = setInterval(() => window.__ponderTicks++, 20);
                }"""
            )
            page.wait_for_timeout(420)
            ticks = page.evaluate(
                """() => {
                  clearInterval(window.__ponderTimer);
                  return window.__ponderTicks;
                }"""
            )
            check("UI heartbeat stays responsive during ponder", ticks >= 12)

            # Exercise a visible UI action while the worker may still be in a
            # ponder slice. It must not stall or kill the worker.
            page.click("#btnSettings")
            page.wait_for_timeout(80)
            check("settings opens during ponder", page.is_visible("#overlay"))
            page.keyboard.press("Escape")
            page.wait_for_timeout(80)

            check("second human/engine cycle", play_human_pawn())
            check("worker still healthy after second reply",
                  page.evaluate("ANW.ready && !ANW.failed"))

            check("zero page/console errors", not errors)
            if errors:
                for err in errors[:10]:
                    print("  ", err)

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()

    if failures:
        print("RESULT: FAIL")
        for failure in failures:
            print(" -", failure)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
