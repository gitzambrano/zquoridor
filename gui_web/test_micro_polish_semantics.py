"""Targeted regression for the semantic micro-polish pass.
Run from gui_web/: python test_micro_polish_semantics.py
"""
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    srv = subprocess.Popen([sys.executable, "dev_server.py", "8213"], cwd=HERE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    failed = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e).split("\n")[0]))
            page.goto("http://127.0.0.1:8213/style.html")
            page.wait_for_timeout(2600)

            def check(name, cond):
                print(("ok: " if cond else "FAIL: ") + name)
                if not cond:
                    failed.append(name)

            def board_hash():
                return page.evaluate("document.getElementById('board').toDataURL()")

            check("boot", "Loading" not in (page.text_content("#status") or ""))
            check("eval readout final size is 12px",
                  page.evaluate("getComputedStyle(document.getElementById('evalNum')).fontSize") == "12px")

            # Freeze optional dynamic overlays so static-effect hashes are stable.
            page.evaluate("setOpt('paths', false); setOpt('dots', false); setOpt('lastMoveStyle', 'off'); setOpt('board', 'wood')")
            page.wait_for_timeout(120)

            # Contrast must repaint the Canvas without a CSS filter on the completed board.
            page.evaluate("setOpt('boardTexture', 'off'); setOpt('goalRows', 'off'); setOpt('boardContrast', 'standard')")
            page.wait_for_timeout(100)
            c0 = board_hash()
            page.evaluate("setOpt('boardContrast', 'strong')")
            page.wait_for_timeout(100)
            c1 = board_hash()
            css_filter = page.evaluate("getComputedStyle(document.getElementById('board')).filter")
            check("board contrast repaints substrate", c0 != c1)
            check("board contrast is not a whole-canvas CSS filter", css_filter == "none")

            # Texture now lives inside QBoard; Off/Subtle/Natural must be distinct.
            page.evaluate("setOpt('boardContrast', 'standard'); setOpt('boardTexture', 'off')")
            page.wait_for_timeout(100)
            t0 = board_hash()
            page.evaluate("setOpt('boardTexture', 'subtle')")
            page.wait_for_timeout(100)
            t1 = board_hash()
            page.evaluate("setOpt('boardTexture', 'natural')")
            page.wait_for_timeout(100)
            t2 = board_hash()
            pseudo = page.evaluate("getComputedStyle(document.getElementById('boardWrap'),'::after').content")
            check("texture off/subtle/natural all differ", len({t0, t1, t2}) == 3)
            check("texture is not a boardWrap pseudo overlay", pseudo in ("none", "normal"))

            # Off means no goal wash AND no coloured goal-edge marker.
            page.evaluate("setOpt('boardTexture', 'off'); setOpt('goalRows', 'off')")
            page.wait_for_timeout(100)
            g0 = board_hash()
            svg_off = page.evaluate("window.__qb.toSVG({coords:true})")
            page.evaluate("setOpt('goalRows', 'subtle')")
            page.wait_for_timeout(100)
            g1 = board_hash()
            svg_subtle = page.evaluate("window.__qb.toSVG({coords:true})")
            check("goal rows repaint", g0 != g1)
            check("goal rows Off removes SVG wash and edges", 'data-zq-goal-rows=' not in svg_off)
            check("goal rows Subtle exports", 'data-zq-goal-rows="subtle"' in svg_subtle)

            # A pawn last move gets a fine halo; Clear is stronger; Off removes it.
            page.evaluate("setOpt('goalRows', 'off'); setOpt('lastMoveStyle', 'off')")
            page.evaluate("window.__qb.lastMove=null; window.__qb.render()")
            lm0 = board_hash()
            page.evaluate("setOpt('lastMoveStyle', 'subtle'); window.__qb.lastMove={type:'pawn',r:8,c:4}; window.__qb.render()")
            lm1 = board_hash()
            page.evaluate("setOpt('lastMoveStyle', 'clear'); window.__qb.lastMove={type:'pawn',r:8,c:4}; window.__qb.render()")
            lm2 = board_hash()
            check("pawn last-move halo appears", lm0 != lm1)
            check("pawn last-move Clear differs from Subtle", lm1 != lm2)

            # Wall Preview now controls active ghosts as well as passive hover.
            ghost_hashes = []
            for mode in ("subtle", "normal", "strong"):
                page.evaluate(f"setOpt('wallPreview','{mode}'); window.__qb.ghost={{o:0,r:3,c:3,state:'ok'}}; window.__qb.render()")
                page.wait_for_timeout(50)
                ghost_hashes.append(board_hash())
            page.evaluate("window.__qb.ghost=null; window.__qb.render()")
            check("active ghost obeys all three Wall Preview strengths", len(set(ghost_hashes)) == 3)

            # PNG uses QBoard renderer; SVG consumes the same deterministic primitives.
            page.evaluate("setOpt('boardTexture','off'); setOpt('boardContrast','standard'); setOpt('goalRows','off'); setOpt('wallProfile','slim')")
            png0 = page.evaluate("window.__qb.renderExport({size:640,coords:true}).toDataURL()")
            page.evaluate("setOpt('boardTexture','natural'); setOpt('boardContrast','strong'); setOpt('goalRows','clear'); setOpt('wallProfile','bold')")
            png1 = page.evaluate("window.__qb.renderExport({size:640,coords:true}).toDataURL()")
            svg = page.evaluate("window.__qb.toSVG({size:640,coords:true})")
            check("PNG export carries renderer-owned effects", png0 != png1)
            check("SVG export carries material texture", 'data-zq-texture="wood:natural"' in svg)
            check("SVG export carries clear goal rows", 'data-zq-goal-rows="clear"' in svg)
            check("SVG export carries wall profile", 'data-zq-wall-profile="bold"' in svg)
            check("SVG export still has coordinates", '<text' in svg and 'abcdefghi' not in svg)
            check("zero page errors", not errors)

            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=2)
        except subprocess.TimeoutExpired:
            srv.kill()

    print(f"\n{len(failed)} failed")
    if failed:
        for f in failed:
            print(" -", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
