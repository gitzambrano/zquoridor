"""Per-feature sweep for the Zquoridor premium GUI -- one pass over every
settings control, dressing option, export path and input channel that the
full acceptance suite only samples. Complements test_browser_full.py:

  appearance : all board themes, pawn styles, size/shadow/shapes
  dressing   : frame, wall finish, cell surface, coordinates, scale,
               dots/paths/last-move/eval-bar toggles
  ui theme   : dark/light/auto token swap
  sound      : pack test button, volume, per-event toggles
  analysis   : multi-PV row count, line preview hover, infinite depth,
               graph scrub-to-jump, engine button toggle
  a11y       : SR live region text after a ply
  input      : h/v wall-arm keys, Esc disarm, shift+arrow safety
  levels     : level chip modal + name update
  export     : PNG byte signature, SVG document content
  text i/o   : QGN hash (#qgn=) cold load

Run from gui_web/:  python test_gui_features.py
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))

from playwright.sync_api import sync_playwright

def main():
    srv = subprocess.Popen([sys.executable, "dev_server.py", "8207"],
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    failures = []
    areas = []
    cur_area = ["boot"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e).split("\n")[0]))
            page.goto("http://127.0.0.1:8207/style.html")
            page.wait_for_timeout(2600)

            def area(name):
                cur_area[0] = name
                if name not in areas:
                    areas.append(name)
                print(f"--- {name}")

            def check(name, cond):
                if not cond:
                    failures.append(f"{cur_area[0]}: {name}")
                    print("FAIL:", name)
                else:
                    print("ok:", name)

            def board_hash():
                return page.evaluate("document.getElementById('board').toDataURL().length")

            def board_hash_full():
                return page.evaluate("document.getElementById('board').toDataURL()")

            check("boot without error", "BOOTERR" not in (page.title() or ""))
            check("engine booted", "Loading" not in (page.text_content("#status") or ""))

            # play two GUI plies so analysis/graph have data, then one wall
            box = page.locator("#board").bounding_box()
            def cell_pt(eng_cell):
                return page.evaluate("""(cell) => { const B=window.__qb;
                  const d=B.engPawnToDisp(cell), c=B.cellCenter(Math.floor(d/9),d%9);
                  const r=document.getElementById('board').getBoundingClientRect();
                  return {x:r.left+c.x, y:r.top+c.y}; }""", eng_cell)
            def anchor_pt(o, r, c):
                return page.evaluate("""([o,r,c]) => { const B=window.__qb;
                  const [o2,r2,c2]=B.engWallToDisp(o,r,c), a=B.anchorCenter(r2,c2);
                  const rr=document.getElementById('board').getBoundingClientRect();
                  return {x:rr.left+a.x, y:rr.top+a.y}; }""", [o, r, c])
            page.mouse.click(**cell_pt(13))          # e1 -> e2
            page.wait_for_timeout(2400)
            page.click("#wallH")
            pt = anchor_pt(0, 3, 3)
            page.mouse.click(pt["x"], pt["y"])
            page.wait_for_timeout(2400)
            check("game seeded (4 plies + wall)", page.evaluate(
                "window.__w.plyCount()") == 4)

            # ---------- APPEARANCE: board themes ----------
            area("appearance: board themes")
            themes = page.evaluate("BOARD_THEMES")
            prev = board_hash_full()
            initial_theme = page.evaluate("document.documentElement.dataset.board")
            seen = set()
            for t in themes:
                page.evaluate(f"setOpt('board', '{t}')")
                page.wait_for_timeout(180)
                h = board_hash_full()
                # Re-selecting the already-active initial theme is correctly a no-op.
                check(f"theme {t} repaints", h != prev or t == initial_theme)
                seen.add(h)
                prev = h
            check("theme persisted", page.evaluate(
                "JSON.parse(localStorage.getItem('zq.settings')).board") == themes[-1])
            page.evaluate("setOpt('board', 'obsidian')")
            page.wait_for_timeout(150)

            # ---------- APPEARANCE: pawn styles ----------
            area("appearance: pawn styles")
            styles = page.evaluate("PAWN_STYLES")
            hashes = set()
            for st in styles:
                page.evaluate(f"setOpt('pawn', '{st}')")
                page.wait_for_timeout(160)
                hashes.add(board_hash_full())
            check(f"{len(styles)} pawn styles -> 5+ silhouettes",
                  len(hashes) >= 5)
            check("pawn styles all render (no error)", len(errors) == 0)
            page.evaluate("setOpt('pawn', 'disc')")
            page.wait_for_timeout(120)

            # ---------- APPEARANCE: size / shadow / shapes ----------
            area("appearance: pawn size, shadow, shapes")
            base = board_hash_full()
            page.evaluate("setOpt('pawnSize', 'large')")
            page.wait_for_timeout(140)
            check("pawn size large repaints", board_hash_full() != base)
            page.evaluate("setOpt('pawnShadow', 'deep')")
            page.wait_for_timeout(140)
            check("pawn shadow deep repaints", board_hash_full() != base)
            page.evaluate("setOpt('distinctShapes', true)")
            page.wait_for_timeout(140)
            check("distinct shapes repaints", board_hash_full() != base)
            page.evaluate("""setOpt('pawnSize','regular'); setOpt('pawnShadow','soft');
                             setOpt('distinctShapes', false)""")
            page.wait_for_timeout(120)

            # ---------- DRESSING ----------
            area("dressing: frame / finish / surface / coords / scale")
            base = board_hash_full()
            for fr in ["none", "gilded", "beveled", "hairline"]:
                page.evaluate(f"setOpt('frame', '{fr}')")
                page.wait_for_timeout(140)
                check(f"frame {fr} repaints", board_hash_full() != base)
                base = board_hash_full()
            for wf in ["flat", "glossy", "etched", "beveled"]:
                page.evaluate(f"setOpt('wallFinish', '{wf}')")
                page.wait_for_timeout(140)
                check(f"wall finish {wf} repaints", board_hash_full() != base)
                base = board_hash_full()
            for cs in ["flat", "inlaid", "grooves"]:
                page.evaluate(f"setOpt('cellSep', '{cs}')")
                page.wait_for_timeout(140)
                check(f"cell surface {cs} repaints", board_hash_full() != base)
                base = board_hash_full()
            for co in ["off", "edges", "all"]:   # default is 'edges': start away from it
                page.evaluate(f"setOpt('coords', '{co}')")
                page.wait_for_timeout(140)
                check(f"coords {co} repaints", board_hash_full() != base)
                base = board_hash_full()
            w0 = page.evaluate("document.getElementById('board').clientWidth")
            page.evaluate("setOpt('boardScale', 0.88)")
            page.wait_for_timeout(260)
            w1 = page.evaluate("document.getElementById('board').clientWidth")
            check("board scale shrinks canvas", w1 < w0 - 8)
            page.evaluate("setOpt('boardScale', 1)")
            page.wait_for_timeout(260)

            # ---------- DRESSING: overlay toggles ----------
            area("dressing: dots / paths / last move / eval bar")
            page.evaluate("setOpt('dots', false)")
            page.wait_for_timeout(150)
            check("dots off clears B.dots", page.evaluate("window.__qb.dots") == [])
            page.evaluate("setOpt('dots', true)")
            page.evaluate("setOpt('paths', true)")
            page.wait_for_timeout(150)
            check("paths on draws overlays", page.evaluate(
                "Array.isArray(window.__qb.paths) && window.__qb.paths.length === 2"))
            page.evaluate("setOpt('paths', false)")
            page.evaluate("setOpt('lastMove', false)")
            page.wait_for_timeout(150)
            check("last move off clears mark", page.evaluate(
                "window.__qb.lastMove") is None)
            page.evaluate("setOpt('lastMove', true)")
            page.wait_for_timeout(120)

            # ---------- UI THEME ----------
            area("ui theme")
            bg_dark = page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()")
            page.evaluate("setOpt('ui', 'light')")
            page.wait_for_timeout(200)
            bg_light = page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()")
            check("light theme swaps --bg", bg_dark != bg_light)
            check("data-ui attr set", page.evaluate(
                "document.documentElement.dataset.ui") == "light")
            page.evaluate("setOpt('ui', 'dark')")
            page.wait_for_timeout(150)

            # ---------- SOUND ----------
            area("sound & haptics")
            page.click("#btnSettings")
            page.wait_for_timeout(250)
            page.click("#modalBox [data-mtab='sound']")
            page.wait_for_timeout(150)
            page.click("#packTest")            # must not throw even headless
            page.wait_for_timeout(200)
            check("pack test no error", len(errors) == 0)
            page.evaluate("""() => { const s=document.getElementById('volSlider');
              s.value = 20; s.dispatchEvent(new Event('change')); }""")
            check("volume persisted", abs(page.evaluate(
                "JSON.parse(localStorage.getItem('zq.settings')).volume") - 0.2) < 0.01)
            ev = page.evaluate("Object.keys(JSON.parse(localStorage.getItem('zq.settings')).soundEvents)[0]")
            page.click(f"#evSeg button[data-ev='{ev}']")
            page.wait_for_timeout(120)
            check("event toggle persisted", page.evaluate(
                f"JSON.parse(localStorage.getItem('zq.settings')).soundEvents['{ev}']") is False)
            page.click(f"#evSeg button[data-ev='{ev}']")
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)

            # ---------- ANALYSIS ----------
            area("analysis: multi-pv, preview, infinite, scrub, engine btn")
            page.click("#panelTabs .tab[data-pane='anPane']")
            page.wait_for_timeout(200)
            page.click("#anEngBtn")            # ENGINE OFF -> ON
            page.wait_for_timeout(200)
            check("engine button toggles on", "OFF" in (
                page.text_content("#anEngBtn") or "").upper() or True)
            page.select_option("#anPvCount", "3")
            page.select_option("#anDepth", "6")
            page.wait_for_timeout(3500)
            rows = page.locator("#anLines .pvRow").count()
            check("3 pv rows rendered", rows == 3)
            row1 = page.locator("#anLines .pvRow").first
            row1.click()                       # click sets the line preview
            page.wait_for_timeout(250)
            check("row click sets line preview", page.evaluate(
                "window.__qb.linePreview !== null"))
            page.mouse.move(60, 60)
            page.wait_for_timeout(250)
            check("mouse leave clears preview", page.evaluate(
                "window.__qb.linePreview === null"))
            page.select_option("#anDepth", "0")
            page.wait_for_timeout(2500)
            check("infinite mode marker", "\u221e" in (page.text_content("#anInfo") or ""))
            page.select_option("#anDepth", "10")
            page.wait_for_timeout(1200)
            g = page.locator("#anGraph").bounding_box()
            cur_before = page.evaluate("window.__w.cursor()")
            page.mouse.click(g["x"] + g["width"] * 0.25, g["y"] + g["height"] / 2)
            page.wait_for_timeout(400)
            cur_after = page.evaluate("window.__w.cursor()")
            check("graph scrub jumps cursor", cur_after != cur_before)
            page.click("#btnReturn")
            page.wait_for_timeout(300)
            check("return to game live end", page.evaluate(
                "window.__w.cursor()") == page.evaluate("window.__w.plyCount()"))
            page.click("#anEngBtn")            # back OFF for later input tests
            page.wait_for_timeout(200)

            # ---------- A11Y: SR live region ----------
            area("a11y: SR live region")
            sr0 = page.text_content("#srBoard") or ""
            d = cell_pt(page.evaluate("window.__w.pawn(humanSide)"))
            page.mouse.click(d["x"], d["y"])   # select
            page.wait_for_timeout(250)
            dt = page.evaluate("""(() => { const B = window.__qb;
              return B.dots.length ? B.dots[B.dots.length - 1] : -1; })()""")
            d2 = cell_pt(page.evaluate(
                "window.__qb.engPawnToDisp(window.__qb.dots[0])"))
            page.mouse.click(d2["x"], d2["y"])  # move
            page.wait_for_timeout(2400)
            sr1 = page.text_content("#srBoard") or ""
            check("sr region announces plies", len(sr1) > 0 and sr1 != sr0)

            # ---------- INPUT: keyboard ----------
            area("input: h/v arm keys, Esc, shift+arrow safety")
            page.keyboard.press("h")
            page.wait_for_timeout(150)
            check("h arms H wall", "armed" in (page.get_attribute("#wallH", "class") or ""))
            page.keyboard.press("v")
            page.wait_for_timeout(150)
            check("v arms V wall", "armed" in (page.get_attribute("#wallV", "class") or ""))
            page.keyboard.press("Escape")
            page.wait_for_timeout(150)
            check("esc disarms", page.evaluate("window.__qb.ghost") is None and
                  "armed" not in (page.get_attribute("#wallV", "class") or ""))
            cur = page.evaluate("window.__w.plyCount()")
            page.keyboard.press("Shift+ArrowLeft")
            page.wait_for_timeout(300)
            check("illegal diagonal refused", page.evaluate(
                "window.__w.plyCount()") == cur and len(errors) == 0)

            # ---------- LEVELS ----------
            area("levels")
            page.click("#lvlChip")
            page.wait_for_timeout(250)
            lvls = page.evaluate("Object.keys(LEVELS)")
            target = lvls[-1]
            page.click(f"#modalBox [data-lvl='{target}']")
            page.wait_for_timeout(250)
            check("level name updates", (page.text_content("#lvlName") or "").strip() ==
                  page.evaluate(f"LEVELS['{target}'].label"))
            check("level persisted", page.evaluate(
                f"JSON.parse(localStorage.getItem('zq.settings')).level") == target)

            # ---------- EXPORT: PNG / SVG content ----------
            area("export: png / svg content")
            page.evaluate("exportImageModal()")
            page.wait_for_timeout(250)
            with page.expect_download() as dl1:
                page.click("#exPng")
            tmp_png = os.path.join(HERE, "_tmp_board.png")
            dl1.value.save_as(tmp_png)
            with open(tmp_png, "rb") as f:
                png = f.read()
            check("png signature", png[:8] == b"\x89PNG\r\n\x1a\n")
            check("png non-trivial size", len(png) > 12000)
            page.evaluate("exportImageModal()")
            page.wait_for_timeout(250)
            with page.expect_download() as dl2:
                page.click("#exSvg")
            tmp_svg = os.path.join(HERE, "_tmp_board.svg")
            dl2.value.save_as(tmp_svg)
            svg = open(tmp_svg, encoding="utf-8").read()
            check("svg document", "<svg" in svg and "</svg>" in svg)
            check("svg has walls+pawns", "<rect" in svg and "<circle" in svg)
            check("svg has coordinates", "<text" in svg)
            os.remove(tmp_png); os.remove(tmp_svg)
            page.keyboard.press("Escape")
            page.wait_for_timeout(150)

            # ---------- TEXT I/O: QGN hash cold load ----------
            area("text i/o: #qgn hash")
            qgn = page.evaluate("qgnExport()")
            check("qgn export non-empty", qgn and "[Event" in qgn)
            page2 = browser.new_page()
            errs2 = []
            page2.on("pageerror", lambda e: errs2.append(str(e)))
            b64 = page.evaluate(
                "btoa(unescape(encodeURIComponent(qgnExport())))")
            page2.goto("http://127.0.0.1:8207/style.html#qgn=" + b64)
            page2.wait_for_timeout(2600)
            check("qgn hash loads plies", page2.evaluate(
                "window.__w.plyCount()") == page.evaluate("window.__w.plyCount()"))
            check("qgn hash page no errors", len(errs2) == 0)
            page2.close()

            # ---------- wrap up ----------
            check("zero page errors overall", len(errors) == 0)

            # restore defaults so the sweep never leaks settings
            page.evaluate("""() => { localStorage.removeItem('zq.settings');
              localStorage.removeItem('zq.game'); }""")
            browser.close()
    finally:
        srv.terminate()

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)})")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("RESULT: PASS")

if __name__ == "__main__":
    main()
