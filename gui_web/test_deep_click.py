"""Deep click-through for the Zquoridor premium GUI -- drives every primary
user flow with real pointer/keyboard input and asserts engine/board state
after each step. Complements the three existing suites by going WIDER on
interaction paths they only sample:

  pawns     : every legal opening destination clicked, selection dots
              exact, takeback after each, illegal + opponent clicks dead
  walls     : arm+click commit, crossing refusal, overlap refusal,
              keyboard arm/disarm, dock drag-drop, magnetic assist snap,
              confirm-chip commit AND cancel (Enter / ccNo)
  settings  : every card driven through REAL control clicks (swatches,
              segs, sliders, selects), persistence to zq.settings checked,
              preset bundle, worker toggle for the analysis fallback below
  analysis  : main-thread slicing fallback (worker off), 3 PV rows,
              line preview draw/clear, infinite depth marker, graph scrub,
              blunder check accuracy card, then worker round trip
  editor    : pawn tool, canonical 2x2 seal -> path refusal, wall-budget
              steppers vs the 20-wall invariant, turn switch, gated Apply
  i/o       : QGN/QFEN modal round trip, bad-token diagnostic, #qfen=
              cold load, autosave Resume chip, Recent games lifecycle

Run from gui_web/:  python test_deep_click.py
"""
import subprocess
import sys
import time
import os
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))

from playwright.sync_api import sync_playwright

PORT = 8213


def main():
    srv = subprocess.Popen([sys.executable, "dev_server.py", str(PORT)],
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    failures = []
    n_ok = [0]
    try:
        run_suite(failures, n_ok)
    finally:
        srv.terminate()
    print()
    print(f"checks passed: {n_ok[0]}  failed: {len(failures)}")
    if failures:
        for f in failures:
            print("FAIL:", f)
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


def run_suite(failures, n_ok):
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        def check(name, cond):
            if cond:
                n_ok[0] += 1
                print("ok:", name)
            else:
                failures.append(name)
                print("FAIL:", name)

        page.goto(f"http://localhost:{PORT}/zquoridor.html")
        page.wait_for_timeout(900)
        check("boot without error", "BOOTERR" not in (page.title() or ""))
        check("engine booted", "Loading" not in (page.text_content("#status") or ""))

        box = page.locator("#board").bounding_box()

        def cell_pt(eng_cell):
            return page.evaluate("""(cell) => {
              const B = window.__qb;
              const d = B.engPawnToDisp(cell);
              const c = B.cellCenter(Math.floor(d/9), d%9);
              const r = document.getElementById('board').getBoundingClientRect();
              return {x: r.left + c.x, y: r.top + c.y};
            }""", eng_cell)

        def anchor_pt(o, er, ec):
            return page.evaluate("""([o,r,c]) => {
              const B = window.__qb;
              const [do_,dr,dc] = B.engWallToDisp(o,r,c);
              const a = B.anchorCenter(dr,dc);
              const rr = document.getElementById('board').getBoundingClientRect();
              return {x: rr.left+a.x, y: rr.top+a.y};
            }""", [o, er, ec])

        def dot_pt(dd):
            return page.evaluate("""(dd) => { const c = window.__qb.cellCenter(Math.floor(dd/9), dd%9);
                              const r = document.getElementById('board').getBoundingClientRect();
                              return {x: r.left+c.x, y: r.top+c.y}; }""", dd)

        def human():
            return page.evaluate("humanSide")

        def state():
            return page.evaluate("""(() => ({
              cur: window.__w.cursor(), n: window.__w.plyCount(),
              turn: window.__w.turn(), think: engineThinking,
              over: gameOver, h0: window.__w.wallsLeft(0), h1: window.__w.wallsLeft(1)
            }))()""")

        def wait_reply(prev_cur, timeout=15):
            t0 = time.time()
            hs = human()
            while time.time() - t0 < timeout:
                s = state()
                if s["turn"] == hs and not s["think"] and s["cur"] > prev_cur:
                    return s
                page.wait_for_timeout(120)
            raise AssertionError(f"engine reply timeout (state={state()})")

        def fresh_game():
            page.evaluate("newGame()")
            page.wait_for_timeout(250)
            page.evaluate("newGame()")
            page.wait_for_timeout(500)

        def click_pawn_to(dst_eng):
            """One human pawn move by clicking select + destination dot."""
            cur = page.evaluate("window.__w.cursor()")
            src = cell_pt(page.evaluate(f"window.__w.pawn({human()})"))
            page.mouse.click(src["x"], src["y"])
            page.wait_for_timeout(150)
            dd = page.evaluate(f"window.__qb.engPawnToDisp({dst_eng})")
            pt = dot_pt(dd)
            page.mouse.click(pt["x"], pt["y"])
            return wait_reply(cur)

        # ================= A. PAWN MATRIX =================
        fresh_game()
        hs = human()
        src = cell_pt(page.evaluate(f"window.__w.pawn({hs})"))
        page.mouse.click(src["x"], src["y"])
        page.wait_for_timeout(200)
        sel = page.evaluate("window.__qb.selected")
        dots = page.evaluate("window.__qb.dots")
        exp = sorted(page.evaluate("[13,5,3].map(c => window.__qb.engPawnToDisp(c))"))
        check("pawn click selects", sel >= 0)
        check("selection dots exact {d1,f1,e2}", sorted(dots) == exp)
        # an illegal destination click does nothing
        far = cell_pt(40)
        page.mouse.click(far["x"], far["y"])
        page.wait_for_timeout(150)
        check("illegal dest refused", page.evaluate("window.__w.cursor()") == 0)
        # a click on the opponent pawn does nothing
        opp = cell_pt(page.evaluate(f"window.__w.pawn({1-hs})"))
        page.mouse.click(opp["x"], opp["y"])
        page.wait_for_timeout(150)
        check("opponent pawn inert", page.evaluate("window.__w.cursor()") == 0)

        for eng_dst in (13, 3, 5):   # every opening destination, clicked
            fresh_game()
            click_pawn_to(eng_dst)
            s = state()
            check(f"clicked pawn move to {eng_dst} committed + engine replied",
                  s["cur"] == 2 and s["turn"] == hs)
            page.click("#btnUndo")
            page.wait_for_timeout(400)
            check(f"takeback after {eng_dst}", page.evaluate("window.__w.cursor()") == 0)

        # ================= B. WALL MATRIX =================
        fresh_game()
        # B1: arm + click commits; ghost ok before release
        page.click("#wallH"); page.wait_for_timeout(150)
        check("H armed via dock", "armed" in (page.get_attribute("#wallH", "class") or ""))
        a = anchor_pt(0, 3, 3)
        page.mouse.move(a["x"], a["y"]); page.mouse.down(); page.wait_for_timeout(100)
        g = page.evaluate("window.__qb.ghost")
        check("ghost ok on legal anchor", bool(g) and g.get("state") == "ok"
              and g.get("o") == 0)
        page.mouse.up(); page.wait_for_timeout(150)
        s = wait_reply(0)
        check("arm+click wall placed", s["cur"] == 2 and
              page.evaluate("window.__w.wallHBit(3*8+3)") == 1 and
              page.evaluate("window.__w.wallsLeft(humanSide)") == 9)

        # B2: crossing refusal (V over the H we just placed)
        page.keyboard.press("v"); page.wait_for_timeout(150)
        a = anchor_pt(1, 3, 3)
        page.mouse.move(a["x"], a["y"]); page.mouse.down(); page.wait_for_timeout(120)
        g = page.evaluate("window.__qb.ghost")
        st_txt = page.text_content("#status") or ""
        check("crossing ghost is bad", bool(g) and g.get("state") == "bad")
        check("crossing reason named", "Crosses" in st_txt)
        page.mouse.up(); page.wait_for_timeout(200)
        check("crossing wall NOT placed",
              page.evaluate("window.__w.cursor()") == 2 and
              page.evaluate("window.__w.wallVBit(3*8+3)") == 0)

        # B3: overlap refusal (same slot again)
        page.keyboard.press("h"); page.wait_for_timeout(150)
        a = anchor_pt(0, 3, 3)
        page.mouse.move(a["x"], a["y"]); page.mouse.down(); page.wait_for_timeout(120)
        g = page.evaluate("window.__qb.ghost")
        st_txt = page.text_content("#status") or ""
        check("overlap ghost is bad", bool(g) and g.get("state") == "bad")
        check("overlap reason named", "Overlaps" in st_txt)
        page.mouse.up(); page.wait_for_timeout(150)
        page.keyboard.press("Escape"); page.wait_for_timeout(120)
        check("Esc disarms", page.evaluate("window.__qb.ghost") is None and
              "armed" not in (page.get_attribute("#wallH", "class") or ""))

        # B4: dock drag-drop places a V wall. Hold past the 150 ms threshold
        # first -- the drag gate accepts movement delta OR a held press, and
        # an instant release must stay a click-to-arm.
        dock = page.locator("#wallV").bounding_box()
        a = anchor_pt(1, 4, 4)
        page.mouse.move(dock["x"] + dock["width"]/2, dock["y"] + dock["height"]/2)
        page.mouse.down()
        page.wait_for_timeout(230)
        page.mouse.move(a["x"], a["y"], steps=10); page.wait_for_timeout(120)
        g = page.evaluate("window.__qb.ghost")
        check("drag ghost follows to V anchor", bool(g) and g.get("o") == 1)
        page.mouse.up()
        s = wait_reply(2)
        check("dock drag placed V wall", s["cur"] == 4 and
              page.evaluate("window.__w.wallVBit(4*8+4)") == 1)

        # B5: magnetic assist -- find a live illegal anchor with a legal
        # cardinal neighbour (engine replies make fixed spots unreliable),
        # then press 0.48 U from the illegal one toward the neighbour: the
        # nearest anchor stays the illegal one while the neighbour sits
        # inside the 0.9 U assist radius
        pair = page.evaluate("""(() => {
          for (const o of [0, 1]) {
            for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
              if (legalWall[o*64 + r*8 + c]) continue;
              for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                const r2 = r+dr, c2 = c+dc;
                if (r2 < 0 || r2 > 7 || c2 < 0 || c2 > 7) continue;
                if (!legalWall[o*64 + r2*8 + c2]) continue;
                return {o, r, c, dr, dc};
              }
            }
          }
          return null;
        })()""")
        check("found illegal/legal anchor pair", pair is not None)
        pt = page.evaluate("""([p]) => {
          const B = window.__qb;
          const a = B.anchorCenter(p.r, p.c);
          const U = B.U, br = document.getElementById('board').getBoundingClientRect();
          return {x: br.left + a.x + p.dr*0.48*U, y: br.top + a.y + p.dc*0.48*U,
                  o: p.o, er: p.r, ec: p.c};
        }""", [pair])
        page.keyboard.press("v" if pt["o"] == 1 else "h"); page.wait_for_timeout(120)
        page.mouse.move(pt["x"], pt["y"]); page.mouse.down(); page.wait_for_timeout(120)
        g = page.evaluate("window.__qb.ghost")
        check("assist snaps off an illegal anchor",
              bool(g) and g.get("state") == "assisted" and g.get("o") == pt["o"])
        page.mouse.up()
        s = wait_reply(4)
        check("assisted press committed somewhere legal",
              s["cur"] == 6 and page.evaluate("window.__w.wallsLeft(humanSide)") == 7)

        # B6: confirm-chip mode (On): pending ghost, Enter commits, ccNo cancels
        page.click("#btnSettings"); page.wait_for_timeout(250)
        page.click("#modalBox [data-set='confirmWalls'] [data-v='true']")
        page.wait_for_timeout(150)
        page.evaluate("closeModal()"); page.wait_for_timeout(150)
        a = anchor_pt(0, 1, 6)
        page.click("#wallH"); page.wait_for_timeout(120)
        page.mouse.move(a["x"], a["y"]); page.mouse.down(); page.mouse.up()
        page.wait_for_timeout(200)
        chip_on = page.is_visible("#confirmChip")
        cur_before = page.evaluate("window.__w.cursor()")
        check("confirm chip shown, wall held", chip_on and
              page.evaluate("(window.__qb.ghost||{}).state") == "pending")
        page.click("#ccNo"); page.wait_for_timeout(200)
        check("ccNo cancels pending wall",
              page.evaluate("window.__w.cursor()") == cur_before)
        page.click("#wallH"); page.wait_for_timeout(120)
        page.mouse.move(a["x"], a["y"]); page.mouse.down(); page.mouse.up()
        page.wait_for_timeout(200)
        page.keyboard.press("Enter")
        s = wait_reply(cur_before)
        check("Enter commits pending wall", s["cur"] == cur_before + 2)
        page.click("#btnSettings"); page.wait_for_timeout(250)
        page.click("#modalBox [data-set='confirmWalls'] [data-v='null']")
        page.evaluate("closeModal()"); page.wait_for_timeout(150)

        # hint draws a suggestion during play
        page.click("#btnHint"); page.wait_for_timeout(700)
        hinted = page.evaluate(
            "window.__qb.ghost !== null || window.__qb.selected >= 0 || window.__qb.dots.length > 0")
        check("hint draws suggestion", hinted)
        page.wait_for_timeout(3600)   # let the hint clear

        # ============ C. SETTINGS VIA REAL CONTROL CLICKS ============
        gold_before = page.evaluate(
            "getComputedStyle(document.documentElement).getPropertyValue('--gold').trim()")
        page.click("#btnSettings"); page.wait_for_timeout(300)
        page.click("#setBoards .swatch[data-b='walnut']")
        page.wait_for_timeout(150)
        check("theme swatch applies", page.evaluate(
            "document.documentElement.dataset.board") == "walnut")
        page.click("#setPawns .swatch[data-b='beacon']")
        page.wait_for_timeout(100)
        check("pawn style swatch applies", page.evaluate(
            "document.documentElement.dataset.pawn") == "beacon")
        page.click("#modalBox [data-set='frame'] [data-v='gilded']")
        page.wait_for_timeout(100)
        check("frame seg applies", page.evaluate(
            "document.documentElement.dataset.frame") == "gilded")
        page.click("#modalBox [data-set='ui'] [data-v='light']")
        page.wait_for_timeout(100)
        check("ui theme light", page.evaluate(
            "document.documentElement.dataset.ui") == "light")
        page.click("#modalBox [data-set='anim'] [data-v='off']")
        page.wait_for_timeout(100)
        check("animations off attr", page.evaluate(
            "document.documentElement.getAttribute('data-anim')") == "off")
        page.click("#modalBox [data-set='anim'] [data-v='full']")
        page.wait_for_timeout(100)
        check("animations full clears attr", page.evaluate(
            "document.documentElement.getAttribute('data-anim')") is None)
        page.click("#modalBox [data-set='handedness'] [data-v='left']")
        page.wait_for_timeout(100)
        check("handedness left", page.evaluate(
            "document.documentElement.dataset.handed") == "left")
        page.click("#modalBox [data-set='fs'] [data-v='1.12']")
        page.wait_for_timeout(100)
        check("text size 112%", page.evaluate(
            "document.documentElement.style.fontSize") == "112%")
        page.click("#setAccents .swatch:not(.on)")
        page.wait_for_timeout(150)
        gold_after = page.evaluate(
            "getComputedStyle(document.documentElement).getPropertyValue('--gold').trim()")
        check("accent swatch changes --gold", gold_after != gold_before)
        page.select_option("#packSel", "modern")
        vol = page.locator("#volSlider")
        vol.fill("30"); vol.dispatch_event("change")
        page.click("#evSeg button[data-ev='walls']")
        page.wait_for_timeout(100)
        persisted = page.evaluate("JSON.parse(localStorage.getItem('zq.settings'))")
        check("settings persist (pack/volume/events/custom)",
              persisted["soundPack"] == "modern" and persisted["volume"] == 0.3 and
              persisted["soundEvents"]["walls"] is False and
              persisted["preset"] == "custom")
        # worker OFF now: analysis below proves the main-thread fallback
        page.click("#modalBox [data-set='worker'] [data-v='false']")
        page.wait_for_timeout(100)
        # a preset restores a coherent bundle over the custom scatter
        page.click("#presetSeg [data-p='classic']"); page.wait_for_timeout(250)
        preset_state = page.evaluate("JSON.parse(localStorage.getItem('zq.settings'))")
        check("classic preset applied", preset_state["preset"] == "classic" and
              preset_state["board"] == "ivory" and preset_state["ui"] == "light")
        page.evaluate("closeModal()"); page.wait_for_timeout(200)

        # ================= D. ANALYSIS DEEP =================
        fresh_game()               # clean opening: the B-section walls may
        click_pawn_to(13)          # have sealed every first destination
        click_pawn_to(22)          # e3 -- a legal continuation from e2
        page.click("#panelTabs .tab[data-pane='anPane']"); page.wait_for_timeout(250)
        page.click("#anEngBtn"); page.wait_for_timeout(400)
        check("engine ON label", "ON" in (page.text_content("#anEngBtn") or ""))
        check("worker NOT used (setting off)", page.evaluate(
            "typeof ANW !== 'undefined' && ANW.ok() && ANW.ready") is False)
        deadline = time.time() + 12
        rows = 0
        while time.time() < deadline:
            rows = page.locator(".pvRow").count()
            if rows >= 1:
                break
            page.wait_for_timeout(250)
        check("PV rows rendered via slicing fallback", rows >= 1)
        check("info nodes counted", "nodes" in (page.text_content("#anInfo") or ""))
        page.select_option("#anPvCount", "3")
        page.wait_for_timeout(3500)
        check("3 PV rows", page.locator(".pvRow").count() >= 3)
        n_rows = min(3, page.locator(".pvRow").count())
        row_txt = [page.locator(".pvRow").nth(i).text_content() or "" for i in range(n_rows)]
        check("PV rows distinct", len(set(row_txt)) == len(row_txt))
        page.locator(".pvRow").nth(1).click(); page.wait_for_timeout(250)
        check("line preview draws", page.evaluate("window.__qb.linePreview !== null"))
        page.mouse.move(box["width"]/2, box["height"] - 8); page.wait_for_timeout(250)
        check("line preview clears on leave",
              page.evaluate("window.__qb.linePreview === null"))
        page.select_option("#anDepth", "0")
        page.wait_for_timeout(1200)
        check("infinite depth marker", "\u221e" in (page.text_content("#anInfo") or ""))
        page.select_option("#anDepth", "10")
        page.wait_for_timeout(800)
        gb = page.locator("#anGraph").bounding_box()
        page.mouse.move(gb["x"] + gb["width"]*0.45, gb["y"] + gb["height"]/2)
        page.mouse.down(); page.mouse.up(); page.wait_for_timeout(400)
        cur_n = page.evaluate("window.__w.plyCount()")
        scrubbed = page.evaluate("window.__w.cursor()")
        check("graph scrub jumps cursor",
              0 <= scrubbed <= cur_n and scrubbed != cur_n)
        page.click("#btnReturn"); page.wait_for_timeout(300)
        check("return to live end", page.evaluate("window.__w.cursor()") ==
              page.evaluate("window.__w.plyCount()"))
        # blunder check + accuracy card
        page.click("#anBlunderBtn")
        t0 = time.time()
        while time.time() - t0 < 60:
            if page.evaluate("document.getElementById('bcBox').style.display === 'none'"):
                break
            page.wait_for_timeout(600)
        check("blunder check finished", page.evaluate(
            "document.getElementById('bcBox').style.display === 'none'"))
        acc = page.text_content("#bcSummary") or ""
        check("accuracy card names both sides", "You" in acc and "Zquoridor" in acc)
        # re-enable the worker, prove the round trip too
        page.click("#btnSettings"); page.wait_for_timeout(250)
        page.click("#modalBox [data-set='worker'] [data-v='true']")
        page.evaluate("closeModal()"); page.wait_for_timeout(200)
        page.click("#anEngBtn"); page.wait_for_timeout(300)   # off
        page.click("#anEngBtn")                               # on again
        wk_ok = False
        t0 = time.time()
        while time.time() - t0 < 15:
            if page.evaluate("typeof ANW !== 'undefined' && ANW.ready"):
                wk_ok = True; break
            page.wait_for_timeout(200)
        check("analysis worker ready after re-enable", wk_ok)
        info2 = ""
        t0 = time.time()
        while time.time() - t0 < 10:
            info2 = page.text_content("#anInfo") or ""
            if "nodes" in info2:
                break
            page.wait_for_timeout(250)
        check("worker analysis updates info", "nodes" in info2)
        s_end = state()
        check("analysis left the game intact", s_end["cur"] == s_end["n"] and
              s_end["turn"] == human() and not s_end["over"])
        page.click("#panelTabs .tab[data-pane='playPane']"); page.wait_for_timeout(200)

        # ================= E. EDITOR MATRIX =================
        page.click("#panelTabs .tab[data-pane='edPane']"); page.wait_for_timeout(300)
        check("editor opens legal",
              "Legal position" in (page.text_content("#edValidity") or ""))
        # move pawn0 to e5 with the pawn tool (click)
        page.click("#etPawn0")
        e5 = cell_pt(40)
        page.mouse.click(e5["x"], e5["y"]); page.wait_for_timeout(150)
        check("editor pawn moved to e5", page.evaluate("window.__w.scrPawn(0)") == 40)
        # canonical 2x2 seal: H (3,4)/(5,4), V (4,3)/(4,5) -- the same shape
        # tests/test_notation.cpp pins as the smallest legal sealed region
        page.click("#etWallH")
        for (r, c) in ((3, 4), (5, 4)):
            p = anchor_pt(0, r, c)
            page.mouse.click(p["x"], p["y"]); page.wait_for_timeout(100)
        page.click("#etWallV")
        for (r, c) in ((4, 3), (4, 5)):
            p = anchor_pt(1, r, c)
            page.mouse.click(p["x"], p["y"]); page.wait_for_timeout(100)
        val = page.text_content("#edValidity") or ""
        check("sealed player 0 refused", "no path" in val.lower() and "player 0" in val.lower())
        check("Apply gated on invalid", page.evaluate(
            "document.getElementById('btnEdApply').disabled") is True)
        # wall-budget steppers vs the 20-wall invariant: with all four box
        # walls up the position is over budget AND sealed -- dropping each
        # hand twice fixes only the budget half (4 + 8 + 8 = 20)
        for _ in range(2):
            page.click("[data-st='w0-1']"); page.wait_for_timeout(60)
            page.click("[data-st='w1-1']"); page.wait_for_timeout(60)
        check("steppers update hand counts",
              (page.text_content("#edW0") or "") == "8" and
              (page.text_content("#edW1") or "") == "8")
        val_mid = (page.text_content("#edValidity") or "").lower()
        check("budget fixed but path still refused",
              "no path" in val_mid and "walls" not in val_mid)
        # side-to-move switch
        page.click("#edTurn button[data-t='1']"); page.wait_for_timeout(120)
        check("side-to-move switch", page.evaluate("window.__w.scrTurn()") == 1)
        page.click("#edTurn button[data-t='0']"); page.wait_for_timeout(120)
        # erase one wall -> the path reopens even though pawn0 sits inside
        page.click("#etErase")
        p = anchor_pt(0, 3, 4)
        page.mouse.click(p["x"], p["y"]); page.wait_for_timeout(150)
        check("erase reopens the path", "Legal position" in
              (page.text_content("#edValidity") or ""))
        # a conflicting editor click is ignored without a crash (overlap)
        page.click("#etWallH")
        p = anchor_pt(0, 5, 4)   # overlaps the surviving H(5,4)
        page.mouse.click(p["x"], p["y"]); page.wait_for_timeout(150)
        check("conflicting editor click ignored", page.evaluate(
            "window.__w.scrWallHBit(5*8+4)") == 1 and
            "Legal position" in (page.text_content("#edValidity") or ""))
        page.click("#btnEdApply"); page.wait_for_timeout(600)
        check("apply starts edited game", page.evaluate("window.__w.plyCount()") == 0 and
              page.evaluate("window.__w.pawn(0)") == 40 and
              page.evaluate("window.__w.wallsLeft(0)") == 8)
        page.click("#panelTabs .tab[data-pane='playPane']"); page.wait_for_timeout(200)

        # ============ F. TEXT I/O / HASH / RESUME / RECENT ============
        fresh_game()
        qfen_live = ""
        click_pawn_to(13)
        qfen_live = page.evaluate("window.__w.qfenExportStr()")
        page.click("#btnMenu"); page.wait_for_timeout(200)
        page.click("#menuDrop button:has-text('Paste game')"); page.wait_for_timeout(300)
        body = page.input_value("#ioArea")
        check("IO opens with QGN export", body.startswith("[Event"))
        page.click("#ioFmt button[data-f='qfen']"); page.wait_for_timeout(150)
        check("fmt switch loads QFEN", page.input_value("#ioArea") == qfen_live)
        # a bad wall token inside the field list keeps the modal open with
        # a diagnostic naming the failing token
        parts = qfen_live.split(" ")
        bad_qfen = " ".join(parts[:4] + ["zz9"] + parts[4:]) or "e2 e9 10 10 zz9 0"
        page.fill("#ioArea", bad_qfen)
        page.click("#ioLoad"); page.wait_for_timeout(300)
        diag = page.text_content("#ioDiag") or ""
        check("bad token diagnostic", ("zz9" in diag or "token" in diag.lower())
              and page.is_visible("#ioArea"))
        # reload the clean QGN captured at open
        page.click("#ioFmt button[data-f='qgn']"); page.wait_for_timeout(150)
        page.fill("#ioArea", body)
        page.click("#ioLoad"); page.wait_for_timeout(400)
        check("QGN reload clean", not page.is_visible("#ioArea") and
              page.evaluate("window.__w.cursor()") >= 2)

        # hash cold load
        page.goto(f"http://localhost:{PORT}/zquoridor.html#qfen={urllib.parse.quote(qfen_live)}")
        page.wait_for_timeout(900)
        check("hash load restores position",
              page.evaluate("window.__w.qfenExportStr()") == qfen_live)
        check("hash page zero errors", len(errors) == 0)

        # autosave resume chip across reload -- first strip the #qfen= hash
        # left by the previous section: a hash load preempts the resume chip
        page.evaluate("history.replaceState({}, '', '/zquoridor.html')")
        fresh_game()
        click_pawn_to(13)
        page.wait_for_timeout(1600)   # autosave debounce
        page.reload(); page.wait_for_timeout(900)
        chip_visible = page.is_visible("#resumeChip")
        check("resume chip offered after reload", chip_visible)
        if chip_visible:
            page.click("#resumeChip"); page.wait_for_timeout(600)
            check("resume restores plies", page.evaluate("window.__w.cursor()") == 2)

        # recent games lifecycle (resign -> sheet -> load -> delete last)
        page.evaluate("localStorage.removeItem('zq.recent')")
        click_pawn_to(22)   # a resigned game needs plies to be recorded
        page.click("#btnMenu"); page.wait_for_timeout(200)
        page.click("#menuDrop button:has-text('Resign')")
        page.wait_for_selector("#cfYes", timeout=4000)
        page.click("#cfYes"); page.wait_for_timeout(500)
        check("resign recorded in recent", page.evaluate(
            "JSON.parse(localStorage.getItem('zq.recent')||'[]').length") == 1)
        page.click("#movesChip"); page.wait_for_timeout(300)
        check("recent sheet lists entry", page.locator("[data-rload='0']").count() == 1)
        page.click("[data-rload='0']"); page.wait_for_timeout(400)
        check("recent Load restores game", page.evaluate("window.__w.cursor()") > 0)
        page.click("#movesChip"); page.wait_for_timeout(300)
        page.click("[data-rdel='0']"); page.wait_for_timeout(300)
        check("delete-last clears list and closes sheet",
              page.evaluate("JSON.parse(localStorage.getItem('zq.recent')||'[]').length") == 0 and
              not page.locator("[data-rload='0']").is_visible())

        # ================= G. FINAL SWEEP =================
        check("ZERO page/console errors overall", len(errors) == 0)
        if errors:
            print("   errors:", errors[-5:])


if __name__ == "__main__":
    sys.exit(main())
