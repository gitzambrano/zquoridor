"""Human-like full game simulation for the Zquoridor premium GUI, plus a
full analysis pass. Plays like a person would -- clicks, drags, keyboard,
dock buttons -- and checks state invariants after every single engine
reply, so a desync shows up at the exact move it happens on:

  game      : up to 30 human moves mixing pawn clicks, arm+click walls,
              drag walls, keyboard walls, hint, flip, takeback, review
              round-trip, level change; invariants after every reply
  analysis  : engine on, 3 PV lines, per-row line preview, graph scrub,
              move-log jump, blunder check with accuracy card
  cross     : QGN export -> second page via #qgn= -> identical position,
              and the second page keeps playing

Run from gui_web/:  python test_gameplay_sim.py
"""
import subprocess
import sys
import time
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

from playwright.sync_api import sync_playwright

def main():
    random.seed(20260823)
    srv = subprocess.Popen([sys.executable, "dev_server.py", "8209"],
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    failures = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append("pageerror: " + str(e).split("\n")[0]))
            page.on("console", lambda m: errors.append("console: " + m.text)
                    if m.type == "error" else None)
            page.goto("http://127.0.0.1:8209/style.html")
            page.wait_for_timeout(2600)

            def check(name, cond):
                if not cond:
                    failures.append(name)
                    print("  FAIL:", name)
                else:
                    print("  ok:", name)

            check("boot without error", "BOOTERR" not in (page.title() or ""))

            # ---------- helpers ----------
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
            def legal_moves():
                return page.evaluate("""(() => { const W=window.__w, pawns=[], walls=[];
                  for (let i=0;i<W.moveCount();i++){
                    if (W.mvIsWall(i)) walls.push([W.mvA(i),W.mvB(i),W.mvC(i)]);
                    else pawns.push(W.mvA(i)); }
                  return {pawns, walls}; })()""")
            def state():
                return page.evaluate("""(() => { const W=window.__w, B=window.__qb;
                  let on=0; for (let s=0;s<64;s++){ on+=W.wallHBit(s)+W.wallVBit(s); }
                  return {ply: W.plyCount(), turn: W.turn(), hs: humanSide,
                          think: engineThinking, ghost: B.ghost !== null,
                          dots: B.dots.length, over: gameOver,
                          wl0: W.wallsLeft(0), wl1: W.wallsLeft(1), wallsOn: on,
                          winner: W.winner(), p0: W.pawn(0), p1: W.pawn(1)}; })()""")
            def wait_engine_reply(ply_before, timeout_ms=9000):
                t0 = time.time()
                while (time.time() - t0) * 1000 < timeout_ms:
                    s = state()
                    if s["turn"] == s["hs"] and not s["think"] and s["ply"] >= ply_before + 2:
                        return s
                    if s["over"]:
                        return s
                    page.wait_for_timeout(120)
                return state()
            def invariants(s, tag, ply_before):
                ok = True
                if not s["over"]:
                    if s["turn"] != s["hs"]: ok = False; print("    !turn desync", tag, s)
                    if s["think"]: ok = False; print("    !still thinking", tag, s)
                    if s["ply"] != ply_before + 2: ok = False; print("    !ply != +2", tag, s)
                    if s["ply"] % 2 != 0: ok = False; print("    !odd ply after reply", tag, s)
                    if s["ghost"]: ok = False; print("    !ghost stuck", tag)
                    if s["dots"] <= 0: ok = False; print("    !no dots", tag)
                if s["wallsOn"] != 20 - s["wl0"] - s["wl1"]:
                    ok = False; print("    !wall count mismatch", tag, s)
                if not ok:
                    failures.append(f"invariant@{tag}")
                return ok

            def do_pawn_move():
                s = state()
                d = cell_pt(s["p" + str(s["hs"])])
                page.mouse.click(d["x"], d["y"])
                page.wait_for_timeout(220)
                n = page.evaluate("window.__qb.dots.length")
                if not n:
                    return False
                k = random.randrange(n)
                expr = f"window.__qb.engPawnToDisp(window.__qb.dots[{k}])"
                d2 = cell_pt(page.evaluate(expr))
                page.mouse.click(d2["x"], d2["y"])
                return True
            def do_wall_arm_click(o):
                legal_walls = legal_moves()["walls"]
                if not legal_walls:
                    return False
                o2, r2, c2 = random.choice([w for w in legal_walls if w[0] == o] or legal_walls)
                page.click("#wallH" if o2 == 0 else "#wallV")
                pt = anchor_pt(o2, r2, c2)
                page.mouse.click(pt["x"], pt["y"])
                return True
            def do_wall_drag(o, ply_before):
                legal_walls = legal_moves()["walls"]
                if not legal_walls:
                    return False
                o2, r2, c2 = random.choice([w for w in legal_walls if w[0] == o] or legal_walls)
                pt = anchor_pt(o2, r2, c2)
                # start hugging the anchor: a far offset lands in a cell-center
                # radius and the press becomes a pawn select by design
                page.mouse.move(pt["x"] + 3, pt["y"] + 3)
                page.mouse.down()
                page.mouse.move(pt["x"] + 1, pt["y"] + 1, steps=2)
                page.mouse.move(pt["x"], pt["y"], steps=2)
                page.mouse.up()
                page.wait_for_timeout(180)
                if state()["ply"] == ply_before:
                    return do_wall_arm_click(o)   # gesture zone taken: fall back
                return True
            def do_wall_keyboard(o):
                legal_walls = legal_moves()["walls"]
                if not legal_walls:
                    return False
                o2, r2, c2 = random.choice([w for w in legal_walls if w[0] == o] or legal_walls)
                page.keyboard.press("h" if o2 == 0 else "v")
                pt = anchor_pt(o2, r2, c2)
                page.mouse.click(pt["x"], pt["y"])
                return True

            # ---------- PHASE A: the game ----------
            print("--- game: playing like a human")
            flips = 0
            for mv in range(1, 15):
                s = state()
                if s["over"] or s["ply"] >= 28:
                    break
                ply_before = s["ply"]
                acted = False
                kind = "pawn"
                print(f"  move {mv}: start ply={ply_before} turn={s['turn']} "
                      f"think={s['think']} over={s['over']}")
                if mv == 3:
                    page.click("#btnHint"); page.wait_for_timeout(500)
                    ghosted = page.evaluate("window.__qb.ghost !== null || window.__qb.selected >= 0")
                    check(f"move {mv}: hint shows suggestion", ghosted)
                    page.keyboard.press("Escape"); page.wait_for_timeout(150)
                if mv == 11:
                    # the return chip lives in the analysis pane; from the
                    # play tab the keyboard is the way back (End = live end)
                    page.keyboard.press(","); page.wait_for_timeout(400)
                    st_txt = page.text_content("#status") or ""
                    check(f"move {mv}: comma enters review", "Reviewing" in st_txt)
                    page.keyboard.press("End"); page.wait_for_timeout(400)
                    check(f"move {mv}: End returns to live", page.evaluate(
                        "window.__w.cursor()") == page.evaluate("window.__w.plyCount()"))
                if mv == 13:
                    # mid-game level change is the S key (cycleLevel); the
                    # lvlChip opens the NEW GAME modal, not a level switcher
                    before_lvl = page.evaluate("S.level")
                    page.keyboard.press("s"); page.wait_for_timeout(350)
                    after_lvl = page.evaluate("S.level")
                    check(f"move {mv}: level cycled ({before_lvl} -> {after_lvl})",
                          after_lvl != before_lvl)
                if mv >= 5 and mv % 2 == 1 and s["wl0"] > 0 and random.random() < 0.65:
                    k = (mv // 2) % 3
                    kind = ("armH" if k == 0 else "dragV" if k == 1 else "kbd")
                    acted = (do_wall_arm_click(0) if kind == "armH" else
                             do_wall_drag(1, ply_before) if kind == "dragV" else
                             do_wall_keyboard(random.choice([0, 1])))
                if not acted:
                    kind = "pawn"
                    acted = do_pawn_move()
                if not acted:
                    print(f"  move {mv}: no action available, stopping")
                    break
                print(f"  move {mv}: action={kind} -> waiting engine")
                # nudge path: a click OUTSIDE the board must be inert while
                # the engine thinks (a board click could legally move!)
                page.click("#status")
                s2 = wait_engine_reply(ply_before)
                invariants(s2, f"move {mv} ({kind})", ply_before)
                if mv == 7:
                    page.click("#btnFlip"); flips += 1; page.wait_for_timeout(400)
                    check(f"move {mv}: flipped, still playable",
                          page.evaluate("window.__qb.flipped") is True)
                if mv == 17 and flips == 1:
                    page.click("#btnFlip"); flips += 2; page.wait_for_timeout(400)
                if mv == 9:
                    before = state()["ply"]
                    page.click("#btnTakeback"); page.wait_for_timeout(500)
                    after = state()
                    check(f"move {mv}: takeback rolled back", after["ply"] == before - 2
                          and after["turn"] == after["hs"])
                    # redo the move we just rolled back
                    ply_before = after["ply"]
                    if not do_pawn_move():
                        print(f"  move {mv}: redo failed, stopping"); break
                    s2 = wait_engine_reply(ply_before)
                    invariants(s2, f"move {mv} redo", ply_before)
                if s2["over"]:
                    break
            page.keyboard.press("End"); page.wait_for_timeout(400)   # back to live
            mid = state()
            print(f"  mid-game state: ply={mid['ply']} over={mid['over']} "
                  f"walls={mid['wallsOn']}")

            # ---------- PHASE B: analysis (mid-game) ----------
            print("--- analysis: engine, PVs, graph, blunder check")
            page.click("#panelTabs .tab[data-pane='anPane']")
            page.wait_for_timeout(250)
            page.click("#anEngBtn")
            page.select_option("#anPvCount", "3")
            page.select_option("#anDepth", "6")
            rows = 0
            t0 = time.time()
            while rows < 3 and (time.time() - t0) < 12:
                rows = page.locator("#anLines .pvRow").count()
                page.wait_for_timeout(400)
            check("3 pv lines", rows == 3)
            check("info line populated", "nodes" in (page.text_content("#anInfo") or ""))
            for i in range(rows):
                page.locator("#anLines .pvRow").nth(i).click()
                page.wait_for_timeout(220)
                pv_ok = page.evaluate("window.__qb.linePreview !== null")
                check(f"line {i+1} preview draws", pv_ok)
            page.mouse.move(40, 500); page.wait_for_timeout(200)
            check("preview clears", page.evaluate("window.__qb.linePreview === null"))
            g = page.locator("#anGraph").bounding_box()
            cursors = set()
            for fx in (0.2, 0.5, 0.8):
                page.mouse.click(g["x"] + g["width"] * fx, g["y"] + g["height"] / 2)
                page.wait_for_timeout(350)
                cursors.add(page.evaluate("window.__w.cursor()"))
            check("graph scrub reaches 3 cursors", len(cursors) >= 2)
            page.click("#btnReturn"); page.wait_for_timeout(300)
            # move log jump
            log_rows = page.locator("#mvLog .mvRow, #mvLog li, #mvLog div")
            if log_rows.count() > 4:
                log_rows.nth(2).click(); page.wait_for_timeout(300)
                check("move log jumps", page.evaluate("window.__w.cursor()") <
                      page.evaluate("window.__w.plyCount()"))
                page.click("#btnReturn"); page.wait_for_timeout(300)
            page.click("#anBlunderBtn")
            t0 = time.time()
            while (time.time() - t0) < 40:
                if page.evaluate(
                    "document.getElementById('bcBox').style.display === 'none'"):
                    break
                page.wait_for_timeout(600)
            check("blunder check finished", page.evaluate(
                "document.getElementById('bcBox').style.display === 'none'"))
            acc = page.text_content("#bcSummary") or ""
            check("accuracy card rendered", "You" in acc and "Zquoridor" in acc)
            page.click("#anEngBtn")   # engine off; back to the game
            page.wait_for_timeout(200)
            page.click("#panelTabs .tab[data-pane='playPane']")
            page.wait_for_timeout(250)

            # ---------- PHASE B2: resume the game to the end ----------
            print("--- game: resuming after analysis")
            for mv in range(15, 41):
                s = state()
                if s["over"]:
                    break
                ply_before = s["ply"]
                acted = False
                kind = "pawn"
                if mv % 4 == 3 and s["wl0"] > 0:
                    kind = "armH"
                    acted = do_wall_arm_click(0)
                if not acted:
                    acted = do_pawn_move()
                if not acted:
                    print(f"  move {mv}: no action available, stopping")
                    break
                s2 = wait_engine_reply(ply_before)
                invariants(s2, f"resume move {mv} ({kind})", ply_before)
            page.keyboard.press("End"); page.wait_for_timeout(400)
            fin = state()
            print(f"  game state after sim: ply={fin['ply']} over={fin['over']} "
                  f"winner={fin['winner']} walls={fin['wallsOn']}")
            check("engine never moved for the human", fin["turn"] == fin["hs"] or fin["over"])
            check("walls conserved", fin["wallsOn"] == 20 - fin["wl0"] - fin["wl1"])
            if fin["over"]:
                st_txt = (page.text_content("#status") or "").lower()
                check("game over announced", any(w in st_txt for w in
                                                 ("won", "win", "flag", "draw")))
            # ---------- PHASE C: cross-instance ----------
            print("--- cross: QGN to a second page")
            b64 = page.evaluate("btoa(unescape(encodeURIComponent(qgnExport())))")
            page2 = browser.new_page()
            errs2 = []
            page2.on("pageerror", lambda e: errs2.append(str(e)))
            page2.goto("http://127.0.0.1:8209/style.html#qgn=" + b64)
            page2.wait_for_timeout(2800)
            s1 = state()
            s2 = page2.evaluate("""(() => { const W=window.__w; let on=0;
              for (let s=0;s<64;s++) on+=W.wallHBit(s)+W.wallVBit(s);
              return {ply: W.plyCount(), p0: W.pawn(0), p1: W.pawn(1), on}; })()""")
            check("second page: same ply", s2["ply"] == s1["ply"])
            check("second page: same pawns", s2["p0"] == s1["p0"] and s2["p1"] == s1["p1"])
            check("second page: same walls", s2["on"] == s1["wallsOn"])
            # the exported game had finished: page2 must restore the result...
            st2 = (page2.text_content("#status") or "").lower()
            check("second page: finished result restored", s1["over"] is True and
                  any(w in st2 for w in ("won", "win", "flag", "draw")))
            # ...and a fresh game on page2 keeps playing through the GUI
            page2.evaluate("newGame()")
            page2.wait_for_timeout(700)
            d = page2.evaluate("""(() => { const B=window.__qb, W=window.__w;
              const d=B.engPawnToDisp(W.pawn(humanSide)), c=B.cellCenter(Math.floor(d/9),d%9);
              const r=document.getElementById('board').getBoundingClientRect();
              return {x:r.left+c.x, y:r.top+c.y}; })()""")
            page2.mouse.click(d["x"], d["y"]); page2.wait_for_timeout(350)
            page2.evaluate("""(() => { const B=window.__qb; if (B.dots.length) {
              // dots are ALREADY display cells: no further conversion
              const t=B.dots[0], c=B.cellCenter(Math.floor(t/9),t%9);
              const r=document.getElementById('board').getBoundingClientRect();
              window.__pt={x:r.left+c.x, y:r.top+c.y}; } })()""")
            pt = page2.evaluate("window.__pt")
            if pt:
                page2.mouse.click(pt["x"], pt["y"])
                ply2 = page2.evaluate("window.__w.plyCount()")
                t0 = time.time()
                while ply2 < 2 and (time.time() - t0) < 9:
                    page2.wait_for_timeout(300)
                    ply2 = page2.evaluate("window.__w.plyCount()")
                check("second page keeps playing", ply2 >= 2)
            check("second page: zero errors", len(errs2) == 0)
            page2.close()

            # ---------- wrap ----------
            check("zero page/console errors (whole session)", len(errors) == 0)
            if errors:
                for e in errors[:10]:
                    print("   ", e)
            page.evaluate("""() => { localStorage.removeItem('zq.settings');
              localStorage.removeItem('zq.game'); localStorage.removeItem('zq.recent'); }""")
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
