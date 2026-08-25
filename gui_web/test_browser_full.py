"""Full browser acceptance run for the Zquoridor premium GUI (gui-premium.md
sections 15/16/17 spot checks). Exercises every tab and the major functions:

  play      : human pawn move, wall arm+click placement, hint, takeback,
              resign confirm modal
  analysis  : engine toggle (worker), PV rows, info line, eval graph,
              navigation + return-to-game, blunder check
  editor    : tools, steppers, validity strip messages, apply -> new game
  text i/o  : QGN export headers, QFEN round trip, dialect imports
  settings  : theme/accent/preset live application, sound pack select
  misc      : keyboard map, hash load, recent games, image export

Run from gui_web/:  python test_browser_full.py
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))

from playwright.sync_api import sync_playwright

def main():
    srv = subprocess.Popen([sys.executable, "dev_server.py", "8201"],
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    failures = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(
                (str(e).split("\n")[0] + " || " +
                 (e.stack or "").split("\n")[1] if getattr(e, "stack", None) else str(e))))
            page.goto("http://127.0.0.1:8201/style.html")
            page.wait_for_timeout(2600)

            def check(name, cond):
                if not cond:
                    failures.append(name)
                    print("FAIL:", name)
                else:
                    print("ok:", name)

            check("boot without error", "BOOTERR" not in (page.title() or ""))
            status = page.text_content("#status") or ""
            check("engine booted", "Loading" not in status)

            # ---------- PLAY ----------
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

            # pawn move via UI clicks: select pawn, click destination
            src = cell_pt(page.evaluate("window.__w.pawn(0)"))
            dst = cell_pt(13)   # e2
            page.mouse.click(src["x"], src["y"])
            page.mouse.click(dst["x"], dst["y"])
            page.wait_for_timeout(2800)
            check("pawn moved by UI", page.evaluate("window.__w.cursor()") >= 2)

            # wall placement M2 (arm then tap a legal groove) on a fresh turn
            check("human to move before wall", page.evaluate("window.__w.turn()") == 0)
            n_before = page.evaluate("window.__w.plyCount()")
            page.click("#wallH")
            page.wait_for_timeout(150)
            check("wall armed", "armed" in (page.get_attribute("#wallH", "class") or ""))
            pt = anchor_pt(0, 3, 3)
            page.mouse.click(pt["x"], pt["y"])
            page.wait_for_timeout(600)
            placed = page.evaluate("window.__w.plyCount()") > n_before
            check("wall placed via dock", placed)
            if not placed:
                page.keyboard.press("Escape")
                page.evaluate("window.__w.truncateHistory(%d)" % n_before)
            page.wait_for_timeout(2600)   # let the engine answer the wall

            # hint draws something and clears later
            page.evaluate("newGame()")
            page.wait_for_timeout(200)
            page.evaluate("newGame()")
            page.wait_for_timeout(400)
            page.click("#btnHint")
            page.wait_for_timeout(600)
            ghosted = page.evaluate("window.__qb.ghost !== null || window.__qb.selected >= 0")
            check("hint shows suggestion", ghosted)

            # takeback removes plies
            page.evaluate("window.__w.applyPawn(13); afterHumanMove();")
            page.wait_for_timeout(2800)
            before = page.evaluate("window.__w.plyCount()")
            page.click("#btnTakeback")
            page.wait_for_timeout(500)
            check("takeback", page.evaluate("window.__w.plyCount()") < before)

            # resign opens confirm modal; cancel keeps game
            page.click("#btnMenu")
            try:
                page.wait_for_selector("#menuDrop.open", timeout=3000)
            except Exception:
                pass
            page.click("#menuDrop button:has-text('Resign')")
            shown = True
            try:
                page.wait_for_selector("#cfYes", timeout=4000)
            except Exception:
                shown = False
            check("resign confirm shown", shown)
            if not shown:
                dbg = page.evaluate("""() => ({
                  overlay: document.getElementById('overlay').className,
                  menu: document.getElementById('menuDrop').className,
                  status: document.getElementById('status').textContent,
                  ply: window.__w.plyCount(),
                  over: typeof gameOver !== 'undefined' ? gameOver : null })""")
                print("   resign debug:", dbg)
            if shown:
                page.click("#cfNo")
                page.wait_for_timeout(250)
                cancelled = not page.is_visible("#overlay")
                check("resign cancelled", cancelled)

            # fresh plies so analysis has something to chew on
            page.evaluate("window.__w.applyPawn(13); afterHumanMove();")
            page.wait_for_timeout(2800)
            check("plies for analysis >= 2",
                  page.evaluate("window.__w.plyCount()") >= 2)

            # ---------- ANALYSIS ----------
            page.click("#panelTabs .tab[data-pane='anPane']")
            page.wait_for_timeout(200)
            page.click("#anEngBtn")
            worker_ok = False
            for _ in range(40):
                if page.evaluate("typeof ANW !== 'undefined' && (ANW.ready || ANW.failed)"):
                    worker_ok = True
                    break
                page.wait_for_timeout(200)
            check("worker state resolved", worker_ok)
            page.wait_for_timeout(2500)
            rows = page.locator(".pvRow").count()
            check("pv rows rendered", rows >= 1)
            check("info line filled", "nodes" in (page.text_content("#anInfo") or ""))
            check("graph drew", page.evaluate(
                "document.getElementById('anGraph').width > 0"))
            cur = page.evaluate("window.__w.cursor()")
            page.click("#navPrev"); page.wait_for_timeout(350)
            check("nav prev", page.evaluate("window.__w.cursor()") == cur - 1)
            check("return chip", page.is_visible("#btnReturn"))
            page.click("#navFirst"); page.wait_for_timeout(300)
            check("nav first", page.evaluate("window.__w.cursor()") == 0)
            page.click("#btnReturn"); page.wait_for_timeout(300)
            check("nav end", page.evaluate("window.__w.cursor()") ==
                  page.evaluate("window.__w.plyCount()"))
            page.select_option("#anDepth", "6")
            page.click("#anBlunderBtn")
            page.wait_for_timeout(9000)
            check("blunder check finished", page.evaluate(
                "document.getElementById('bcBox').style.display === 'none'"))

            # ---------- EDITOR ----------
            page.click("#panelTabs .tab[data-pane='edPane']")
            page.wait_for_timeout(250)
            check("validity ok initially", "Legal position" in (
                page.text_content("#edValidity") or ""))
            # place pawn1 somewhere via tool
            page.click("#etPawn1")
            free = cell_pt(40)   # e5
            page.mouse.click(free["x"], free["y"])
            page.wait_for_timeout(150)
            check("editor pawn1 moved", page.evaluate(
                "window.__w.scrPawn(1)") == 40)
            # wall tool + conflict toast path: place H then crossing V refused
            page.click("#etWallH")
            wpt = anchor_pt(0, 3, 3)
            page.mouse.click(wpt["x"], wpt["y"]); page.wait_for_timeout(100)
            check("editor wall placed", page.evaluate(
                "window.__w.scrWallHBit(3*8+3)") == 1)
            page.click("#etWallV")
            page.mouse.click(wpt["x"], wpt["y"]); page.wait_for_timeout(150)
            check("crossing refused", page.evaluate(
                "window.__w.scrWallVBit(3*8+3)") == 0)
            # budget flag: set both hands to 10 (already), walls=1 -> over 20
            check("budget validity flagged", "walls" in (
                page.text_content("#edValidity").lower()))
            page.click("#etErase")
            page.mouse.click(wpt["x"], wpt["y"]); page.wait_for_timeout(100)
            check("erase removes wall", page.evaluate(
                "window.__w.scrWallHBit(3*8+3)") == 0)
            check("validity back to legal", "Legal position" in (
                page.text_content("#edValidity") or ""))
            click_res = page.evaluate("""() => {
              const b = document.getElementById('btnEdApply');
              if (b.disabled) return 'disabled';
              try { b.click(); return 'clicked'; } catch(e){ return 'EXC '+String(e); }
            }""")
            page.wait_for_timeout(400)
            print("   [apply]", click_res,
                  "| ply:", page.evaluate("window.__w.plyCount()"),
                  "| errs:", errors[-3:])
            check("apply starts fresh game", page.evaluate(
                "window.__w.plyCount()") == 0)
            if errors:
                print("   errors so far:", errors)

            # ---------- TEXT I/O ----------
            page.click("#panelTabs .tab[data-pane='playPane']")
            page.wait_for_timeout(150)
            page.evaluate("openTextIO('qgn')")
            page.wait_for_timeout(250)
            body = page.input_value("#ioArea")
            check("qgn export has headers", "[Event" in body and "[Result" in body)
            page.fill("#ioArea", body)
            page.click("#ioLoad")
            page.wait_for_timeout(400)
            reloaded = not page.is_visible("#overlay")
            check("qgn reloads clean", reloaded)
            if not reloaded:
                print("   io diag:", repr(page.text_content("#ioDiag")))
                print("   errs now:", errors[-3:])
                page.evaluate("closeModal()")
                page.wait_for_timeout(150)

            # QFEN round trip through C surface + UI
            qfen = page.evaluate("window.__w.qfenExportStr()")
            page.evaluate("(q) => routeImport(q)", qfen)
            page.wait_for_timeout(300)
            prefix = " ".join(qfen.split(" ")[:4])
            check("qfen reimport identical", page.evaluate(
                "(q) => window.__w.qfenExportStr().indexOf(q)===0", prefix))

            # dialects: orientation-first + numbering + bare list
            page.evaluate("newGame()")
            ok = page.evaluate("""() => {
              try { return importQGN('1. e2 e8 2. e3 Ha5') } catch(e) { return 'EXC '+e }
            }""")
            page.wait_for_timeout(300)
            plies = page.evaluate("window.__w.plyCount()")
            check("dialect qgn imported", ok is True and plies == 4)
            wall_ok = page.evaluate(
                "window.__w.plyIsWall(3) && window.__w.plyA(3)===0 && window.__w.plyB(3)===4 && window.__w.plyC(3)===0")
            check("Ha5 parsed as H slot a5", bool(wall_ok))
            pair_ok = page.evaluate("""() => {
              newGame();
              try { return importQGN('c6-d6') } catch(e){ return false } }""")
            pair_wall = page.evaluate(
                "window.__w.plyCount()===1 && window.__w.plyIsWall(0) && window.__w.plyB(0)===5 && window.__w.plyC(0)===2")
            check("coordinate-pair wall imported", bool(pair_ok) and pair_wall)
            bad = page.evaluate("importQGN('1. z9 e8')")
            check("bad token rejected", bad is False)
            page.evaluate("closeModal()")   # importFailed reopens Text I/O
            page.wait_for_timeout(150)

            # ---------- SETTINGS ----------
            page.click("#btnSettings")
            page.wait_for_timeout(300)
            page.evaluate("""() => { const b=[...document.querySelectorAll('#setBoards .swatch')].find(x=>x.dataset.b==='marble'); b.click(); }""")
            page.wait_for_timeout(200)
            check("theme setting saved", page.evaluate(
                "(JSON.parse(localStorage.getItem('zq.settings')).board)") == "marble")
            gold_before = page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--gold').trim()")
            page.evaluate("""() => { [...document.querySelectorAll('#setAccents .swatch')].find(x=>x.dataset.a==='jade').click(); }""")
            page.wait_for_timeout(150)
            gold_after = page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--gold').trim()")
            check("accent changes --gold", gold_before != gold_after)
            preset_ok = page.evaluate("""() => { applyPreset('highContrast');
              return JSON.parse(localStorage.getItem('zq.settings')).board === 'noir'; }""")
            check("preset highContrast -> noir", preset_ok)
            check("preset flips custom on edit", page.evaluate(
                """() => { setOpt('board','slate'); return JSON.parse(localStorage.getItem('zq.settings')).preset==='custom'; }"""))
            page.evaluate("applyPreset('premiumDark')")
            page.keyboard.press("Escape")

            # sound pack select persists
            page.click("#btnSettings"); page.wait_for_timeout(250)
            page.select_option("#packSel", "modern")
            page.evaluate("document.querySelector('[data-close]').click()")
            check("soundPack persisted", page.evaluate(
                "JSON.parse(localStorage.getItem('zq.settings')).soundPack") == "modern")

            # keyboard help
            page.keyboard.press("?")
            page.wait_for_timeout(200)
            check("kbd help opens", page.locator(".kbdRow").count() >= 10)
            page.keyboard.press("Escape")
            # nav keys
            page.evaluate("newGame(); window.__w.applyPawn(13); afterHumanMove();")
            page.wait_for_function("() => !engineThinking", timeout=20000)
            page.wait_for_timeout(300)
            page.keyboard.press(",")
            page.wait_for_timeout(200)
            cur2 = page.evaluate("window.__w.cursor()")
            page.keyboard.press(".")
            page.wait_for_timeout(200)
            check("comma/dot nav", page.evaluate("window.__w.cursor()") == cur2 + 1)

            # recent games: force a finished game into the ring by walking
            # both pawns to alternating legal steps until someone lands
            fin = page.evaluate("""() => {
              newGame();
              const seq = { 0: [13,22,31,40,49,58,67,76],      // e2..e9 (win)
                            1: [77,76,77,76,77,76,77] };       // f9/e9 shuffle
              const idx = { 0: 0, 1: 0 };
              let guard = 0;
              while (window.__w.winner() === -1 && guard++ < 40) {
                const t = window.__w.turn();
                if (idx[t] >= seq[t].length) break;
                if (!window.__w.applyPawn(seq[t][idx[t]])) break;
                idx[t]++;
              }
              return window.__w.winner();
            }""")
            check("fast-forward reaches win", fin in (0, 1))
            page.evaluate("checkEnd()")
            rec = page.evaluate("recentList().length")
            check("finished game recorded", rec >= 1)

            # image export modal triggers downloads without errors
            page.evaluate("exportImageModal()")
            page.wait_for_timeout(200)
            with page.expect_download(timeout=8000) as dl:
                page.click("#exSvg")
            dname = dl.value.suggested_filename
            check("svg download", dname.endswith('.svg'))
            page.evaluate("exportImageModal()")   # svg handler closed the modal
            page.wait_for_timeout(200)
            with page.expect_download(timeout=8000) as dl2:
                page.click("#exPng")
            check("png download", dl2.value.suggested_filename.endswith('.png'))

            # ---------- WAVE 2: remaining functions ----------
            # keyboard arrow pawn movement
            page.evaluate("newGame()")
            page.wait_for_timeout(400)
            page.keyboard.press("ArrowUp")
            page.wait_for_timeout(300)
            arrow_ok = page.evaluate(
                "window.__w.plyCount() >= 1 && window.__w.plyA(0) === 13")
            check("arrow moves pawn e2", arrow_ok)
            if not arrow_ok:
                print("   arrow debug:", page.evaluate(
                    "({turn: window.__w.turn(), hs: humanSide,"
                    " liveEnd: atLiveEnd(), think: engineThinking,"
                    " over: gameOver, ply: window.__w.plyCount(),"
                    " focus: document.activeElement ?"
                    " document.activeElement.tagName : null})"))
            page.wait_for_timeout(2400)

            # flip board + paths toggle
            page.keyboard.press("f")
            page.wait_for_timeout(150)
            flip_ok = page.evaluate("window.__qb.flipped") is True
            check("flip sets B.flipped", flip_ok)
            if not flip_ok:
                print("   flip debug:", page.evaluate(
                    "({flipped: window.__qb.flipped, focus:"
                    " document.activeElement ? document.activeElement.tagName : null,"
                    " over: gameOver, think: engineThinking})"))
            page.evaluate("window.__qb.flipped=false")
            page.click("#btnPaths")
            page.wait_for_timeout(150)
            check("paths overlay drawn", page.evaluate(
                "Array.isArray(window.__qb.paths) && window.__qb.paths.length === 2"))
            page.click("#btnPaths")

            # M3 direct board gesture: press near anchor, clear horizontal drag
            n0 = page.evaluate("window.__w.plyCount()")
            pt = anchor_pt(0, 4, 4)
            page.mouse.move(pt["x"], pt["y"])
            page.mouse.down()
            page.mouse.move(pt["x"] + 60, pt["y"], steps=8)
            page.mouse.up()
            page.wait_for_timeout(600)
            m3 = page.evaluate("window.__w.plyCount()") > n0
            check("M3 direct gesture wall", m3)
            if not m3:
                page.keyboard.press("Escape")
                page.evaluate(f"window.__w.truncateHistory({n0})")
            page.wait_for_timeout(2600)

            # confirm-walls mode: pending ghost + Enter commits.
            # First guarantee: human to move, cursor at live end.
            page.evaluate("""() => {
              const W = window.__w;
              for (let p = Math.min(W.cursor(), W.plyCount()); p >= 0; p--) {
                W.scratchFromPly(p);
                if (W.scrTurn() === humanSide) { W.truncateHistory(p); break; }
              }
              gameOver = false;
            }""")
            page.wait_for_timeout(400)
            page.evaluate("S.confirmWalls = true")
            n1 = page.evaluate("window.__w.plyCount()")
            armed = page.evaluate("(function(){ armWall(1); return forcedO === 1; })()")
            check("wallV armed for confirm", armed)
            pt2 = anchor_pt(1, 2, 2)
            page.mouse.click(pt2["x"], pt2["y"])
            page.wait_for_timeout(250)
            chip_on = page.is_visible("#confirmChip")
            check("confirm chip shown", chip_on)
            if chip_on:
                page.keyboard.press("Enter")
                page.wait_for_timeout(500)
                check("Enter commits pending wall",
                      page.evaluate("window.__w.plyCount()") > n1)
            page.evaluate("S.confirmWalls = null")
            page.wait_for_timeout(2200)

            # clock: 5+0 mode ticks and flag logic initializes
            page.evaluate("S.clockMode='5+0'; S.baseMin=5; startClock(); newGame()")
            page.wait_for_timeout(700)
            clk = page.text_content("#hudBottom .clock")
            check("clock ticking", clk not in ("--:--", ""))
            page.evaluate("stopClock(); S.clockMode='none'")

            # level separator in move log on mid-game change
            page.evaluate("newGame()")
            page.wait_for_timeout(250)
            page.evaluate("window.__w.applyPawn(13); afterHumanMove();")
            page.wait_for_timeout(2600)
            page.evaluate("setOpt('level', 'knight'); setLevelMidGame('sage')")
            seps = page.locator(".mlSep").count()
            check("level separator rendered", seps >= 1)
            if seps < 1:
                print("   sep debug:", page.evaluate(
                    "({marks: levelMarks.length, level: S.level,"
                    " ply: window.__w.plyCount(),"
                    " logHtml: document.getElementById('moveLog').innerHTML.slice(0,120)})"))

            # explicit worker round-trip probe (independent of UI rows)
            wres = page.evaluate("""() => new Promise(res => {
              const ok = ANW.analyze({moves: [], depth: 4, timeMs: 300, lines: 1},
                r => res(r ? {lines: r.lines.length, nodes: r.nodes} : null));
              if (!ok) res(null);
              setTimeout(() => res('timeout'), 8000);
            })""")
            check("worker analyze round trip", bool(wres) and wres != 'timeout'
                  and wres.get("lines", 0) >= 1)

            # blunder-check cancel path stops without finishing
            page.click("#panelTabs .tab[data-pane='anPane']")
            page.wait_for_timeout(200)
            page.evaluate("AN.scores={}; AN.annots={};")
            page.select_option("#anDepth", "10")
            page.evaluate("blunderCheck()")
            page.wait_for_timeout(1200)
            page.click("#anBlunderBtn")   # second click cancels
            page.wait_for_timeout(1200)
            bc_ok = page.evaluate("AN.bcRun") is False
            check("blunder cancel works", bc_ok)
            if not bc_ok:
                print("   bc debug:", page.evaluate(
                    "({run: AN.bcRun, cancel: AN.bcCancel,"
                    " box: document.getElementById('bcBox').style.display,"
                    " label: document.getElementById('bcLabel').textContent})"))

            # eval bar off hides the strip
            page.evaluate("setOpt('evalBar', false)")
            page.wait_for_timeout(150)
            check("evalBar off hides strip", page.evaluate(
                "document.getElementById('evalStrip').style.display") == "none")
            page.evaluate("setOpt('evalBar', true)")

            # handedness mirrors dock direction
            page.evaluate("setOpt('handedness', 'left')")
            check("handedness attr", page.evaluate(
                "document.documentElement.dataset.handed") == "left")
            page.evaluate("setOpt('handedness', 'right')")

            # anim off attribute + duration scaling
            page.evaluate("setOpt('anim', 'off')")
            check("anim off attribute", page.evaluate(
                "document.documentElement.hasAttribute('data-anim')"))
            page.evaluate("setOpt('animSpeed', 1.5)")
            check("anim speed scales --dur", page.evaluate(
                "document.documentElement.style.getPropertyValue('--dur')") == "120ms")
            page.evaluate("setOpt('anim', 'full'); setOpt('animSpeed', 1)")

            # resume chip from seeded autosave (fresh context page)
            import json as _sj
            qgn_sample = page.evaluate("qgnExport()")
            ctx = browser.new_context()
            pg = ctx.new_page()
            pg.add_init_script(
                "localStorage.setItem('zq.game', %s);" % _sj.dumps(qgn_sample))
            pg.add_init_script("""
                localStorage.setItem('zq.game',
                  localStorage.getItem('zq.game').replace(/\\[Result "[^"]*"\\]\\n?/, ''));
            """)
            pg.goto("http://127.0.0.1:8201/style.html")
            pg.wait_for_timeout(2600)
            visible = pg.is_visible("#resumeChip")
            check("resume chip offered", visible)
            if visible:
                pg.click("#resumeChip")
                pg.wait_for_timeout(600)
                check("resume loads plies", pg.evaluate(
                    "document.getElementById('resumeChip').style.display") == "none")
            ctx.close()

            # recent games sheet: delete row works
            page.evaluate("showRecentGames()")
            page.wait_for_timeout(300)
            before_rows = page.locator("#modalBox .mlRow").count()
            check("recent list non-empty", before_rows >= 1)
            if before_rows:
                page.click("[data-rdel='0']")
                page.wait_for_timeout(350)
                still_open = page.is_visible("#modalBox")
                if before_rows == 1:
                    # deleting the last entry closes the sheet with a toast
                    check("recent delete closes sheet", not still_open)
                    page.evaluate("closeModal()")
                else:
                    after_rows = page.locator("#modalBox .mlRow").count()
                    check("recent delete removes row", after_rows == before_rows - 1)
                    page.evaluate("closeModal()")

            # settings export download + import via file chooser
            page.click("#btnSettings")
            page.wait_for_timeout(250)
            with page.expect_download(timeout=6000) as dl3:
                page.click("#btnSetExport")
            check("settings export download",
                  dl3.value.suggested_filename.endswith(".json"))
            import json as _json
            tmp_settings = os.path.join(HERE, "_tmp_settings.json")
            dl3.value.save_as(tmp_settings)
            data = _json.load(open(tmp_settings, encoding="utf-8"))
            data["board"] = "emerald"
            _json.dump(data, open(tmp_settings, "w", encoding="utf-8"))
            with page.expect_file_chooser() as fc:
                page.click("#btnSetImport")
            fc.value.set_files(tmp_settings)
            page.wait_for_timeout(400)
            check("settings import applies", page.evaluate(
                "(JSON.parse(localStorage.getItem('zq.settings')).board)") == "emerald")
            os.remove(tmp_settings)
            page.evaluate("applyPreset('premiumDark'); closeModal()")

            # synthetic drop event imports a QFEN
            page.evaluate("""() => {
              const dt = new DataTransfer();
              dt.setData('text/plain', 'e2 e9 8 6 c6h 0');
              const ev = new DragEvent('drop', {dataTransfer: dt, bubbles: true});
              window.dispatchEvent(ev);
            }""")
            page.wait_for_timeout(500)
            drop_ok = page.evaluate("window.__w.pawn(0)") == 13
            check("drop imports position", drop_ok)
            if not drop_ok:
                print("   drop debug:", page.evaluate(
                    "({pawn: window.__w.pawn(0), err:"
                    " (function(){ window.__w.qfenImportStr('e2 e9 8 6 c6h 0');"
                    " return window.__w.lastErrStr(); })() })"))
                page.evaluate("newGame()")

            # ---------- hash load (fresh page) ----------
            page2 = browser.new_page()
            errs2 = []
            page2.on("pageerror", lambda e: errs2.append(str(e)))
            page2.goto("http://127.0.0.1:8201/style.html#qfen=" +
                       "e2%20e9%208%206%20c6h%200")
            page2.wait_for_timeout(2600)
            check("hash qfen loads position", page2.evaluate(
                "window.__w.pawn(0)") == 13)
            check("hash wall present", page2.evaluate(
                "window.__w.wallHBit(5*8+2)") == 1)
            check("hash page no errors", len(errs2) == 0)
            page2.close()

            # ---------- mobile viewport smoke (390x844) ----------
            mctx = browser.new_context(viewport={"width": 390, "height": 844},
                                       has_touch=True, is_mobile=True)
            mp = mctx.new_page()
            merrs = []
            mp.on("pageerror", lambda e: merrs.append(str(e)))
            mp.goto("http://127.0.0.1:8201/style.html")
            mp.wait_for_timeout(2600)
            check("mobile: sidePanel hidden by default",
                  not mp.is_visible("#sidePanel"))
            check("mobile: tab bar visible", mp.is_visible("#tabBar"))
            mp.click("#tabBar .tab[data-pane='anPane']")
            mp.wait_for_timeout(250)
            check("mobile: analysis opens sheet", mp.is_visible("#sidePanel"))
            mp.click("#tabBar .tab[data-pane='playPane']")
            mp.wait_for_timeout(250)
            check("mobile: play closes sheet", not mp.is_visible("#sidePanel"))
            # board is square and fits the width
            dims = mp.evaluate("""() => {
              const r = document.getElementById('board').getBoundingClientRect();
              return {w: Math.round(r.width), h: Math.round(r.height), vw: innerWidth};
            }""")
            check("mobile: board square", abs(dims["w"] - dims["h"]) <= 2)
            check("mobile: board fits viewport",
                  dims["w"] <= dims["vw"] and dims["w"] >= 300)
            # pawn move via touch tap
            src = mp.evaluate("""() => {
              const B = window.__qb, r = document.getElementById('board').getBoundingClientRect();
              const d = B.engPawnToDisp(4), c = B.cellCenter(Math.floor(d/9), d%9);
              return {x: r.left + c.x, y: r.top + c.y};
            }""")
            dst = mp.evaluate("""() => {
              const B = window.__qb, r = document.getElementById('board').getBoundingClientRect();
              const d = B.engPawnToDisp(13), c = B.cellCenter(Math.floor(d/9), d%9);
              return {x: r.left + c.x, y: r.top + c.y};
            }""")
            mp.touchscreen.tap(src["x"], src["y"])
            mp.touchscreen.tap(dst["x"], dst["y"])
            mp.wait_for_timeout(2800)
            check("mobile: pawn moved by touch",
                  mp.evaluate("window.__w.plyCount()") >= 2)
            check("mobile: zero page errors", len(merrs) == 0)
            if merrs:
                print("   mobile errors:", merrs[:5])
            mctx.close()

            check("ZERO page errors overall", len(errors) == 0)
            if errors:
                print("PAGE ERRORS:", errors[:8])
            browser.close()
    finally:
        srv.terminate()
    print("RESULT:", "PASS" if not failures else f"{len(failures)} failure(s): {failures}")
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
