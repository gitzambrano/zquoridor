import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8123/style.html"
ok, bad = [], []


def chk(c, m):
    (ok if c else bad).append(("PASS " if c else "FAIL ") + m)


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
    pg.goto(URL)
    pg.wait_for_timeout(3800)

    def geo():
        return pg.evaluate("()=>{const r=__qb.cv.getBoundingClientRect();"
                           "return{x:r.x,y:r.y,M:__qb.M,C:__qb.C,U:__qb.U,G:__qb.G}}")

    def cell(r, c):
        g = geo()
        return (g["x"] + g["M"] + c * g["U"] + g["C"] / 2,
                g["y"] + g["M"] + r * g["U"] + g["C"] / 2)

    def anch(r, c):
        g = geo()
        return (g["x"] + g["M"] + (c + 1) * g["U"] - g["G"] / 2,
                g["y"] + g["M"] + (r + 1) * g["U"] - g["G"] / 2)

    def wait_human(t=25000):
        pg.wait_for_function("()=>!engineThinking && __w.turn()===humanSide", timeout=t)

    chk("BOOTERR" not in (pg.title() or ""), "boots clean")
    pg.evaluate("S.level='knight'; saveSettings(); newGame();")
    pg.wait_for_timeout(400)

    n0 = pg.evaluate("__w.plyCount()")
    x, y = cell(7, 4)
    pg.mouse.click(x, y)
    wait_human()
    chk(pg.evaluate("__w.plyCount()") > n0 + 1, "pawn move by click, engine replied")

    n1 = pg.evaluate("__w.plyCount()")
    sr, sc = pg.evaluate("()=>{const d=__qb.engPawnToDisp(__w.pawn(humanSide));return [Math.floor(d/9),d%9]}")
    dr, dc = pg.evaluate("()=>{const d=[...legalPawn][0];return [Math.floor(d/9),d%9]}")
    src, dst = cell(sr, sc), cell(dr, dc)
    pg.mouse.move(*src)
    pg.mouse.down()
    pg.mouse.move(dst[0], dst[1], steps=8)
    pg.mouse.up()
    wait_human()
    chk(pg.evaluate("__w.plyCount()") > n1 + 1, "pawn move by drag")

    n2 = pg.evaluate("__w.plyCount()")
    w0 = pg.evaluate("__w.wallsLeft(humanSide)")
    ax, ay = anch(4, 2)
    pg.mouse.move(ax, ay)
    pg.wait_for_timeout(150)
    chk(pg.evaluate("()=>!!__qb.hover"), "hover shows a wall preview")
    pg.mouse.down()
    pg.mouse.up()
    wait_human()
    chk(pg.evaluate("__w.plyCount()") > n2 + 1 and pg.evaluate("__w.wallsLeft(humanSide)") == w0 - 1,
        "wall placed contextually, no mode button")

    n3 = pg.evaluate("__w.plyCount()")
    ax, ay = anch(3, 5)
    C = pg.evaluate("__qb.C")
    pg.mouse.move(ax, ay)
    pg.mouse.down()
    pg.mouse.move(ax, ay + C * 0.9, steps=6)
    o = pg.evaluate("()=>__qb.ghost && __qb.ghost.o")
    chk(o == 1, "drag across flips the ghost to vertical (o=%s)" % o)
    pg.mouse.up()
    pg.wait_for_timeout(400)
    if pg.evaluate("__w.plyCount()") > n3:
        wait_human()

    n4 = pg.evaluate("__w.plyCount()")
    pg.click("#wallV")
    pg.wait_for_timeout(200)
    chk(pg.evaluate("forcedO") == 1, "V button forces the vertical orientation")
    vx, vy = anch(2, 6)
    pg.mouse.click(vx, vy)
    pg.wait_for_timeout(500)
    placed = pg.evaluate("__w.plyCount()") > n4
    chk(placed, "wall placed through the V button")
    if placed:
        wait_human()

    nb = pg.evaluate("__w.plyCount()")
    pg.click("#btnTakeback")
    pg.wait_for_timeout(700)
    chk(pg.evaluate("__w.plyCount()") < nb, "takeback removes plies")
    chk(pg.evaluate("__w.turn()") == pg.evaluate("humanSide"), "takeback leaves your move")

    pg.click("#btnHint")
    pg.wait_for_timeout(3000)
    chk(pg.evaluate("()=>__qb.ghost!==null||__qb.selected>=0||!!__qb.linePreview"),
        "hint shows a suggestion")

    f0 = pg.evaluate("__qb.flipped")
    pg.click("#btnFlip")
    pg.wait_for_timeout(300)
    chk(pg.evaluate("__qb.flipped") != f0, "flip board")
    pg.click("#btnFlip")
    pg.wait_for_timeout(250)

    pg.click("#btnPaths")
    pg.wait_for_timeout(350)
    chk(pg.evaluate("()=>!!__qb.paths && __qb.paths.length===2"), "paths overlay on")
    pg.click("#btnPaths")
    pg.wait_for_timeout(250)

    ev = pg.evaluate("()=>{const s=document.getElementById('evalStrip').getBoundingClientRect();"
                     "const f=document.getElementById('evalFill').getBoundingClientRect();"
                     "return {w:Math.round(s.width), share:Math.round(f.height/s.height*100),"
                     "num:document.getElementById('evalNum').textContent}}")
    chk(ev["w"] == 24, "vertical eval bar is 24px wide (got %s)" % ev["w"])
    chk(ev["num"].endswith("%"), "eval readout is a percentage (%s)" % ev["num"])
    chk(0 <= ev["share"] <= 100, "eval fill stays within 0 to 100 percent (%s)" % ev["share"])

    mlog = pg.text_content("#moveLog") or ""
    chk("%" not in mlog and "+" not in mlog, "play move log carries no evaluation")

    pg.click("#panelTabs .tab[data-pane='anPane']")
    pg.wait_for_timeout(400)
    pg.select_option("#anPvCount", "3")
    pg.wait_for_timeout(250)
    pg.click("#anEngBtn")
    pg.wait_for_timeout(7000)
    rows = pg.locator("#anLines .pvRow").count()
    chk(rows == 3, "analysis shows 3 best moves (got %d)" % rows)
    pv = pg.text_content("#anLines") or ""
    chk("a1 a1" not in pv, "PV notation is real, not a repeated a1")
    chk("nodes" in (pg.text_content("#anInfo") or ""), "analysis info line filled")
    chk(pg.locator("#anMoveLog .alRow").count() > 0, "analysis move list rendered")

    pg.click("#anBlunderBtn")
    pg.wait_for_function("()=>!AN.bcRun", timeout=180000)
    pg.wait_for_timeout(800)
    scored = pg.evaluate("()=>{const n=__w.plyCount();let c=0;"
                         "for(let i=0;i<n;i++) if(AN.scores[i]!=null)c++;return [c,n]}")
    chk(scored[0] == scored[1],
        "every ply has an evaluation after blunder check (%d of %d)" % (scored[0], scored[1]))

    cur = pg.evaluate("__w.cursor()")
    pg.click("#navPrev")
    pg.wait_for_timeout(500)
    chk(pg.evaluate("__w.cursor()") == cur - 1, "nav previous")
    pg.click("#navLast")
    pg.wait_for_timeout(500)
    chk(pg.evaluate("__w.cursor()") == pg.evaluate("__w.plyCount()"), "nav last")
    pg.click("#anEngBtn")
    pg.wait_for_timeout(400)

    pg.click("#panelTabs .tab[data-pane='edPane']")
    pg.wait_for_timeout(500)
    chk(pg.is_visible("#edValidity"), "editor pane opens")
    pg.click("#etWallH")
    pg.wait_for_timeout(200)
    ex, ey = anch(5, 5)
    pg.mouse.click(ex, ey)
    pg.wait_for_timeout(400)
    chk(pg.evaluate("()=>{for(let i=0;i<64;i++) if(__w.scrWallHBit(i)) return true; return false;}"),
        "editor places a wall on the scratch board")
    pg.click("#btnEdClear")
    pg.wait_for_timeout(400)
    pg.click("#panelTabs .tab[data-pane='playPane']")
    pg.wait_for_timeout(400)

    pg.click("#btnSettings")
    pg.wait_for_timeout(600)
    chk(pg.is_visible("#overlay"), "settings modal opens")
    pg.click("#modalBox [data-mtab='board']")
    pg.wait_for_timeout(200)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#setBoards .swatch')]"
                ".find(x=>x.dataset.b==='walnut'); if(b) b.click();}")
    pg.wait_for_timeout(400)
    chk(pg.evaluate("document.documentElement.dataset.board") == "walnut",
        "board theme applies live")
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#setBoards .swatch')]"
                ".find(x=>x.dataset.b==='wood'); if(b) b.click();}")
    pg.wait_for_timeout(300)
    pg.evaluate("closeModal()")
    pg.wait_for_timeout(300)

    pg.click("#btnNew")
    pg.wait_for_timeout(600)
    lv = pg.evaluate("()=>[...document.querySelectorAll('[data-lvl]')].map(e=>e.dataset.lvl)")
    chk(lv == ["pawn", "knight", "bishop", "rook", "queen", "king"],
        "levels are the chess pieces: %s" % lv)
    pg.evaluate("closeModal()")
    pg.wait_for_timeout(300)

    chk(not errs, "no page errors: %s" % errs[:3])
    b.close()

for l in ok:
    print(l)
for l in bad:
    print(l)
print("")
print("%d passed, %d failed" % (len(ok), len(bad)))
