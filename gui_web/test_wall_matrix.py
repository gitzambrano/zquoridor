"""Wall placement matrix: drag direction and every board edge, mouse and touch."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8123/style.html"
ok, bad = [], []


def chk(c, m):
    (ok if c else bad).append(("PASS " if c else "FAIL ") + m)


def geo(pg):
    return pg.evaluate("()=>{const r=__qb.cv.getBoundingClientRect();"
                       "return{x:r.x,y:r.y,M:__qb.M,C:__qb.C,U:__qb.U,G:__qb.G}}")


def anch(pg, r, c):
    g = geo(pg)
    return (g["x"] + g["M"] + (c + 1) * g["U"] - g["G"] / 2,
            g["y"] + g["M"] + (r + 1) * g["U"] - g["G"] / 2)


def reset(pg):
    """Fresh position, human to move, engine idle, no clock pressure."""
    pg.evaluate("S.clockMode='none'; S.level='pawn'; saveSettings(); newGame();")
    pg.wait_for_timeout(350)
    pg.wait_for_function("()=>!engineThinking && __w.turn()===humanSide", timeout=20000)


def drag(pg, r, c, dx, dy, touch=False):
    """Press an anchor, pull by (dx,dy) cells, release. Returns the ghost seen."""
    x, y = anch(pg, r, c)
    C = pg.evaluate("__qb.C")
    tx, ty = x + dx * C, y + dy * C
    if touch:
        pg.evaluate("""([x,y,tx,ty])=>{
          const cv=document.getElementById('board');
          const mk=(t,px,py)=>new PointerEvent(t,{pointerId:7,pointerType:'touch',
              isPrimary:true,clientX:px,clientY:py,bubbles:true,cancelable:true});
          cv.dispatchEvent(mk('pointerdown',x,y));
          cv.dispatchEvent(mk('pointermove',(x+tx)/2,(y+ty)/2));
          cv.dispatchEvent(mk('pointermove',tx,ty));
          window.__ghost=__qb.ghost?{o:__qb.ghost.o,r:__qb.ghost.r,c:__qb.ghost.c,
                                     state:__qb.ghost.state}:null;
          cv.dispatchEvent(mk('pointerup',tx,ty));
        }""", [x, y, tx, ty])
        g = pg.evaluate("window.__ghost")
    else:
        pg.mouse.move(x, y)
        pg.mouse.down()
        if dx or dy:
            pg.mouse.move(tx, ty, steps=6)
        else:
            pg.mouse.move(x + 1, y + 1, steps=2)
        g = pg.evaluate("()=>__qb.ghost?{o:__qb.ghost.o,r:__qb.ghost.r,c:__qb.ghost.c,"
                        "state:__qb.ghost.state}:null")
        pg.mouse.up()
    pg.wait_for_timeout(350)
    return g


def run(tag, w, h, touch):
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": w, "height": h},
                            has_touch=touch, is_mobile=False)
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL)
        pg.wait_for_timeout(3800)
        pg.evaluate("S.confirmWalls=false; saveSettings();")
        print("--- %s (%dx%d, touch=%s) ---" % (tag, w, h, touch))

        # 1. drag direction decides orientation
        reset(pg)
        g = drag(pg, 3, 3, 1.0, 0.0, touch)
        chk(g and g["o"] == 0, "%s: horizontal drag gives a horizontal wall (%s)" % (tag, g))
        reset(pg)
        g = drag(pg, 3, 3, 0.0, 1.0, touch)
        chk(g and g["o"] == 1, "%s: vertical drag gives a vertical wall (%s)" % (tag, g))

        # 2. a press with no drag keeps the groove's own orientation
        if not touch:
            reset(pg)
            gx, gy = anch(pg, 4, 4)
            C = pg.evaluate("__qb.C")
            # On a horizontal groove only: x in the middle of a cell, y on the groove.
            pg.mouse.move(gx - C * 0.5, gy)
            pg.wait_for_timeout(150)
            hov = pg.evaluate("()=>__qb.hover?__qb.hover.o:null")
            chk(hov == 0, "%s: hovering a horizontal groove previews H (o=%s)" % (tag, hov))
            # On a vertical groove only: x on the groove, y in the middle of a cell.
            pg.mouse.move(gx, gy - C * 0.5)
            pg.wait_for_timeout(150)
            hov = pg.evaluate("()=>__qb.hover?__qb.hover.o:null")
            chk(hov == 1, "%s: hovering a vertical groove previews V (o=%s)" % (tag, hov))
            # In the middle of a cell: no wall preview at all.
            cx = pg.evaluate("()=>{const p=__qb.cellCenter(4,4);const r=__qb.cv.getBoundingClientRect();"
                             "return [r.x+p.x, r.y+p.y]}")
            pg.mouse.move(cx[0], cx[1])
            pg.wait_for_timeout(150)
            chk(pg.evaluate("()=>__qb.hover===null"), "%s: a cell body shows no wall preview" % tag)

        # 3. every corner and edge anchor, both orientations
        spots = [(0, 0, "top-left"), (0, 7, "top-right"), (7, 0, "bottom-left"),
                 (7, 7, "bottom-right"), (0, 4, "top edge"), (7, 4, "bottom edge"),
                 (4, 0, "left edge"), (4, 7, "right edge")]
        for r, c, name in spots:
            for o, dx, dy in ((0, 0.6, 0.0), (1, 0.0, 0.6)):
                reset(pg)
                g = drag(pg, r, c, dx, dy, touch)
                # Assert on the board, not on the last ply: the engine answers
                # immediately, so the last ply is its reply, not the wall.
                # The invariant is what you see is what you get: the wall must
                # land on the slot the ghost showed at release, in the
                # orientation the drag asked for.
                if not g:
                    chk(False, "%s: %s %s no ghost at release" % (tag, name, "H" if o == 0 else "V"))
                    continue
                got = pg.evaluate("""([o,r,c])=>{
                  const e=__qb.dispWallToEng(o,r,c);
                  const bit = e[0]===0 ? __w.wallHBit(e[1]*8+e[2]) : __w.wallVBit(e[1]*8+e[2]);
                  return {eng:e, bit:bit};}""", [g["o"], g["r"], g["c"]])
                chk(g["o"] == o and got["bit"] == 1,
                    "%s: %s %s placed where the ghost showed (ghost=%s,%s,%s eng=%s)" %
                    (tag, name, "H" if o == 0 else "V", g["o"], g["r"], g["c"], got["eng"]))

        # 4. the extreme slots, reached by a plain press with no drag at all.
        #    The pointer sits on one groove only, so the orientation is implied.
        C = pg.evaluate("__qb.C")
        for r, c, name in [(0, 0, "corner (0,0)"), (0, 7, "corner (0,7)"),
                           (7, 0, "corner (7,0)"), (7, 7, "corner (7,7)")]:
            for o in (0, 1):
                reset(pg)
                gx, gy = anch(pg, r, c)
                px = gx - C * 0.5 if o == 0 else gx
                py = gy if o == 0 else gy - C * 0.5
                if touch:
                    pg.evaluate("""([x,y])=>{const cv=document.getElementById('board');
                      const mk=(t)=>new PointerEvent(t,{pointerId:9,pointerType:'touch',
                        isPrimary:true,clientX:x,clientY:y,bubbles:true,cancelable:true});
                      cv.dispatchEvent(mk('pointerdown'));
                      window.__g2=__qb.ghost?{o:__qb.ghost.o,r:__qb.ghost.r,c:__qb.ghost.c}:null;
                      cv.dispatchEvent(mk('pointerup'));}""", [px, py])
                    g2 = pg.evaluate("window.__g2")
                else:
                    pg.mouse.move(px, py)
                    pg.mouse.down()
                    g2 = pg.evaluate("()=>__qb.ghost?{o:__qb.ghost.o,r:__qb.ghost.r,c:__qb.ghost.c}:null")
                    pg.mouse.up()
                pg.wait_for_timeout(350)
                if not g2:
                    chk(False, "%s: %s press %s gave no ghost" % (tag, name, "H" if o == 0 else "V"))
                    continue
                bit = pg.evaluate("""([o,r,c])=>{const e=__qb.dispWallToEng(o,r,c);
                  return e[0]===0 ? __w.wallHBit(e[1]*8+e[2]) : __w.wallVBit(e[1]*8+e[2]);}""",
                  [g2["o"], g2["r"], g2["c"]])
                chk(g2["o"] == o and bit == 1,
                    "%s: %s press places a %s wall (ghost=%s,%s,%s)" %
                    (tag, name, "H" if o == 0 else "V", g2["o"], g2["r"], g2["c"]))

        chk(not errs, "%s: no page errors %s" % (tag, errs[:2]))
        b.close()


run("desktop", 1440, 900, False)
run("mobile", 390, 844, True)

for l in ok:
    print(l)
for l in bad:
    print(l)
print("")
print("%d passed, %d failed" % (len(ok), len(bad)))
