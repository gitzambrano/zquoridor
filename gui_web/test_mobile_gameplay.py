#!/usr/bin/env python3
"""Long-form mobile gameplay/analysis acceptance test.

Exercises the actual browser interaction path rather than only engine calls:
- two mobile games, one from each side
- pawn moves and dock-drag wall moves through the UI
- real engine replies between human moves
- takeback followed by continued play
- HUD/clock/wall-count consistency after every turn
- Analysis engine, PV preview, history navigation, graph and blunder check
- QGN export/import round-trip of the played game
- source page and generated standalone bundle
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def fail(msg, details=None):
    if details is not None:
        msg += f" :: {details}"
    raise AssertionError(msg)


def wait_human(page, timeout=12000):
    page.wait_for_function(
        """() => gameOver ||
          (!engineThinking && atLiveEnd() && window.__w.turn() === humanSide)""",
        timeout=timeout,
    )


def setup_game(page, side):
    page.evaluate(
        """side => {
          S.level='pawn';              // 50 ms engine: fast but real search
          S.clockMode='none';
          S.side=side;
          S.anim='reduced';
          S.sound=false;
          S.haptics='off';
          S.confirmWalls=null;
          saveSettings();
          applySettings();
          newGame();
        }""",
        side,
    )
    wait_human(page)


def legal_choice(page, prefer_wall):
    return page.evaluate(
        """preferWall => {
          const W=window.__w;
          const moves=[];
          for(let i=0;i<W.moveCount();i++) {
            moves.push({
              wall:!!W.mvIsWall(i),
              a:W.mvA(i), b:W.mvB(i), c:W.mvC(i)
            });
          }
          const walls=moves.filter(m=>m.wall);
          const pawns=moves.filter(m=>!m.wall);
          if(preferWall && walls.length && W.wallsLeft(humanSide)>0) {
            // Central legal walls exercise drag/snap without depending on a
            // hard-coded slot surviving the engine's previous reply.
            walls.sort((x,y) => {
              const dx=Math.abs(x.b-3.5)+Math.abs(x.c-3.5);
              const dy=Math.abs(y.b-3.5)+Math.abs(y.c-3.5);
              return dx-dy;
            });
            return walls[0];
          }
          if(!pawns.length) return walls[0] || null;
          // Progress toward the human side's goal where possible. This makes
          // the session game-like rather than random wandering.
          pawns.sort((x,y) => {
            const rx=Math.floor(x.a/9), ry=Math.floor(y.a/9);
            return humanSide===0 ? (ry-rx) : (rx-ry);
          });
          return pawns[0];
        }""",
        prefer_wall,
    )


def play_ui_move(page, mv):
    if mv is None:
        fail("no legal human move")
    before = page.evaluate("window.__w.plyCount()")
    if mv["wall"]:
        target = page.evaluate(
            """m => {
              const B=window.__qb;
              const [o,r,c]=B.engWallToDisp(m.a,m.b,m.c);
              const p=B.anchorCenter(r,c);
              const br=document.getElementById('board').getBoundingClientRect();
              return {x:br.left+p.x,y:br.top+p.y,o};
            }""",
            mv,
        )
        button = page.locator("#wallV" if target["o"] else "#wallH").bounding_box()
        if not button:
            fail("wall dock button has no geometry", target)
        page.mouse.move(button["x"] + button["width"]/2,
                        button["y"] + button["height"]/2)
        page.mouse.down()
        page.mouse.move(target["x"], target["y"], steps=14)
        page.mouse.up()
    else:
        pts = page.evaluate(
            """m => {
              const B=window.__qb;
              const br=document.getElementById('board').getBoundingClientRect();
              const src=B.engPawnToDisp(window.__w.pawn(humanSide));
              const dst=B.engPawnToDisp(m.a);
              const s=B.cellCenter(Math.floor(src/9),src%9);
              const d=B.cellCenter(Math.floor(dst/9),dst%9);
              return {
                sx:br.left+s.x, sy:br.top+s.y,
                dx:br.left+d.x, dy:br.top+d.y
              };
            }""",
            mv,
        )
        page.mouse.click(pts["sx"], pts["sy"])
        page.mouse.click(pts["dx"], pts["dy"])

    page.wait_for_function(
        f"() => window.__w.plyCount() > {before} || gameOver", timeout=5000
    )
    wait_human(page)
    return before, page.evaluate("window.__w.plyCount()")


def check_live_invariants(page, label):
    x = page.evaluate(
        """() => {
          const r=e=>{const x=e.getBoundingClientRect();return {
            left:x.left,right:x.right,top:x.top,bottom:x.bottom,
            width:x.width,height:x.height};};
          const board=r(document.getElementById('board'));
          const top=document.getElementById('hudTop');
          const bottom=document.getElementById('hudBottom');
          const hud=[top,bottom].map(e=>({
            rect:r(e),
            pl:Number(e.dataset.player),
            wn:Number(e.querySelector('.wn').textContent),
            pips:[...e.querySelectorAll('.pips i.on')].length,
            clock:e.querySelector('.clock').textContent
          }));
          return {
            board, hud,
            width:innerWidth,
            ply:window.__w.plyCount(),
            cursor:window.__w.cursor(),
            turn:window.__w.turn(),
            human:humanSide,
            over:gameOver,
            thinking:engineThinking,
            status:document.getElementById('status').textContent,
            moves:document.getElementById('movesChip').textContent,
            confirm:getComputedStyle(document.getElementById('confirmChip')).display,
            bodyScroll:document.documentElement.scrollWidth,
            walls:[window.__w.wallsLeft(0),window.__w.wallsLeft(1)]
          };
        }"""
    )
    if abs(x["board"]["width"] - x["width"]) > 1.5:
        fail(label + ": board lost full mobile width", x)
    if x["bodyScroll"] > x["width"] + 1:
        fail(label + ": horizontal overflow", x)
    if x["hud"][0]["rect"]["bottom"] > x["board"]["top"] + .5:
        fail(label + ": top HUD overlaps board", x)
    if x["board"]["bottom"] > x["hud"][1]["rect"]["top"] + .5:
        fail(label + ": bottom HUD overlaps board", x)
    for h in x["hud"]:
        expected=x["walls"][h["pl"]]
        if h["wn"] != expected or h["pips"] != expected:
            fail(label + ": HUD wall count mismatch", x)
        if not h["clock"]:
            fail(label + ": clock vanished", x)
    if x["cursor"] != x["ply"]:
        fail(label + ": live game cursor left history end", x)
    if x["confirm"] != "none":
        fail(label + ": stale wall confirmation visible", x)
    return x


def play_session(page, side, human_turns, takeback_at=None):
    setup_game(page, side)
    check_live_invariants(page, f"side{side}-start")
    played = 0
    did_takeback = False
    for k in range(human_turns):
        if page.evaluate("gameOver"):
            break
        # Exercise both input families: roughly every third human move is a wall.
        prefer_wall = k in (2, 5, 8)
        mv = legal_choice(page, prefer_wall)
        before, after = play_ui_move(page, mv)
        if after <= before:
            fail("UI move did not advance game", {"before":before,"after":after,"move":mv})
        played += 1
        check_live_invariants(page, f"side{side}-turn{k+1}")

        if takeback_at is not None and k == takeback_at and not page.evaluate("gameOver"):
            n0=page.evaluate("window.__w.plyCount()")
            page.click("#btnTakeback")
            page.wait_for_timeout(250)
            wait_human(page)
            n1=page.evaluate("window.__w.plyCount()")
            if n1 >= n0:
                fail("takeback did not reduce history", {"before":n0,"after":n1})
            did_takeback=True
            check_live_invariants(page, f"side{side}-after-takeback")

    return {
        "played": played,
        "takeback": did_takeback,
        "plies": page.evaluate("window.__w.plyCount()"),
        "over": page.evaluate("gameOver"),
    }


def analyze_played_game(page, shots):
    n=page.evaluate("window.__w.plyCount()")
    if n < 6:
        fail("not enough plies for analysis", n)

    # Open through the actual mobile tab.
    page.click("#tabBar .tab[data-pane='anPane']")
    page.wait_for_timeout(200)
    if not page.evaluate("AN.on"):
        page.click("#anEngBtn")

    terminal=page.evaluate("window.__w.winner() !== -1 || window.__w.isDraw()")
    if terminal:
        page.wait_for_function(
            "() => document.getElementById('anInfo').textContent.includes('Game over')",
            timeout=3000,
        )
        term_txt=(page.text_content("#anLines") or "")
        if "Terminal position" not in term_txt:
            fail("terminal Analysis has no explanatory state", term_txt)
        # Inspect the decisive position from one ply earlier; this should kick
        # the engine automatically and produce a normal PV.
        page.click("#navPrev")
        page.wait_for_timeout(150)

    t0=time.time()
    try:
        page.wait_for_function(
            "() => document.querySelectorAll('.pvRow').length >= 1",
            timeout=15000,
        )
    except Exception:
        diag=page.evaluate("""() => ({
          ply:window.__w.plyCount(), cursor:window.__w.cursor(),
          winner:window.__w.winner(), draw:window.__w.isDraw(),
          pane:currentPane,
          depth:document.getElementById('anDepth').value,
          lines:document.getElementById('anPvCount').value,
          anOn:AN.on, anBusy:AN.busy, anSid:AN.sid, bcRun:AN.bcRun,
          info:document.getElementById('anInfo').textContent,
          lineHtml:document.getElementById('anLines').innerHTML.slice(0,400),
          worker:{
            ready:ANW.ready, failed:ANW.failed,
            pending:ANW.pending.size, nextId:ANW.nextId,
            lastError:ANW.lastError, hasWorker:!!ANW.wk
          }
        })""")
        print("ANALYSIS TIMEOUT DIAG:", diag, flush=True)
        fail("analysis produced no PV after moving off terminal position", diag)
    first_pv_ms=round((time.time()-t0)*1000)
    print("FIRST PV MS:", first_pv_ms, flush=True)
    info=page.text_content("#anInfo") or ""
    rows=page.locator(".pvRow").count()
    if rows < 1 or "nodes" not in info:
        fail("analysis did not produce PV/nodes", {"rows":rows,"info":info})

    # Preview a principal variation on the real board.
    page.locator(".pvRow").first.click()
    page.wait_for_timeout(100)
    preview=page.evaluate(
        "window.__qb.linePreview && window.__qb.linePreview.pawns.length + "
        "window.__qb.linePreview.walls.length"
    )
    if not preview:
        fail("PV click did not preview a line on the board")

    # Navigate several historical positions and verify the board follows.
    live=n
    cur0=page.evaluate("window.__w.cursor()")
    page.click("#navPrev")
    page.wait_for_timeout(200)
    if page.evaluate("window.__w.cursor()") != cur0-1:
        fail("analysis navPrev did not move one ply")
    page.click("#navPrev")
    page.wait_for_timeout(200)
    if page.evaluate("window.__w.cursor()") != cur0-2:
        fail("analysis second navPrev failed")
    if not page.is_visible("#btnReturn"):
        fail("Return to game missing during historical review")

    # Graph scrub should land somewhere in history, then Return restores live.
    graph=page.locator("#anGraph")
    if graph.is_visible():
        box=graph.bounding_box()
        if box:
            page.mouse.click(box["x"] + box["width"]*.35, box["y"] + box["height"]/2)
            page.wait_for_timeout(150)
            cur=page.evaluate("window.__w.cursor()")
            if not (0 <= cur <= live):
                fail("graph scrub cursor outside history", cur)

    page.click("#btnReturn")
    page.wait_for_timeout(200)
    if page.evaluate("window.__w.cursor()") != live:
        fail("Return to game did not restore live end")

    # Run a real blunder scan at shallow depth on the played game.
    page.select_option("#anDepth", "6")
    page.click("#anBlunderBtn")
    page.wait_for_function("() => AN.bcRun === false", timeout=30000)
    bc = page.evaluate(
        """() => ({
          scores:Object.keys(AN.scores).length,
          annots:Object.keys(AN.annots).length,
          summary:document.getElementById('bcSummary').textContent,
          box:getComputedStyle(document.getElementById('bcBox')).display
        })"""
    )
    if bc["scores"] < min(4, live):
        fail("blunder check analyzed too few positions", bc)

    page.screenshot(path=str(shots / "analysis-session.png"), full_page=True)

    # Round-trip the exact game and compare every ply token.
    rt=page.evaluate(
        """() => {
          const before=Array.from({length:window.__w.plyCount()},(_,i)=>plyNotation(i));
          const q=qgnExport();
          const n=window.__w.plyCount();
          const ok=importQGN(q);
          const after=Array.from({length:window.__w.plyCount()},(_,i)=>plyNotation(i));
          return {ok,n,n2:window.__w.plyCount(),before,after};
        }"""
    )
    if not rt["ok"] or rt["n"] != rt["n2"] or rt["before"] != rt["after"]:
        fail("played-game QGN round trip changed history", rt)

    # Play tab must return to live game and remain interactive.
    page.click("#tabBar .tab[data-pane='playPane']")
    page.wait_for_timeout(150)
    wait_human(page)
    check_live_invariants(page, "post-analysis-play")
    return {"pv_rows":rows,"blunder":bc,"plies":live}


def run_page(browser, url, shots, label):
    ctx=browser.new_context(
        viewport={"width":390,"height":844},
        has_touch=True,
        is_mobile=True,
    )
    page=ctx.new_page()
    errors=[]
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_function(
        "() => window.__w && window.__qb && typeof humanCanAct === 'function'",
        timeout=15000,
    )
    page.wait_for_timeout(500)

    first=play_session(page, side=0, human_turns=10, takeback_at=3)
    if first["played"] < 6:
        fail(label + ": first session too short", first)
    analysis=analyze_played_game(page, shots / label)

    # Second game from the opposite side checks engine-start and the mirrored
    # HUD/board path. It also continues using mixed pawn/wall UI moves.
    second=play_session(page, side=1, human_turns=7)
    if second["played"] < 4:
        fail(label + ": second-side session too short", second)

    (shots / label).mkdir(parents=True,exist_ok=True)
    page.screenshot(path=str(shots / label / "second-side.png"), full_page=True)

    if errors:
        fail(label + ": page errors", errors[:10])

    result={"first":first,"analysis":analysis,"second":second}
    print(label, result)
    ctx.close()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--page", choices=["style.html","zquoridor.html"], default="style.html")
    parser.add_argument("--shots", default="gameplay-shots")
    args=parser.parse_args()

    shots=HERE / args.shots
    shots.mkdir(parents=True,exist_ok=True)
    port=8221 if args.page=="style.html" else 8222
    srv=subprocess.Popen(
        [sys.executable,"dev_server.py",str(port)],
        cwd=HERE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL
    )
    time.sleep(1.0)
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch()
            result=run_page(
                browser,
                f"http://127.0.0.1:{port}/{args.page}",
                shots,
                Path(args.page).stem,
            )
            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=3)
        except subprocess.TimeoutExpired:
            srv.kill()

    print("PASS", args.page, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
