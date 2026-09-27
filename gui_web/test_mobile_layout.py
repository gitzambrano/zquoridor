#!/usr/bin/env python3
"""Focused portrait-mobile layout and wall gesture regression gate."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
VIEWPORTS = [(360, 800), (375, 812), (390, 844), (412, 915)]
COMPACT_HEIGHTS = {360: 640, 375: 640, 390: 650, 412: 667}


def fail(msg, details=None):
    if details is not None:
        msg += f" :: {details}"
    raise AssertionError(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="style.html",
                        choices=["style.html", "zquoridor.html"])
    parser.add_argument("--shots", default="mobile-shots")
    args = parser.parse_args()

    shots = HERE / args.shots / Path(args.page).stem
    shots.mkdir(parents=True, exist_ok=True)
    port = 8213 if args.page == "style.html" else 8214
    srv = subprocess.Popen([sys.executable, "dev_server.py", str(port)],
                           cwd=HERE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for width, height in VIEWPORTS:
                ctx = browser.new_context(
                    viewport={"width": width, "height": height},
                    has_touch=True, is_mobile=True,
                )
                ctx.add_init_script("""
                  localStorage.setItem('zq.settings',
                    JSON.stringify({v:4, boardScale:0.88}));
                """)
                page = ctx.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(f"http://127.0.0.1:{port}/{args.page}")
                page.wait_for_function(
                    "() => window.__w && window.__qb && typeof humanCanAct === 'function'",
                    timeout=15000,
                )
                page.wait_for_timeout(500)

                metrics = page.evaluate("""() => {
                  const q = s => document.querySelector(s);
                  const qa = s => [...document.querySelectorAll(s)];
                  const rect = el => {
                    const r = el.getBoundingClientRect();
                    return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,
                            width:r.width,height:r.height};
                  };
                  const board = rect(q('#board'));
                  const controls = rect(q('#controls'));
                  const buttons = qa('#controls > button').map(rect);
                  const hud = ['#hudTop','#hudBottom'].map(sel => {
                    const root=q(sel);
                    return {
                      root:rect(root),
                      walls:rect(root.querySelector('.walls')),
                      clock:rect(root.querySelector('.clock')),
                      dist:rect(root.querySelector('.dist')),
                      pips:qa(sel+' .pips i').map(rect),
                      pathBefore:getComputedStyle(root.querySelector('.dist'),'::before').display,
                    };
                  });
                  return {
                    vw:innerWidth, vh:innerHeight,
                    bodyScroll:document.documentElement.scrollWidth,
                    board, controls, buttons, hud,
                    header:rect(q('#hdr')),
                    tabbar:rect(q('#tabBar')),
                    zone:rect(q('#boardZone')),
                    under:rect(q('#underBoard')),
                    statusRow:rect(q('#statusRow')),
                    statusDisplay:getComputedStyle(q('#statusRow')).display,
                    controlScroll:q('#controls').scrollWidth,
                    controlClient:q('#controls').clientWidth,
                    configuredScale:document.documentElement.dataset.boardScale,
                  };
                }""")

                if abs(metrics["board"]["width"] - width) > 1.5:
                    fail("board is not viewport-wide", metrics)
                if abs(metrics["board"]["height"] - metrics["board"]["width"]) > 1.5:
                    fail("board is not square", metrics["board"])
                if metrics["configuredScale"] != "0.88":
                    fail("persisted boardScale was not reproduced", metrics["configuredScale"])
                if metrics["board"]["left"] < -0.5 or metrics["board"]["right"] > width + 0.5:
                    fail("board is not edge-to-edge inside viewport", metrics["board"])
                if metrics["hud"][0]["root"]["bottom"] > metrics["board"]["top"] + 0.5:
                    fail("top HUD overlaps board", metrics)
                if metrics["board"]["bottom"] > metrics["hud"][1]["root"]["top"] + 0.5:
                    fail("bottom HUD overlaps board", metrics)
                if metrics["header"]["bottom"] > metrics["hud"][0]["root"]["top"] + 0.5:
                    fail("top HUD overlaps header", metrics)
                if metrics["bodyScroll"] > width + 1:
                    fail("document has horizontal overflow", metrics)
                if metrics["controlScroll"] > metrics["controlClient"] + 1:
                    fail("controls have horizontal overflow", metrics)
                if metrics["statusDisplay"] == "none" or metrics["statusRow"]["height"] < 15:
                    fail("mobile status/resume row is hidden", metrics)

                tops = [r["top"] for r in metrics["buttons"]]
                if len(tops) != 7 or max(tops) - min(tops) > 1.5:
                    fail("controls are not one row", tops)
                if any(r["left"] < -0.5 or r["right"] > width + 0.5
                       for r in metrics["buttons"]):
                    fail("control leaves viewport", metrics["buttons"])

                for h in metrics["hud"]:
                    if len(h["pips"]) != 10:
                        fail("wall pip count changed", len(h["pips"]))
                    if any(p["width"] > 2.5 or p["height"] > 10.5
                           for p in h["pips"]):
                        fail("wall pips are not compact", h["pips"])
                    if h["pathBefore"] != "none":
                        fail("PATH text is still visible", h["pathBefore"])
                    if h["walls"]["right"] > h["clock"]["left"] + 0.5:
                        fail("wall stock overlaps clock", h)
                    if h["clock"]["right"] > h["dist"]["left"] + 0.5:
                        fail("clock overlaps path number", h)

                if width == 390:
                    # Resume must be visible and actionable on portrait. Use a
                    # real one-ply QGN, then boot the same resume path used on
                    # startup.
                    resume = page.evaluate("""() => {
                      newGame(); stopClock();
                      window.__w.applyPawn(13); syncAll();
                      const saved=qgnExport();
                      localStorage.setItem('zq.game', saved);
                      newGame(); stopClock();
                      checkAutosaveOnBoot();
                      const chip=document.getElementById('resumeChip');
                      const row=document.getElementById('statusRow');
                      return {
                        visible:getComputedStyle(chip).display !== 'none',
                        rowVisible:getComputedStyle(row).display !== 'none',
                        savedPlies:1
                      };
                    }""")
                    if not resume["visible"] or not resume["rowVisible"]:
                        fail("resume chip is hidden on portrait", resume)
                    page.click("#resumeChip")
                    page.wait_for_timeout(250)
                    if page.evaluate("window.__w.plyCount()") < 1:
                        fail("resume chip did not restore saved game")

                    # Scratch Editor rendering must never leak back to the live
                    # game after leaving without Apply.
                    editor = page.evaluate("""() => {
                      newGame(); stopClock();
                      const live=window.__w.pawn(1);
                      switchPane('edPane');
                      window.__w.editSetPawn(1,40);
                      renderScratchToBoard();
                      const scratch=window.__qb.pawn[1];
                      switchPane('playPane');
                      return {
                        live,
                        scratch,
                        board:window.__qb.pawn[1],
                        expected:window.__qb.engPawnToDisp(window.__w.pawn(1)),
                        flipped:window.__qb.flipped,
                        expectedFlip:((humanSide===1)!==!!S.flipped),
                        pane:currentPane
                      };
                    }""")
                    if editor["scratch"] == editor["expected"]:
                        fail("editor stress did not change scratch rendering", editor)
                    if editor["board"] != editor["expected"] or editor["flipped"] != editor["expectedFlip"]:
                        fail("leaving editor did not restore live board", editor)
                    if editor["pane"] != "playPane":
                        fail("leaving editor did not return to Play", editor)

                    # A historical Analysis position must not strand Play on an
                    # inert old ply.
                    review = page.evaluate("""() => {
                      newGame(); stopClock();
                      window.__w.applyPawn(13); syncAll();
                      const end=window.__w.plyCount();
                      switchPane('anPane');
                      navGo(0);
                      const old=window.__w.cursor();
                      switchPane('playPane');
                      return {old,end,cur:window.__w.cursor(),pane:currentPane};
                    }""")
                    if review["old"] != 0 or review["cur"] != review["end"] or review["pane"] != "playPane":
                        fail("Play did not return historical review to live game", review)

                    # Side-1 flag fall used to clamp side 0 and report the
                    # winner/loser message backwards.
                    flag = page.evaluate("""() => {
                      newGame(); stopClock();
                      window.__w.applyPawn(13); syncAll(); // side 1 to move
                      S.clockMode='5+0';
                      clockMs=[5000,1];
                      lastTickAt=performance.now()-25;
                      clockTick();
                      const out={
                        c0:clockMs[0],c1:clockMs[1],
                        status:document.getElementById('status').textContent,
                        active:[...document.querySelectorAll('.pbar.active')].length,
                        topLose:document.getElementById('hudTop').classList.contains('lose'),
                        bottomWin:document.getElementById('hudBottom').classList.contains('win')
                      };
                      S.clockMode='none'; newGame(); stopClock();
                      return out;
                    }""")
                    if flag["c1"] != 0 or flag["c0"] <= 0:
                        fail("flag fall clamped wrong clock", flag)
                    if flag["status"] != "Zquoridor ran out of time":
                        fail("flag fall message is wrong", flag)
                    if flag["active"] != 0 or not flag["topLose"] or not flag["bottomWin"]:
                        fail("flag fall HUD state is stale", flag)

                    # The portrait settings UI must not offer a board-scale
                    # control that portrait intentionally ignores.
                    hidden_scale = page.evaluate("""() => {
                      modalSettings('board');
                      const e=document.querySelector('.desktopBoardScale');
                      const hidden=e && getComputedStyle(e).display === 'none';
                      closeModal();
                      return hidden;
                    }""")
                    if not hidden_scale:
                        fail("portrait still exposes inactive board-scale control")

                    # Rotate both ways. Portrait and landscape share the <900px
                    # breakpoint but use different DOM homes for the move log.
                    page.set_viewport_size({"width": 844, "height": 390})
                    page.wait_for_timeout(350)
                    land = page.evaluate("""() => ({
                      mode:g_layoutMode,
                      logParent:document.getElementById('moveLog').parentElement.id,
                      panelPos:getComputedStyle(document.getElementById('sidePanel')).position,
                      board:document.getElementById('board').getBoundingClientRect().width
                    })""")
                    if land["mode"] != "phone-landscape" or land["logParent"] != "moveLogHome":
                        fail("portrait-to-landscape reflow failed", land)
                    if land["panelPos"] != "static" or land["board"] < 150:
                        fail("landscape rail/board geometry failed", land)
                    page.set_viewport_size({"width": width, "height": height})
                    page.wait_for_timeout(350)
                    portrait_back = page.evaluate("""() => ({
                      mode:g_layoutMode,
                      logParent:document.getElementById('moveLog').parentElement.id,
                      board:document.getElementById('board').getBoundingClientRect().width
                    })""")
                    if portrait_back["mode"] != "phone-portrait" or portrait_back["logParent"] != "underBoard":
                        fail("landscape-to-portrait reflow failed", portrait_back)
                    if abs(portrait_back["board"] - width) > 1.5:
                        fail("board did not recover full width after rotation", portrait_back)
                    page.evaluate("newGame(); stopClock()")
                    page.wait_for_timeout(100)

                # Stress the exact failure from the field screenshot: a long
                # move history plus a compact mobile-Chrome viewport. The log
                # must scroll in the leftover space; neither HUD may ever be
                # covered by a stale canvas.
                page.evaluate("""() => {
                  const log=document.getElementById('moveLog');
                  log.innerHTML=Array.from({length:120}, (_,i) =>
                    '<div class="mlRow"><span class="mlNum">'+(i+1)+
                    '.</span><span class="mlMv">e2</span><span class="mlMv">e8</span></div>'
                  ).join('');
                  log.scrollTop=log.scrollHeight;
                }""")
                page.wait_for_timeout(250)
                page.set_viewport_size({"width": width, "height": COMPACT_HEIGHTS[width]})
                page.wait_for_timeout(350)

                stressed = page.evaluate("""() => {
                  const q=s=>document.querySelector(s);
                  const rect=el=>{const r=el.getBoundingClientRect();return {
                    left:r.left,right:r.right,top:r.top,bottom:r.bottom,
                    width:r.width,height:r.height};};
                  return {
                    vw:innerWidth,vh:innerHeight,
                    board:rect(q('#board')),
                    zone:rect(q('#boardZone')),
                    top:rect(q('#hudTop')),
                    bottom:rect(q('#hudBottom')),
                    controls:rect(q('#controls')),
                    tabbar:rect(q('#tabBar')),
                    log:rect(q('#moveLog')),
                    status:rect(q('#statusRow')),
                    statusDisplay:getComputedStyle(q('#statusRow')).display,
                    logScroll:q('#moveLog').scrollHeight,
                    logClient:q('#moveLog').clientHeight,
                  };
                }""")
                if abs(stressed["board"]["width"] - width) > 1.5:
                    fail("long log shrank portrait board", stressed)
                if abs(stressed["zone"]["height"] - width) > 1.5:
                    fail("long log shrank board zone", stressed)
                if stressed["top"]["bottom"] > stressed["board"]["top"] + 0.5:
                    fail("long log made top HUD overlap board", stressed)
                if stressed["board"]["bottom"] > stressed["bottom"]["top"] + 0.5:
                    fail("long log made bottom HUD overlap board", stressed)
                if stressed["controls"]["bottom"] > stressed["tabbar"]["top"] + 0.5:
                    fail("controls are hidden under mobile tab bar", stressed)
                if stressed["statusDisplay"] == "none" or stressed["status"]["height"] < 15:
                    fail("long game hid mobile status row", stressed)
                if stressed["status"]["bottom"] > stressed["controls"]["top"] + 0.5:
                    fail("status row overlaps controls", stressed)
                if stressed["logClient"] > 0 and stressed["logScroll"] <= stressed["logClient"]:
                    fail("long move log is not scrollable", stressed)

                page.screenshot(
                    path=str(shots / f"{width}x{COMPACT_HEIGHTS[width]}-long.png"),
                    full_page=True,
                )

                # Analysis on mobile must keep the game surface visible. The
                # lower workspace may replace play controls/log, never the
                # board or either player strip.
                page.click("#tabBar .tab[data-pane='anPane']")
                page.wait_for_timeout(250)
                analysis = page.evaluate("""() => {
                  const q=s=>document.querySelector(s);
                  const rect=el=>{const r=el.getBoundingClientRect();return {
                    left:r.left,right:r.right,top:r.top,bottom:r.bottom,
                    width:r.width,height:r.height};};
                  const b=rect(q('#board')), p=rect(q('#sidePanel'));
                  const t=rect(q('#hudTop')), bot=rect(q('#hudBottom'));
                  const tabs=rect(q('#tabBar'));
                  const ctl=rect(q('#anPane > .card:first-child'));
                  const hit=document.elementFromPoint(
                    b.left+b.width/2, b.top+b.height/2);
                  return {
                    board:b,panel:p,top:t,bottom:bot,tabs,controlCard:ctl,
                    panelVisible:getComputedStyle(q('#sidePanel')).display !== 'none',
                    boardHit:hit ? hit.id : '',
                  };
                }""")
                if not analysis["panelVisible"]:
                    fail("analysis workspace did not open", analysis)
                if abs(analysis["board"]["width"] - width) > 1.5:
                    fail("analysis shrank mobile board", analysis)
                if analysis["panel"]["top"] < analysis["bottom"]["bottom"] - 0.5:
                    fail("analysis covers lower HUD", analysis)
                if analysis["panel"]["top"] < analysis["board"]["bottom"] - 0.5:
                    fail("analysis covers board", analysis)
                if analysis["panel"]["bottom"] > analysis["tabs"]["top"] + 0.5:
                    fail("analysis runs under tab bar", analysis)
                if analysis["boardHit"] != "board":
                    fail("analysis intercepts board surface", analysis)
                if analysis["controlCard"]["height"] > 55:
                    fail("short-screen analysis controls are too tall", analysis)

                page.screenshot(
                    path=str(shots / f"{width}x{COMPACT_HEIGHTS[width]}-analysis.png"),
                    full_page=True,
                )
                page.click("#tabBar .tab[data-pane='playPane']")
                page.wait_for_timeout(150)

                # Deliberate H-button drag to a known legal central slot.
                n0 = page.evaluate("window.__w.plyCount()")
                target = page.evaluate("""() => {
                  const B=window.__qb, br=document.getElementById('board').getBoundingClientRect();
                  const r=B.wallRect(0,3,3);
                  return {x:br.left+r.x+r.w/2, y:br.top+r.y+r.h/2};
                }""")
                btn = page.locator("#wallH").bounding_box()
                page.mouse.move(btn["x"] + btn["width"]/2, btn["y"] + btn["height"]/2)
                page.mouse.down()
                page.mouse.move(target["x"], target["y"], steps=12)
                page.mouse.up()
                page.wait_for_timeout(160)

                drag = page.evaluate("""() => ({
                  ply:window.__w.plyCount(),
                  confirm:getComputedStyle(document.getElementById('confirmChip')).display,
                  state:wallState
                })""")
                if drag["ply"] <= n0:
                    fail("dock drag did not place a wall", drag)
                if drag["confirm"] != "none":
                    fail("dock drag opened confirmation", drag)

                page.screenshot(
                    path=str(shots / f"{width}x{COMPACT_HEIGHTS[width]}-postdrag.png"),
                    full_page=True,
                )
                if errors:
                    fail("page errors", errors)
                ctx.close()

            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=3)
        except subprocess.TimeoutExpired:
            srv.kill()

    print(f"PASS {args.page}: {len(VIEWPORTS)} portrait viewports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
