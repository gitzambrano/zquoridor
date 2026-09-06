from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('ZQ_REPO_ROOT', ROOT))
GUI = REPO / 'gui_web'


def replace_once(path, old, new, label):
    text = path.read_text(encoding='utf-8')
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected 1 match, found {n} in {path}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')


p = GUI / 'style.html'
replace_once(p,
    '--cell-a:#e6d1ab; --cell-b:#d9bb8f;',
    '--cell-a:#e6d1ab; --cell-b:#dcc096;',
    'slightly softer wood checker contrast')
replace_once(p,
'''  /* Walls are pale plaster rails: bright against the wood bed, with a warm
     grey rim so they keep a premium, matte look instead of pure white. */
  --wall:#ecebe5;
  --wall-hi:#faf9f4;
  --wall-edge:#a6a49a;''',
'''  /* Neutral graphite rails: darker than the wood bed, with a restrained
     off-white highlight so the wall reads clearly without growing thicker. */
  --wall:#74797d;
  --wall-hi:#e4e7e8;
  --wall-edge:#4b5054;''',
    'darker neutral wall material')
replace_once(p,
    '#boardZone{padding-left:var(--space-2);padding-bottom:18px;gap:var(--space-4)}',
    '#boardZone{padding-left:var(--space-2);padding-bottom:22px;gap:var(--space-4)}',
    'desktop eval readout room')
replace_once(p,
    '#boardZone > #evalWrap #evalNum{font-size:12px;bottom:-18px}',
    '#boardZone > #evalWrap #evalNum{font-size:14px;font-weight:700;bottom:-20px}',
    'desktop eval readout size')
replace_once(p,
'''#boardZone > #evalWrap #evalNum{font-size:14px;font-weight:700;bottom:-20px}

</style>''',
'''#boardZone > #evalWrap #evalNum{font-size:14px;font-weight:700;bottom:-20px}

@media (max-width:899.98px){
  /* The move counter duplicates the log on a phone. Keep the status line
     focused on game state and give that width back to the board controls. */
  #movesChip{display:none}

  /* On portrait/mobile the evaluation strip belongs immediately below the
     controls. The JS reflow places the element there; these rules keep the
     spacing compact and make the move history the flexible/scrolling area. */
  #underBoard > #evalWrap{margin-top:1px;margin-bottom:1px}
  #underBoard > #moveLog{
    flex:1 1 auto;min-height:64px;max-height:100%;overflow-y:auto;overflow-x:hidden;
    overscroll-behavior:contain;touch-action:pan-y;user-select:text;-webkit-user-select:text;
    -webkit-touch-callout:default;cursor:text;scrollbar-gutter:stable;
  }
  #underBoard > #moveLog .mlRow,
  #underBoard > #moveLog .mlNum,
  #underBoard > #moveLog .mlMv{
    user-select:text;-webkit-user-select:text;
  }
  /* Keep the accent palette on one intentional row instead of leaving the
     custom swatch orphaned on a second line. */
  #setAccents{flex-wrap:nowrap;width:100%}
  #setAccents .swatch{flex:1 1 0;min-width:27px}
  #setAccents #accentPick{flex:0 0 34px}
}

</style>''',
    'mobile layout/readability polish')

p = GUI / 'board.js'
replace_once(p,
'''function qrTextureStrength(mode) {
  return mode === 'off' ? 0 : mode === 'natural' ? 1 : .52;
}
''',
'''function qrTextureStrength(mode) {
  return mode === 'off' ? 0 : mode === 'natural' ? 1 : .52;
}

// Goal rows borrow the player's own colour, but are pulled toward a neutral
// grey before being laid over the cells. The result is a quiet destination
// tint, not a second saturated board colour.
function qrMixHex(a, b, t) {
  const read = v => {
    const m = /^#([0-9a-f]{6})$/i.exec(String(v || '').trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const aa = read(a), bb = read(b);
  if (!aa || !bb) return a;
  const x = Math.max(0, Math.min(1, t));
  const rgb = aa.map((v, i) => Math.round(v * (1 - x) + bb[i] * x));
  return '#' + rgb.map(v => v.toString(16).padStart(2, '0')).join('');
}
function qrGoalTint(color) { return qrMixHex(color, '#85888d', .55); }
''',
    'goal tint helpers')
replace_once(p,
'''    // Goal rows are renderer-owned too: Off really means no wash and no edge.
    const goalMode = ds.goalRows || 'subtle';
    const goalAlpha = goalMode === 'clear' ? .21 : goalMode === 'off' ? 0 : .12;
    if (goalAlpha > 0) {
      g.fillStyle = `rgba(120,122,132,${goalAlpha})`;
      for (const r of [0, 8]) for (let c = 0; c < 9; c++) {
        const p = this.cellXY(r, c);
        if (cellSep === 'grooves') { this.rr(g, p.x, p.y, C, C, rad); g.fill(); }
        else g.fillRect(p.x - G / 2, p.y - G / 2, C + G, C + G);
      }
    }
''',
'''    // Goal rows tint only the nine destination cells. Each row follows the
    // player who is trying to reach it (red player 0, blue player 1), with
    // enough neutral grey mixed in that the tint stays subordinate to play.
    const goalMode = ds.goalRows || 'subtle';
    const goalAlpha = goalMode === 'clear' ? .21 : goalMode === 'off' ? 0 : .14;
    if (goalAlpha > 0) {
      const top = qrGoalTint(this.flipped ? this.css('--p1') : this.css('--p0'));
      const bottom = qrGoalTint(this.flipped ? this.css('--p0') : this.css('--p1'));
      g.save();
      g.globalAlpha = goalAlpha;
      for (const [r, tint] of [[0, top], [8, bottom]]) {
        g.fillStyle = tint;
        for (let c = 0; c < 9; c++) {
          const p = this.cellXY(r, c);
          if (cellSep === 'grooves') { this.rr(g, p.x, p.y, C, C, rad); g.fill(); }
          else g.fillRect(p.x, p.y, C, C);
        }
      }
      g.restore();
    }
''',
    'canvas goal row tint')
replace_once(p,
'''    // Goal edges follow the same control. Off removes every goal-row cue.
    if (goalMode !== 'off') {
      g.globalAlpha = goalMode === 'clear' ? .40 : .30;
      g.fillStyle = this.flipped ? this.css('--p0') : this.css('--p1');
      g.fillRect(bx, bx, bw, 2);
      g.fillStyle = this.flipped ? this.css('--p1') : this.css('--p0');
      g.fillRect(bx, bx + bw - 2, bw, 2);
      g.globalAlpha = 1;
    }

''', '', 'remove goal edge rails')
replace_once(p,
'''    const goalMode = ds.goalRows || 'subtle';
    if (goalMode !== 'off') {
      const goalAlpha = goalMode === 'clear' ? .21 : .12;
      b += `<g data-zq-goal-rows="${goalMode}">`;
      for (const r of [0, 8]) for (let c = 0; c < 9; c++)
        b += `<rect x="${(M + c * U).toFixed(1)}" y="${(M + r * U).toFixed(1)}" width="${C.toFixed(1)}" height="${C.toFixed(1)}" rx="${(C * .10).toFixed(1)}" fill="#787a84" fill-opacity="${goalAlpha}"/>`;
      const top = this.flipped ? cssOf('--p0') : cssOf('--p1');
      const bottom = this.flipped ? cssOf('--p1') : cssOf('--p0');
      const edgeAlpha = goalMode === 'clear' ? .40 : .30;
      b += `<rect x="${bx.toFixed(1)}" y="${bx.toFixed(1)}" width="${bw.toFixed(1)}" height="2" fill="${esc(top)}" fill-opacity="${edgeAlpha}"/>`;
      b += `<rect x="${bx.toFixed(1)}" y="${(bx + bw - 2).toFixed(1)}" width="${bw.toFixed(1)}" height="2" fill="${esc(bottom)}" fill-opacity="${edgeAlpha}"/>`;
      b += `</g>`;
    }
''',
'''    const goalMode = ds.goalRows || 'subtle';
    if (goalMode !== 'off') {
      const goalAlpha = goalMode === 'clear' ? .21 : .14;
      const top = qrGoalTint(this.flipped ? cssOf('--p1') : cssOf('--p0'));
      const bottom = qrGoalTint(this.flipped ? cssOf('--p0') : cssOf('--p1'));
      b += `<g data-zq-goal-rows="${goalMode}">`;
      for (const [r, tint] of [[0, top], [8, bottom]]) for (let c = 0; c < 9; c++)
        b += `<rect x="${(M + c * U).toFixed(1)}" y="${(M + r * U).toFixed(1)}" width="${C.toFixed(1)}" height="${C.toFixed(1)}" rx="${(C * .10).toFixed(1)}" fill="${esc(tint)}" fill-opacity="${goalAlpha}"/>`;
      b += `</g>`;
    }
''',
    'svg goal row tint')

p = GUI / 'app.js'
replace_once(p, 'function checkEnd() {\n',
'''function clearFinishedGameInteraction() {
  if (!B) return;
  clearGhost();
  B.selected = -1;
  B.dots = [];
  B.render();
}
function checkEnd() {
''', 'terminal interaction clear helper')
replace_once(p,
'''  stopClock(); gameOver = true; engineThinking = false;
  setStatus((side === humanSide ? 'Your clock ran out' : 'Zquoridor flagged you'));''',
'''  stopClock(); gameOver = true; engineThinking = false;
  clearFinishedGameInteraction();
  setStatus((side === humanSide ? 'Your clock ran out' : 'Zquoridor flagged you'));''',
    'flag terminal overlays')
replace_once(p,
'''    gameOver = true;
    const youWon = w === humanSide;''',
'''    gameOver = true;
    clearFinishedGameInteraction();
    const youWon = w === humanSide;''',
    'winner terminal overlays')
replace_once(p,
'''  if (W.isDraw()) { gameOver = true; markResult(-1); setStatus('Draw by repetition'); pushRecent('draw'); return true; }''',
'''  if (W.isDraw()) { gameOver = true; clearFinishedGameInteraction(); markResult(-1); setStatus('Draw by repetition'); pushRecent('draw'); return true; }''',
    'draw terminal overlays')
replace_once(p,
'''    gameOver = true; engineThinking = false;
    pushRecent('resign');''',
'''    gameOver = true; engineThinking = false;
    clearFinishedGameInteraction();
    pushRecent('resign');''',
    'resign terminal overlays')
replace_once(p,
'''  // The single evaluation bar is vertical beside the board on a wide screen
  // and horizontal under the board on a phone. It is the same element.
  const zone = $('boardZone'), ew = $('evalWrap');
  if (ew && zone && under) {
    if (wide) { if (ew.parentElement !== zone) zone.insertBefore(ew, zone.firstChild); }
    else if (ew.parentElement !== under) under.insertBefore(ew, under.firstChild);
  }
  const target = wide ? slot : under;
  for (const id of ['statusRow', 'controls']) {
    const el = $(id);
    if (el && el.parentElement !== target) target.appendChild(el);
  }
''',
'''  const target = wide ? slot : under;
  for (const id of ['statusRow', 'controls']) {
    const el = $(id);
    if (el && el.parentElement !== target) target.appendChild(el);
  }
  // The single evaluation bar is vertical beside the board on a wide screen.
  // On a phone it sits immediately BELOW the controls, before the scrolling
  // move history, so the board/readout hierarchy follows the thumb controls.
  const zone = $('boardZone'), ew = $('evalWrap'), controls = $('controls');
  if (ew && zone && under) {
    if (wide) {
      if (ew.parentElement !== zone) zone.insertBefore(ew, zone.firstChild);
    } else if (controls) {
      if (ew.parentElement !== under || ew.previousElementSibling !== controls) controls.after(ew);
    }
  }
''',
    'mobile eval bar order')

p = GUI / 'test_micro_polish_semantics.py'
replace_once(p,
'''            check("eval readout final size is 12px",
                  page.evaluate("getComputedStyle(document.getElementById('evalNum')).fontSize") == "12px")''',
'''            check("desktop eval readout is 14px",
                  page.evaluate("getComputedStyle(document.getElementById('evalNum')).fontSize") == "14px")''',
    'eval regression expectation')
replace_once(p,
'''            check("goal rows Subtle exports", 'data-zq-goal-rows="subtle"' in svg_subtle)''',
'''            check("goal rows Subtle exports", 'data-zq-goal-rows="subtle"' in svg_subtle)
            check("goal tint is player-coloured and has no edge rails",
                  '#ab6e66' in svg_subtle.lower() and '#6485bc' in svg_subtle.lower() and 'height="2"' not in svg_subtle)''',
    'goal export regression')
replace_once(p,
'''            check("SVG export still has coordinates", '<text' in svg and 'abcdefghi' not in svg)
            check("zero page errors", not errors)

            browser.close()''',
'''            check("SVG export still has coordinates", '<text' in svg and 'abcdefghi' not in svg)

            # Any terminal state must clear actionable overlays immediately.
            page.evaluate("window.__qb.dots=[1,2,3]; window.__qb.selected=4; flagFall(humanSide)")
            check("game-over clears legal-move dots", page.evaluate("window.__qb.dots.length") == 0)
            check("game-over clears pawn selection", page.evaluate("window.__qb.selected") == -1)

            # Mobile hierarchy: no redundant Moves chip, controls then eval, then
            # a selectable scrolling move log.
            mobile = browser.new_page(viewport={"width":390,"height":844}, is_mobile=True, has_touch=True)
            mobile.goto("http://127.0.0.1:8213/style.html")
            mobile.wait_for_timeout(2200)
            mob = mobile.evaluate("""() => {
              const q=s=>document.querySelector(s), log=q('#moveLog'), st=getComputedStyle(log);
              return {
                moves:getComputedStyle(q('#movesChip')).display,
                evalParent:q('#evalWrap').parentElement.id,
                evalPrev:q('#evalWrap').previousElementSibling && q('#evalWrap').previousElementSibling.id,
                logParent:log.parentElement.id,
                overflow:st.overflowY, select:st.userSelect, touch:st.touchAction,
                bodyW:document.body.scrollWidth, vw:innerWidth
              };
            }""")
            check("mobile hides redundant Moves counter", mob['moves'] == 'none')
            check("mobile eval bar is directly below controls", mob['evalParent'] == 'underBoard' and mob['evalPrev'] == 'controls')
            check("mobile move log is under board and scrollable/selectable",
                  mob['logParent'] == 'underBoard' and mob['overflow'] == 'auto' and mob['select'] == 'text' and 'pan-y' in mob['touch'])
            check("mobile has no horizontal overflow", mob['bodyW'] == mob['vw'])
            mobile.close()

            check("zero page errors", not errors)

            browser.close()''',
    'final regression coverage')

print('final polish patch applied')
