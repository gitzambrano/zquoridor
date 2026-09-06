from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()

old = "const base = theme === 'walnut' ? .050 : .040;"
new = "const base = theme === 'walnut' ? .050 : .044;"
assert old in s
s = s.replace(old, new, 1)

old = "function qrGoalTint(color) { return qrMixHex(color, '#85888d', .55); }"
new = """function qrGoalTint(color) { return qrMixHex(color, '#85888d', .55); }

// Equal numeric opacity does not mean equal visual emphasis: on the warm wood
// substrate the red tint moves the cell colour less than the blue tint. Pick
// alpha from the actual tint-to-cell distance so both goal sides read with
// comparable strength, including custom pawn colours and other board themes.
function qrGoalAlpha(tint, cellA, cellB, mode) {
  const read = v => {
    const m = /^#([0-9a-f]{6})$/i.exec(String(v || '').trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const t = read(tint), a = read(cellA), b = read(cellB);
  if (!t || !a || !b) return mode === 'clear' ? .23 : .16;
  const d = x => Math.hypot(t[0] - x[0], t[1] - x[1], t[2] - x[2]);
  const span = (d(a) + d(b)) / 2;
  if (span < 1) return mode === 'clear' ? .23 : .16;
  const target = mode === 'clear' ? 31.5 : 21;
  const alpha = target / span;
  const lo = mode === 'clear' ? .18 : .13, hi = mode === 'clear' ? .32 : .22;
  return Math.max(lo, Math.min(hi, alpha));
}"""
assert old in s
s = s.replace(old, new, 1)

old = """    // Goal rows tint only the nine destination cells. Each row follows the
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
    }"""
new = """    // Goal rows tint only the nine cells on each player's visual side of
    // the board. The tint follows the pawn colour when the board is flipped.
    // Alpha is balanced perceptually so one colour does not disappear into
    // the substrate merely because both rows used the same numeric opacity.
    const goalMode = ds.goalRows || 'subtle';
    if (goalMode !== 'off') {
      const top = qrGoalTint(this.flipped ? this.css('--p0') : this.css('--p1'));
      const bottom = qrGoalTint(this.flipped ? this.css('--p1') : this.css('--p0'));
      const topAlpha = qrGoalAlpha(top, ca, cb, goalMode);
      const bottomAlpha = qrGoalAlpha(bottom, ca, cb, goalMode);
      g.save();
      for (const [r, tint, alpha] of [[0, top, topAlpha], [8, bottom, bottomAlpha]]) {
        g.globalAlpha = alpha;
        g.fillStyle = tint;
        for (let c = 0; c < 9; c++) {
          const p = this.cellXY(r, c);
          if (cellSep === 'grooves') { this.rr(g, p.x, p.y, C, C, rad); g.fill(); }
          else g.fillRect(p.x, p.y, C, C);
        }
      }
      g.restore();
    }"""
assert old in s
s = s.replace(old, new, 1)

old = """    const goalMode = ds.goalRows || 'subtle';
    if (goalMode !== 'off') {
      const goalAlpha = goalMode === 'clear' ? .21 : .14;
      const top = qrGoalTint(this.flipped ? cssOf('--p1') : cssOf('--p0'));
      const bottom = qrGoalTint(this.flipped ? cssOf('--p0') : cssOf('--p1'));
      b += `<g data-zq-goal-rows=\"${goalMode}\">`;
      for (const [r, tint] of [[0, top], [8, bottom]]) for (let c = 0; c < 9; c++)
        b += `<rect x=\"${(M + c * U).toFixed(1)}\" y=\"${(M + r * U).toFixed(1)}\" width=\"${C.toFixed(1)}\" height=\"${C.toFixed(1)}\" rx=\"${(C * .10).toFixed(1)}\" fill=\"${esc(tint)}\" fill-opacity=\"${goalAlpha}\"/>`;
      b += `</g>`;
    }"""
new = """    const goalMode = ds.goalRows || 'subtle';
    if (goalMode !== 'off') {
      const top = qrGoalTint(this.flipped ? cssOf('--p0') : cssOf('--p1'));
      const bottom = qrGoalTint(this.flipped ? cssOf('--p1') : cssOf('--p0'));
      const topAlpha = qrGoalAlpha(top, ca, cb, goalMode);
      const bottomAlpha = qrGoalAlpha(bottom, ca, cb, goalMode);
      b += `<g data-zq-goal-rows=\"${goalMode}\">`;
      for (const [r, tint, alpha] of [[0, top, topAlpha], [8, bottom, bottomAlpha]]) for (let c = 0; c < 9; c++)
        b += `<rect x=\"${(M + c * U).toFixed(1)}\" y=\"${(M + r * U).toFixed(1)}\" width=\"${C.toFixed(1)}\" height=\"${C.toFixed(1)}\" rx=\"${(C * .10).toFixed(1)}\" fill=\"${esc(tint)}\" fill-opacity=\"${alpha.toFixed(4)}\"/>`;
      b += `</g>`;
    }"""
assert old in s
s = s.replace(old, new, 1)

p.write_text(s)
