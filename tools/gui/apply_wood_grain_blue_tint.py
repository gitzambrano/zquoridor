from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()

old_alpha = """function qrGoalAlpha(tint, cellA, cellB, mode) {
  const read = v => {
    const m = /^#([0-9a-f]{6})$/i.exec(String(v || '').trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const t = read(tint), a = read(cellA), b = read(cellB);
  if (!t || !a || !b) return mode === 'clear' ? .38 : .30;
  const d = x => Math.hypot(t[0] - x[0], t[1] - x[1], t[2] - x[2]);
  const span = (d(a) + d(b)) / 2;
  if (span < 1) return mode === 'clear' ? .38 : .30;
  const target = mode === 'clear' ? 56 : 42;
  const alpha = target / span;
  const lo = mode === 'clear' ? .34 : .27, hi = mode === 'clear' ? .52 : .40;
  return Math.max(lo, Math.min(hi, alpha));
}
"""
new_alpha = old_alpha + """
// Blue is naturally more conspicuous against the warm wood substrate. Keep
// the same tint colour, but trim only player 1's wash by 8% so the two sides
// remain clear without the blue row dominating the board.
function qrGoalPlayerAlpha(tint, cellA, cellB, mode, player) {
  return qrGoalAlpha(tint, cellA, cellB, mode) * (player === 1 ? .92 : 1);
}
"""
if old_alpha not in s:
    raise SystemExit('missing goal alpha block')
s = s.replace(old_alpha, new_alpha, 1)

old_wood = """    const count = theme === 'walnut' ? 38 : 40;
    const base = theme === 'walnut' ? .050 : .062;
"""
new_wood = """    const count = theme === 'walnut' ? 38 : 46;
    const base = theme === 'walnut' ? .050 : .090;
"""
if old_wood not in s:
    raise SystemExit('missing wood count/base')
s = s.replace(old_wood, new_wood, 1)

repls = {
    "out.push({ kind: 'poly', points: pts, width: .36 + rnd() * .60,":
    "out.push({ kind: 'poly', points: pts, width: .46 + rnd() * .68,",
    "width: .28 + rnd() * .42, tone: t, alpha: a * .38 });":
    "width: .32 + rnd() * .46, tone: t, alpha: a * .45 });",
    "const fibres = theme === 'walnut' ? 34 : 48;":
    "const fibres = theme === 'walnut' ? 34 : 58;",
    ".26 + rnd() * .34, tone(.62), base * s * (.24 + .20 * rnd()));":
    ".30 + rnd() * .40, tone(.62), base * s * (.30 + .24 * rnd()));",
}
for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'missing wood detail: {old}')
    s = s.replace(old, new, 1)

old_canvas = """      const top = qrGoalTint(this.flipped ? this.css('--p0') : this.css('--p1'));
      const bottom = qrGoalTint(this.flipped ? this.css('--p1') : this.css('--p0'));
      const topAlpha = qrGoalAlpha(top, ca, cb, goalMode);
      const bottomAlpha = qrGoalAlpha(bottom, ca, cb, goalMode);
"""
new_canvas = """      const topPlayer = this.flipped ? 0 : 1, bottomPlayer = 1 - topPlayer;
      const top = qrGoalTint(this.css('--p' + topPlayer));
      const bottom = qrGoalTint(this.css('--p' + bottomPlayer));
      const topAlpha = qrGoalPlayerAlpha(top, ca, cb, goalMode, topPlayer);
      const bottomAlpha = qrGoalPlayerAlpha(bottom, ca, cb, goalMode, bottomPlayer);
"""
if old_canvas not in s:
    raise SystemExit('missing canvas goal block')
s = s.replace(old_canvas, new_canvas, 1)

old_svg = """      const top = qrGoalTint(this.flipped ? cssOf('--p0') : cssOf('--p1'));
      const bottom = qrGoalTint(this.flipped ? cssOf('--p1') : cssOf('--p0'));
      const topAlpha = qrGoalAlpha(top, ca, cb, goalMode);
      const bottomAlpha = qrGoalAlpha(bottom, ca, cb, goalMode);
"""
new_svg = """      const topPlayer = this.flipped ? 0 : 1, bottomPlayer = 1 - topPlayer;
      const top = qrGoalTint(cssOf('--p' + topPlayer));
      const bottom = qrGoalTint(cssOf('--p' + bottomPlayer));
      const topAlpha = qrGoalPlayerAlpha(top, ca, cb, goalMode, topPlayer);
      const bottomAlpha = qrGoalPlayerAlpha(bottom, ca, cb, goalMode, bottomPlayer);
"""
if old_svg not in s:
    raise SystemExit('missing svg goal block')
s = s.replace(old_svg, new_svg, 1)
p.write_text(s)

# Update the targeted regression: the blue wash intentionally loses 8% alpha,
# while the player-coloured fills and no-edge-rail contract stay unchanged.
t = Path('gui_web/test_micro_polish_semantics.py')
ts = t.read_text()
old_min = "and goal_alpha and min(goal_alpha) >= .27"
if old_min not in ts:
    raise SystemExit('missing goal alpha minimum assertion')
ts = ts.replace(old_min, "and goal_alpha and min(goal_alpha) >= .24", 1)
anchor = "                  and 'height=\"2\"' not in goal_group)\n"
if anchor not in ts:
    raise SystemExit('missing goal tint check anchor')
extra = """            blue_scale = page.evaluate(\"\"\"() => {
              const st = getComputedStyle(document.documentElement);
              const ca = st.getPropertyValue('--cell-a').trim();
              const cb = st.getPropertyValue('--cell-b').trim();
              const blue = qrGoalTint(st.getPropertyValue('--p1').trim());
              return qrGoalPlayerAlpha(blue, ca, cb, 'subtle', 1) / qrGoalAlpha(blue, ca, cb, 'subtle');
            }\"\"\")
            check(\"blue goal tint is slightly reduced\", abs(blue_scale - .92) < .001)
"""
ts = ts.replace(anchor, anchor + extra, 1)
t.write_text(ts)
