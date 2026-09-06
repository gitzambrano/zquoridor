from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()

# First application from main: add the blue-only alpha trim and initial wood
# visibility bump. On later runs, leave those changes in place.
if 'function qrGoalPlayerAlpha(' not in s:
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

    canvas_old = """      const top = qrGoalTint(this.flipped ? this.css('--p0') : this.css('--p1'));
      const bottom = qrGoalTint(this.flipped ? this.css('--p1') : this.css('--p0'));
      const topAlpha = qrGoalAlpha(top, ca, cb, goalMode);
      const bottomAlpha = qrGoalAlpha(bottom, ca, cb, goalMode);
"""
    canvas_new = """      const topPlayer = this.flipped ? 0 : 1, bottomPlayer = 1 - topPlayer;
      const top = qrGoalTint(this.css('--p' + topPlayer));
      const bottom = qrGoalTint(this.css('--p' + bottomPlayer));
      const topAlpha = qrGoalPlayerAlpha(top, ca, cb, goalMode, topPlayer);
      const bottomAlpha = qrGoalPlayerAlpha(bottom, ca, cb, goalMode, bottomPlayer);
"""
    if canvas_old not in s:
        raise SystemExit('missing canvas goal block')
    s = s.replace(canvas_old, canvas_new, 1)

    svg_old = """      const top = qrGoalTint(this.flipped ? cssOf('--p0') : cssOf('--p1'));
      const bottom = qrGoalTint(this.flipped ? cssOf('--p1') : cssOf('--p0'));
      const topAlpha = qrGoalAlpha(top, ca, cb, goalMode);
      const bottomAlpha = qrGoalAlpha(bottom, ca, cb, goalMode);
"""
    svg_new = """      const topPlayer = this.flipped ? 0 : 1, bottomPlayer = 1 - topPlayer;
      const top = qrGoalTint(cssOf('--p' + topPlayer));
      const bottom = qrGoalTint(cssOf('--p' + bottomPlayer));
      const topAlpha = qrGoalPlayerAlpha(top, ca, cb, goalMode, topPlayer);
      const bottomAlpha = qrGoalPlayerAlpha(bottom, ca, cb, goalMode, bottomPlayer);
"""
    if svg_old not in s:
        raise SystemExit('missing svg goal block')
    s = s.replace(svg_old, svg_new, 1)

# Final wood tuning. The default subtle mode should show recognisable fibres at
# normal viewing size, while remaining well below pieces, walls and goal rows.
wood_repls = {
    "const count = theme === 'walnut' ? 38 : 40;": "const count = theme === 'walnut' ? 38 : 52;",
    "const count = theme === 'walnut' ? 38 : 46;": "const count = theme === 'walnut' ? 38 : 52;",
    "const base = theme === 'walnut' ? .050 : .062;": "const base = theme === 'walnut' ? .050 : .125;",
    "const base = theme === 'walnut' ? .050 : .090;": "const base = theme === 'walnut' ? .050 : .125;",
    "out.push({ kind: 'poly', points: pts, width: .36 + rnd() * .60,": "out.push({ kind: 'poly', points: pts, width: .54 + rnd() * .74,",
    "out.push({ kind: 'poly', points: pts, width: .46 + rnd() * .68,": "out.push({ kind: 'poly', points: pts, width: .54 + rnd() * .74,",
    "width: .28 + rnd() * .42, tone: t, alpha: a * .38 });": "width: .36 + rnd() * .50, tone: t, alpha: a * .50 });",
    "width: .32 + rnd() * .46, tone: t, alpha: a * .45 });": "width: .36 + rnd() * .50, tone: t, alpha: a * .50 });",
    "const fibres = theme === 'walnut' ? 34 : 48;": "const fibres = theme === 'walnut' ? 34 : 70;",
    "const fibres = theme === 'walnut' ? 34 : 58;": "const fibres = theme === 'walnut' ? 34 : 70;",
    ".26 + rnd() * .34, tone(.62), base * s * (.24 + .20 * rnd()));": ".34 + rnd() * .44, tone(.62), base * s * (.34 + .26 * rnd()));",
    ".30 + rnd() * .40, tone(.62), base * s * (.30 + .24 * rnd()));": ".34 + rnd() * .44, tone(.62), base * s * (.34 + .26 * rnd()));",
}
for old, new in wood_repls.items():
    if old in s:
        s = s.replace(old, new, 1)

if "const base = theme === 'walnut' ? .050 : .125;" not in s:
    raise SystemExit('final wood base was not applied')
if "const fibres = theme === 'walnut' ? 34 : 70;" not in s:
    raise SystemExit('final wood fibre count was not applied')
p.write_text(s)

# The blue row has a deliberate 8% reduction, so keep the slightly lower
# minimum-opacity regression floor. Apply only when the older floor is present.
t = Path('gui_web/test_micro_polish_semantics.py')
ts = t.read_text()
if "and goal_alpha and min(goal_alpha) >= .27" in ts:
    ts = ts.replace("and goal_alpha and min(goal_alpha) >= .27",
                    "and goal_alpha and min(goal_alpha) >= .24", 1)
t.write_text(ts)
