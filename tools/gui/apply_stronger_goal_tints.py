from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()
repls = {
    "function qrGoalTint(color) { return qrMixHex(color, '#85888d', .55); }":
    "function qrGoalTint(color) { return qrMixHex(color, '#85888d', .42); }",
    "if (!t || !a || !b) return mode === 'clear' ? .23 : .16;":
    "if (!t || !a || !b) return mode === 'clear' ? .38 : .30;",
    "if (span < 1) return mode === 'clear' ? .23 : .16;":
    "if (span < 1) return mode === 'clear' ? .38 : .30;",
    "const target = mode === 'clear' ? 31.5 : 21;":
    "const target = mode === 'clear' ? 56 : 42;",
    "const lo = mode === 'clear' ? .18 : .13, hi = mode === 'clear' ? .32 : .22;":
    "const lo = mode === 'clear' ? .34 : .27, hi = mode === 'clear' ? .52 : .40;",
}
for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'missing expected source: {old}')
    s = s.replace(old, new, 1)

old_wood = """  if (theme === 'wood' || theme === 'walnut') {
    const count = theme === 'walnut' ? 34 : 28;
    const base = theme === 'walnut' ? .050 : .044;
    for (let i = 0; i < count; i++) {
      const y = bx + rnd() * bw;
      const drift = (rnd() - .5) * bw * .035;
      const a = base * s * (.70 + .30 * rnd());
      const t = tone(.58);
      line(bx - 2, y, bx + bw + 2, y + drift, .45 + rnd() * .75, t, a);
      if (rnd() > .70) {
        const off = (rnd() - .5) * C * .18;
        line(bx, y + off, bx + bw, y + drift + off,
             .35 + rnd() * .45, t, a * .55);
      }
    }
"""
new_wood = """  if (theme === 'wood' || theme === 'walnut') {
    // Default wood uses long, gently wandering fibres plus a sparse layer of
    // shorter secondary fibres. It should read as real grain only after the
    // eye settles on the board, rather than as a set of drawn parallel lines.
    const count = theme === 'walnut' ? 38 : 36;
    const base = theme === 'walnut' ? .050 : .040;
    for (let i = 0; i < count; i++) {
      const y0 = bx + rnd() * bw;
      const drift = (rnd() - .5) * C * .34;
      const amp = C * (.020 + rnd() * .040);
      const phase = rnd() * Math.PI * 2;
      const waves = .55 + rnd() * 1.25;
      const pts = [];
      const segments = 9;
      for (let j = 0; j <= segments; j++) {
        const u = j / segments;
        const x = bx - 3 + u * (bw + 6);
        const y = y0 + drift * u + Math.sin(phase + u * Math.PI * 2 * waves) * amp;
        pts.push({ x, y });
      }
      const a = base * s * (.72 + .28 * rnd());
      out.push({ kind: 'poly', points: pts, width: .38 + rnd() * .62,
                 tone: tone(.61), alpha: a });
    }

    // Fine broken fibres stop the surface looking digitally smooth. Keep them
    // short and faint so the checker pattern and goal rows remain dominant.
    const fibres = theme === 'walnut' ? 30 : 38;
    for (let i = 0; i < fibres; i++) {
      const x = bx + rnd() * bw, y = bx + rnd() * bw;
      const len = C * (.22 + rnd() * .52);
      line(x, y, x + len, y + (rnd() - .5) * C * .08,
           .28 + rnd() * .38, tone(.62), base * s * (.25 + .25 * rnd()));
    }
"""
if old_wood not in s:
    raise SystemExit('missing expected wood texture block')
s = s.replace(old_wood, new_wood, 1)

p.write_text(s)
