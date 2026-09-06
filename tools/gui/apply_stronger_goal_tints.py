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
    // Soft organic grain inspired by lightly finished wood: long fibres wander
    // gently, a few close companions suggest growth layers, and fine broken
    // fibres keep the surface from looking digitally flat.
    const count = theme === 'walnut' ? 38 : 40;
    const base = theme === 'walnut' ? .050 : .062;
    for (let i = 0; i < count; i++) {
      const y0 = bx + rnd() * bw;
      const drift = (rnd() - .5) * C * .34;
      const amp = C * (.018 + rnd() * .038);
      const phase = rnd() * Math.PI * 2;
      const waves = .55 + rnd() * 1.20;
      const pts = [];
      const segments = 10;
      for (let j = 0; j <= segments; j++) {
        const u = j / segments;
        const x = bx - 3 + u * (bw + 6);
        const y = y0 + drift * u + Math.sin(phase + u * Math.PI * 2 * waves) * amp;
        pts.push({ x, y });
      }
      const a = base * s * (.72 + .28 * rnd());
      const t = tone(.61);
      out.push({ kind: 'poly', points: pts, width: .36 + rnd() * .60,
                 tone: t, alpha: a });

      if (rnd() > .68) {
        const off = C * (.045 + rnd() * .090) * (rnd() < .5 ? -1 : 1);
        out.push({ kind: 'poly', points: pts.map(q => ({ x:q.x, y:q.y + off })),
                   width: .28 + rnd() * .42, tone: t, alpha: a * .38 });
      }
    }

    const fibres = theme === 'walnut' ? 34 : 48;
    for (let i = 0; i < fibres; i++) {
      const x = bx + rnd() * bw, y = bx + rnd() * bw;
      const len = C * (.18 + rnd() * .55);
      line(x, y, x + len, y + (rnd() - .5) * C * .075,
           .26 + rnd() * .34, tone(.62), base * s * (.24 + .20 * rnd()));
    }

    const flecks = theme === 'walnut' ? 16 : 28;
    for (let i = 0; i < flecks; i++)
      dot(bx + rnd() * bw, bx + rnd() * bw, .22 + rnd() * .42,
          tone(.58), base * s * (.10 + .12 * rnd()));
"""
if old_wood not in s:
    raise SystemExit('missing expected wood texture block')
s = s.replace(old_wood, new_wood, 1)

p.write_text(s)
