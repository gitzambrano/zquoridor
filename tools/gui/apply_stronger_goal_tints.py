from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()

# Goal tint strengthening is already present on this validation branch.
for marker in (
    "function qrGoalTint(color) { return qrMixHex(color, '#85888d', .42); }",
    "const target = mode === 'clear' ? 56 : 42;",
    "const lo = mode === 'clear' ? .34 : .27, hi = mode === 'clear' ? .52 : .40;",
):
    if marker not in s:
        raise SystemExit(f'missing validated goal-tint marker: {marker}')

old_wood = """  if (theme === 'wood' || theme === 'walnut') {
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
    raise SystemExit('missing validated wood-v1 texture block')
s = s.replace(old_wood, new_wood, 1)
p.write_text(s)
