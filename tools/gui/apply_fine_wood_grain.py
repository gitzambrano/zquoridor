from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()

repls = {
    "const count = theme === 'walnut' ? 38 : 52;": "const count = theme === 'walnut' ? 38 : 84;",
    "const base = theme === 'walnut' ? .050 : .125;": "const base = theme === 'walnut' ? .050 : .135;",
    "const drift = (rnd() - .5) * C * .34;": "const drift = (rnd() - .5) * C * (theme === 'walnut' ? .34 : .22);",
    "const amp = C * (.018 + rnd() * .038);": "const amp = C * (theme === 'walnut' ? (.018 + rnd() * .038) : (.008 + rnd() * .020));",
    "const waves = .55 + rnd() * 1.20;": "const waves = theme === 'walnut' ? .55 + rnd() * 1.20 : .45 + rnd() * .85;",
    "out.push({ kind: 'poly', points: pts, width: .54 + rnd() * .74,": "out.push({ kind: 'poly', points: pts, width: theme === 'walnut' ? .54 + rnd() * .74 : .18 + rnd() * .22,",
    "const off = C * (.045 + rnd() * .090) * (rnd() < .5 ? -1 : 1);": "const off = C * (theme === 'walnut' ? (.045 + rnd() * .090) : (.022 + rnd() * .050)) * (rnd() < .5 ? -1 : 1);",
    "width: .36 + rnd() * .50, tone: t, alpha: a * .50 });": "width: theme === 'walnut' ? .36 + rnd() * .50 : .10 + rnd() * .12, tone: t, alpha: a * (theme === 'walnut' ? .50 : .38) });",
    "const fibres = theme === 'walnut' ? 34 : 70;": "const fibres = theme === 'walnut' ? 34 : 115;",
    "const len = C * (.18 + rnd() * .55);": "const len = C * (theme === 'walnut' ? (.18 + rnd() * .55) : (.12 + rnd() * .38));",
    ".34 + rnd() * .44, tone(.62), base * s * (.34 + .26 * rnd()));": "theme === 'walnut' ? .34 + rnd() * .44 : .08 + rnd() * .12, tone(.62), base * s * (theme === 'walnut' ? (.34 + .26 * rnd()) : (.18 + .18 * rnd())));",
    "const flecks = theme === 'walnut' ? 16 : 28;": "const flecks = theme === 'walnut' ? 16 : 14;",
    "dot(bx + rnd() * bw, bx + rnd() * bw, .22 + rnd() * .42,": "dot(bx + rnd() * bw, bx + rnd() * bw, theme === 'walnut' ? .22 + rnd() * .42 : .10 + rnd() * .16,",
}
for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'missing expected wood source: {old}')
    s = s.replace(old, new, 1)

old_comment = """    // Soft organic grain inspired by lightly finished wood: long fibres wander
    // gently, a few close companions suggest growth layers, and fine broken
    // fibres keep the surface from looking digitally flat.
"""
new_comment = """    // Soft organic grain inspired by lightly finished wood. Default wood uses
    // many hairline fibres with low contrast and small wander, so the surface
    // reads as wood rather than as a set of drawn scratches. Walnut keeps its
    // broader, darker grain profile.
"""
if old_comment not in s:
    raise SystemExit('missing wood comment')
s = s.replace(old_comment, new_comment, 1)
p.write_text(s)
