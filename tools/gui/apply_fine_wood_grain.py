from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()

# Refine the first hairline pass: keep the lines thin, but make them visible
# through density, a mostly-dark tone bias and slightly higher alpha rather
# than by increasing stroke width. Walnut is intentionally left unchanged.
repls = {
    "const count = theme === 'walnut' ? 38 : 84;": "const count = theme === 'walnut' ? 38 : 96;",
    "const base = theme === 'walnut' ? .050 : .135;": "const base = theme === 'walnut' ? .050 : .160;",
    "const drift = (rnd() - .5) * C * (theme === 'walnut' ? .34 : .22);": "const drift = (rnd() - .5) * C * (theme === 'walnut' ? .34 : .16);",
    "const amp = C * (theme === 'walnut' ? (.018 + rnd() * .038) : (.008 + rnd() * .020));": "const amp = C * (theme === 'walnut' ? (.018 + rnd() * .038) : (.006 + rnd() * .014));",
    "const waves = theme === 'walnut' ? .55 + rnd() * 1.20 : .45 + rnd() * .85;": "const waves = theme === 'walnut' ? .55 + rnd() * 1.20 : .40 + rnd() * .65;",
    "const t = tone(.61);": "const t = tone(theme === 'walnut' ? .61 : .78);",
    "width: theme === 'walnut' ? .54 + rnd() * .74 : .18 + rnd() * .22,": "width: theme === 'walnut' ? .54 + rnd() * .74 : .28 + rnd() * .18,",
    "const off = C * (theme === 'walnut' ? (.045 + rnd() * .090) : (.022 + rnd() * .050)) * (rnd() < .5 ? -1 : 1);": "const off = C * (theme === 'walnut' ? (.045 + rnd() * .090) : (.018 + rnd() * .040)) * (rnd() < .5 ? -1 : 1);",
    "width: theme === 'walnut' ? .36 + rnd() * .50 : .10 + rnd() * .12, tone: t, alpha: a * (theme === 'walnut' ? .50 : .38) });": "width: theme === 'walnut' ? .36 + rnd() * .50 : .14 + rnd() * .10, tone: t, alpha: a * (theme === 'walnut' ? .50 : .42) });",
    "const fibres = theme === 'walnut' ? 34 : 115;": "const fibres = theme === 'walnut' ? 34 : 135;",
    "const len = C * (theme === 'walnut' ? (.18 + rnd() * .55) : (.12 + rnd() * .38));": "const len = C * (theme === 'walnut' ? (.18 + rnd() * .55) : (.10 + rnd() * .32));",
    "theme === 'walnut' ? .34 + rnd() * .44 : .08 + rnd() * .12, tone(.62), base * s * (theme === 'walnut' ? (.34 + .26 * rnd()) : (.18 + .18 * rnd())));": "theme === 'walnut' ? .34 + rnd() * .44 : .12 + rnd() * .08, tone(theme === 'walnut' ? .62 : .72), base * s * (theme === 'walnut' ? (.34 + .26 * rnd()) : (.20 + .18 * rnd())));",
    "const flecks = theme === 'walnut' ? 16 : 14;": "const flecks = theme === 'walnut' ? 16 : 6;",
    "theme === 'walnut' ? .22 + rnd() * .42 : .10 + rnd() * .16,": "theme === 'walnut' ? .22 + rnd() * .42 : .08 + rnd() * .10,",
}
for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'missing expected fine-grain source: {old}')
    s = s.replace(old, new, 1)

p.write_text(s)
