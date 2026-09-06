from pathlib import Path

p = Path('gui_web/board.js')
s = p.read_text()
repls = {
    "function qrGoalTint(color) { return qrMixHex(color, '#85888d', .55); }":
    "function qrGoalTint(color) { return qrMixHex(color, '#85888d', .45); }",
    "if (!t || !a || !b) return mode === 'clear' ? .23 : .16;":
    "if (!t || !a || !b) return mode === 'clear' ? .32 : .24;",
    "if (span < 1) return mode === 'clear' ? .23 : .16;":
    "if (span < 1) return mode === 'clear' ? .32 : .24;",
    "const target = mode === 'clear' ? 31.5 : 21;":
    "const target = mode === 'clear' ? 46 : 34;",
    "const lo = mode === 'clear' ? .18 : .13, hi = mode === 'clear' ? .32 : .22;":
    "const lo = mode === 'clear' ? .28 : .22, hi = mode === 'clear' ? .46 : .36;",
}
for old, new in repls.items():
    if old not in s:
        raise SystemExit(f'missing expected source: {old}')
    s = s.replace(old, new, 1)
p.write_text(s)
