#!/usr/bin/env python3
"""contrast_check.py -- accessibility gate for the GUI tokens (gui-premium.md,
section 10). Parses the token table out of gui_web/style.html and asserts:

  - every board theme x UI theme pair has wall/cell >= 3:1 and wall/groove
    >= 3:1 contrast (the R5 constraint: contrast first, decoration after);
  - the core text colours meet 4.5:1 against their surfaces.

Run standalone or as part of the build (build_standalone.py calls it before
writing the bundles). Exits 1 listing every failure.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSS = HERE.parent.parent / "gui_web" / "style.html"

def parse_css_blocks(text):
    """Return list of (selector, {prop: value}) for top-level blocks."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)   # strip comments
    blocks = []
    for m in re.finditer(r'([^{}]+)\{([^{}]*)\}', text):
        sel = m.group(1).strip().splitlines()[-1].strip()
        body = m.group(2)
        props = {}
        for line in body.split(';'):
            if ':' in line:
                k, v = line.split(':', 1)
                props[k.strip()] = v.strip()
        blocks.append((sel, props))
    return blocks

def resolve_color(tok, variables):
    tok = tok.strip()
    while tok.startswith('var('):
        name = tok[4:-1].strip()
        if ',' in name:
            name, fallback = [s.strip() for s in name.split(',', 1)]
            tok = variables.get(name, fallback)
        else:
            tok = variables.get(name, '')
    return tok

def hex_to_rgb(c):
    c = c.strip()
    m = re.fullmatch(r'#([0-9a-fA-F]{6})', c)
    if m:
        v = int(m.group(1), 16)
        return ((v >> 16) & 255, (v >> 8) & 255, v & 255)
    m = re.fullmatch(r'#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])', c)
    if m:
        return tuple(int(ch * 2, 16) for ch in m.groups())
    return None

def luminance(rgb):
    def chan(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)

def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

def main():
    raw = CSS.read_text(encoding='utf-8')
    m = re.search(r'<style>(.*?)</style>', raw, re.S)
    if not m:
        print('contrast check: no <style> block found')
        return 1
    text = m.group(1)
    blocks = parse_css_blocks(text)

    themes = {}     # key -> var overrides
    bases = {'dark': {}, 'light': {}}
    for sel, props in blocks:
        sel = sel.strip().split('}')[-1].strip()   # guard against nested junk
        m = re.fullmatch(r'html\[data-board="([\w]+)"\](?:\[data-ui="light"\])?', sel)
        if m:
            is_light = '[data-ui="light"]' in sel
            key = m.group(1) + ('#light' if is_light else '')
            themes[key] = dict(props)
        elif sel == ':root':
            bases['dark'] = dict(props)
        elif sel == 'html[data-ui="light"]':
            bases['light'] = dict(props)

    failures = []
    boards = ['obsidian', 'walnut', 'ivory', 'slate', 'emerald', 'parchment', 'marble', 'noir']
    uis = ['dark', 'light']
    for ui in uis:
        for board in boards:
            variables = {}
            dark_vars = dict(bases['dark'])
            light_vars = dict(themes.get(board + '#light', {})) if False else {}
            variables.update(dark_vars)
            variables.update(themes.get(board, {}))
            if ui == 'light':
                variables.update(bases['light'])
                variables.update(themes.get(board + '#light', {}))
            get = lambda name: hex_to_rgb(resolve_color(variables.get(name, ''), variables))
            pairs = [
                ('wall/cell', get('--wall'), get('--cell-a'), 3.0),
                ('wall/groove', get('--wall'), get('--groove'), 3.0),
                # The rank and file labels sit on the frame margin. They are
                # small by design, so they need the full text ratio to stay
                # readable.
                ('coord/frame', get('--coord'), get('--frame'), 4.5),
                # The two cell tones. This floor is deliberately low: the
                # chequer must be readable without becoming a second pattern
                # competing with the pieces. Below it the board reads as one
                # flat slab, which is what obsidian and noir used to do.
                ('cell-a/cell-b', get('--cell-a'), get('--cell-b'), 1.20),
            ]
            # Chrome text, checked in BOTH themes. The light theme used to go
            # unchecked, and --muted in the dark theme sat at 2.54:1, which is
            # the colour of the move-log numbers and the empty-state line.
            pairs += [
                ('txt/surf', get('--txt'), get('--surf'), 4.5),
                ('txt2/surf', get('--txt2'), get('--surf'), 4.5),
                ('muted/surf', get('--muted'), get('--surf'), 4.5),
            ]
            for name, a, b, floor in pairs:
                if a is None or b is None:
                    failures.append(f'{board}/{ui}: {name} - unresolvable colour')
                    continue
                ratio = contrast(a, b)
                if ratio < floor:
                    failures.append(
                        f'{board}/{ui}: {name} = {ratio:.2f} < {floor}')

    if failures:
        print('CONTRAST CHECK FAILED:')
        for f in failures:
            print('  -', f)
        return 1
    print(f'contrast check OK ({len(boards)}x{len(uis)} theme combinations)')
    return 0

if __name__ == '__main__':
    sys.exit(main())
