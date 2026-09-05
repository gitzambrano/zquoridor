#!/usr/bin/env python3
from pathlib import Path

FILES = [
    Path('src/nnue.hpp'),
    Path('training/train_nnue.py'),
    Path('training/quantize_nnue.py'),
]

for p in FILES:
    s = p.read_text()
    old = 'VALUE_HIDDEN = 32'
    new = 'VALUE_HIDDEN = 64'
    if new in s and old not in s:
        continue
    if old not in s:
        raise SystemExit(f'missing wide64 anchor in {p}')
    p.write_text(s.replace(old, new, 1))

print('antisymmetric value width changed 32 -> 64')
