#!/usr/bin/env python3
from pathlib import Path

p = Path('training/train_nnue.py')
s = p.read_text()

old = '''    head_floats = HIDDEN * 32 + 32 + 32 + 1  # wv1 + bv1 + wv2 + bv2 de UMA cabeca 256->32->1
    base_floats = NUM_FEATURES * HIDDEN + HIDDEN + head_floats
    tail_floats = POLICY_OUT * HIDDEN + POLICY_OUT
'''
new = '''    head_floats = HEAD_INPUT * 32 + 32 + 32 + 1  # wv1 + bv1 + wv2 + bv2 de UMA cabeca 512->32->1
    base_floats = NUM_FEATURES * HIDDEN + HIDDEN + head_floats
    tail_floats = POLICY_OUT * HEAD_INPUT + POLICY_OUT
'''
if old not in s:
    raise SystemExit('full-acc loader size block not found')
s = s.replace(old, new, 1)

old = '''        def read_head():
            wv1 = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
            bv1 = np.fromfile(f, dtype="<f4", count=32)
            wv2 = np.fromfile(f, dtype="<f4", count=32)
            bv2 = np.fromfile(f, dtype="<f4", count=1)
            return wv1, bv1, wv2, bv2
'''
new = '''        def read_head():
            wv1 = np.fromfile(f, dtype="<f4", count=HEAD_INPUT * 32).reshape(HEAD_INPUT, 32)
            bv1 = np.fromfile(f, dtype="<f4", count=32)
            wv2 = np.fromfile(f, dtype="<f4", count=32)
            bv2 = np.fromfile(f, dtype="<f4", count=1)
            return wv1, bv1, wv2, bv2
'''
if old not in s:
    raise SystemExit('full-acc read_head block not found')
s = s.replace(old, new, 1)

old = '''        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
'''
new = '''        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HEAD_INPUT).reshape(POLICY_OUT, HEAD_INPUT)
'''
if old not in s:
    raise SystemExit('full-acc policy loader block not found')
s = s.replace(old, new, 1)

# Keep the top-level architecture summary accurate.
s = s.replace('Linear(256, 32) -> ClippedReLU -> Linear(32, 1)',
              'Linear(512, 32) -> ClippedReLU -> Linear(32, 1)', 1)
s = s.replace('policy head: Linear(256, 209)', 'policy head: Linear(512, 209)', 1)

p.write_text(s)
print('patched training/train_nnue.py for 512-wide full-accumulator init-from')
