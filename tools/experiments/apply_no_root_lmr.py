#!/usr/bin/env python3
from pathlib import Path

p = Path("src/search.hpp")
s = p.read_text()
old = "if (moveIndex > lmrMinMoveIndex && depth >= lmrMinDepth && !isKillerMove) {"
new = "if (ply > 0 && moveIndex > lmrMinMoveIndex && depth >= lmrMinDepth && !isKillerMove) {"
if s.count(old) != 1:
    raise SystemExit(f"expected one LMR guard, found {s.count(old)}")
p.write_text(s.replace(old, new))
print("patched search.hpp: disable LMR at root only")
