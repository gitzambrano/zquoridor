#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "src" / "mcab.hpp"
text = p.read_text(encoding="utf-8")
old = "struct hasNnueEvalInt<Acc, std::void_t<decltype(nnueEvalInt(std::declval<const Acc&>(), 0))>>"
new = "struct hasNnueEvalInt<Acc, std::void_t<decltype(nnueEvalInt(std::declval<Acc&>(), 0))>>"
if text.count(old) != 1:
    raise RuntimeError(f"expected one mutable-eval detection site, found {text.count(old)}")
p.write_text(text.replace(old, new), encoding="utf-8")
print("updated MCAB nnueEvalInt detection for mutable AccPair")
