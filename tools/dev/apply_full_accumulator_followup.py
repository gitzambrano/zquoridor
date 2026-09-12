#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace(path, old, new, expected=1):
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    found = text.count(old)
    if found != expected:
        raise RuntimeError(f"{path}: expected {expected} matches, found {found}: {old!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


old = "                forwardPolicyQuant(mcabAccStack[depthInTree].acc[node.side], policyOut);"
new = "                resolvePending(mcabAccStack[depthInTree], 1 - node.side, mcabPathCache(engine, 0));\n                forwardPolicyQuant(mcabAccStack[depthInTree].acc[node.side],\n                                   mcabAccStack[depthInTree].acc[1 - node.side], policyOut);"
replace("src/mcab.hpp", old, new, expected=2)

replace(
    "tests/test_mcab_core.cpp",
    "float winProb = nnueWinProbQuant(ap.acc[side]);",
    "float winProb = nnueWinProbQuant(ap.acc[side], ap.acc[1 - side]);",
)

print("MCAB full accumulator patch applied.")
