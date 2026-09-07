from pathlib import Path

p = Path("src/mcab.hpp")
s = p.read_text()
old = """        if (!params.progressiveWidening) {\n            node.moves = legalMoves(node.state);\n"""
new = """        // vNext experiment: never narrow the root. At the root we need the\n        // complete legal move set so a low-policy but tactically essential\n        // wall cannot be hidden by widening. Below the root, progressive\n        // widening focuses the simulation budget on the strongest policy\n        // candidates and defers expensive wall path-legality checks.\n        if (!params.progressiveWidening || depthInTree == 0) {\n            node.moves = legalMoves(node.state);\n"""
if old not in s:
    raise SystemExit("progressive-widening insertion point not found; source drifted")
s2 = s.replace(old, new, 1)
if s2 == s:
    raise SystemExit("patch made no change")
p.write_text(s2)
print("patched src/mcab.hpp: full root, progressive widening below root")
