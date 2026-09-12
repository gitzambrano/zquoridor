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


# Keep mcab.hpp source-compatible with older refs/test doubles. Full-accumulator
# engines expose resolvePending(pair, side) and forwardPolicyQuant(own, opp,
# out); legacy engines expose only forwardPolicyQuant(own, out).
anchor = "namespace mcab {\n"
helpers = r'''namespace mcab {

template <typename AccPairT>
inline auto mcabResolveOtherView(AccPairT& ap, int side, int)
    -> decltype(resolvePending(ap, side), void()) {
    resolvePending(ap, side);
}

template <typename AccPairT>
inline void mcabResolveOtherView(AccPairT&, int, long) {}

template <typename AccT, size_t PolicyDim>
inline auto mcabForwardPolicy(const AccT& own, const AccT& opp,
                              std::array<float, PolicyDim>& out, int)
    -> decltype(forwardPolicyQuant(own, opp, out), void()) {
    forwardPolicyQuant(own, opp, out);
}

template <typename AccT, size_t PolicyDim>
inline auto mcabForwardPolicy(const AccT& own, const AccT&,
                              std::array<float, PolicyDim>& out, long)
    -> decltype(forwardPolicyQuant(own, out), void()) {
    forwardPolicyQuant(own, out);
}
'''
replace("src/mcab.hpp", anchor, helpers, expected=1)

old = "                forwardPolicyQuant(mcabAccStack[depthInTree].acc[node.side], policyOut);"
new = "                mcabResolveOtherView(mcabAccStack[depthInTree], 1 - node.side, 0);\n                mcabForwardPolicy(mcabAccStack[depthInTree].acc[node.side],\n                                  mcabAccStack[depthInTree].acc[1 - node.side], policyOut, 0);"
replace("src/mcab.hpp", old, new, expected=2)

replace(
    "tests/test_mcab_core.cpp",
    "float winProb = nnueWinProbQuant(ap.acc[side]);",
    "float winProb = nnueWinProbQuant(ap.acc[side], ap.acc[1 - side]);",
)

print("MCAB full accumulator patch applied.")
