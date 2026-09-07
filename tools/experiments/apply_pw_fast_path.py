from pathlib import Path

p = Path("src/mcab.hpp")
s = p.read_text()

old_limit = """        double raw = (double)params.wideningInitialMoves +\n                     params.wideningCoefficient *\n                         std::pow((double)std::max(0, visits), params.wideningExponent);\n"""
new_limit = """        const double n = (double)std::max(0, visits);\n        // Production experiments use alpha=0.5. std::pow() here is on the\n        // hottest MCAB path (once per visited node, per simulation); avoid\n        // the general transcendental implementation for common exponents.\n        double growth;\n        if (std::abs(params.wideningExponent - 0.5) < 1e-12)\n            growth = std::sqrt(n);\n        else if (std::abs(params.wideningExponent - 1.0) < 1e-12)\n            growth = n;\n        else\n            growth = std::pow(n, params.wideningExponent);\n        double raw = (double)params.wideningInitialMoves +\n                     params.wideningCoefficient * growth;\n"""
if old_limit not in s:
    raise SystemExit("wideningLimit block not found")
s = s.replace(old_limit, new_limit, 1)

old_update = """        int desired = wideningLimit((int)node.candidateMoves.size(), node.totalN);\n        activateWidening(node, desired);\n"""
new_update = """        if (node.nextCandidate >= node.candidateMoves.size()) {\n            node.activeMoves = (int)node.moves.size();\n            return;\n        }\n        int desired = wideningLimit((int)node.candidateMoves.size(), node.totalN);\n        if (desired <= node.activeMoves) return;\n        activateWidening(node, desired);\n"""
if old_update not in s:
    raise SystemExit("updateWidening block not found")
s = s.replace(old_update, new_update, 1)

old_exp = """                float maxLogit = *std::max_element(logits.begin(), logits.end());\n                float sumExp = 0.f;\n                // perf: exp computado UMA vez por elemento (era 2x: soma +\n                // normalização) e via mcabFastExp\n                for (float logit : logits) sumExp += mcabFastExp(logit - maxLogit);\n                node.candidateP.resize(nc);\n                for (size_t i = 0; i < nc; i++)\n                    node.candidateP[i] = sumExp > 0.f ? mcabFastExp(logits[i] - maxLogit) / sumExp\n                                                      : 1.f / (float)nc;\n"""
new_exp = """                float maxLogit = *std::max_element(logits.begin(), logits.end());\n                float sumExp = 0.f;\n                node.candidateP.resize(nc);\n                // Compute exp exactly once per candidate; the previous lazy\n                // path recomputed the same exponential during normalization.\n                for (size_t i = 0; i < nc; i++) {\n                    node.candidateP[i] = mcabFastExp(logits[i] - maxLogit);\n                    sumExp += node.candidateP[i];\n                }\n                if (sumExp > 0.f) {\n                    const float inv = 1.f / sumExp;\n                    for (size_t i = 0; i < nc; i++) node.candidateP[i] *= inv;\n                } else {\n                    for (size_t i = 0; i < nc; i++) node.candidateP[i] = 1.f / (float)nc;\n                }\n"""
if old_exp not in s:
    raise SystemExit("candidate prior exp block not found")
s = s.replace(old_exp, new_exp, 1)

p.write_text(s)
print("patched src/mcab.hpp progressive-widening hot path")
