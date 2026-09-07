from pathlib import Path

p = Path("src/search.hpp")
s = p.read_text()
old = """        if (qply >= effCap) return best;\n\n        if (s.wallsLeft[side] <= 0) return best;  // sem muro pra jogar, nada a estender\n"""
new = """        if (qply >= effCap) return best;\n\n        // vNext: Quoridor has tactical pawn races as well as tactical walls.\n        // Extend only pawn moves that STRICTLY shorten the mover's own\n        // shortest path. This keeps the qsearch monotone and bounded: quiet\n        // lateral/backward shuffles never enter the extension, while jumps\n        // and forward steps that change a race are resolved before stand-pat\n        // is trusted. This mirrors the useful part of Titanium's noisy-move\n        // definition without opening the full pawn branching factor.\n        {\n            const int sideDistBefore = cachedShortestPathLen(sideCache);\n            MoveList pawnMoves;\n            pawnStepMoves(s, side, pawnMoves);\n            for (size_t i = 0; i < pawnMoves.size(); i++) {\n                const Move& m = pawnMoves[i];\n                State ns = applyMove(s, m);\n                PlayerPathCache sideCacheAfter;\n                computeDistCached(ns.wallsH, ns.wallsV, ns.pawn[side], side,\n                                  &xdistCache, sideCacheAfter);\n                if (cachedShortestPathLen(sideCacheAfter) >= sideDistBefore) continue;\n\n                AccPair* childAcc = nullptr;\n                if (curAcc) {\n                    childAcc = curAcc + 1;\n                    makeChildAccPair(*curAcc, *childAcc, s, m, &xdistCache);\n                }\n                // Pawn moves are reversible for repetition bookkeeping.\n                reptbl.push(ns.hash, /*irreversible=*/false);\n                int score = -quiescence(ns, -beta, -localAlpha, qply + 1,\n                                        stats, reptbl, !rootParity, childAcc);\n                reptbl.pop();\n                if (stopped) return 0;\n                if (score > best) best = score;\n                if (score > localAlpha) localAlpha = score;\n                if (localAlpha >= beta) return best;\n            }\n        }\n\n        if (s.wallsLeft[side] <= 0) return best;  // no wall left: pawn race was already extended above\n"""
if old not in s:
    raise SystemExit("qsearch insertion point not found; source drifted")
s2 = s.replace(old, new, 1)
if s2 == s:
    raise SystemExit("patch made no change")
p.write_text(s2)
print("patched src/search.hpp with monotone pawn-progress qsearch")
