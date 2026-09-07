from pathlib import Path

p = Path("src/search.hpp")
s = p.read_text()
old = """        if (qply >= effCap) return best;\n\n        if (s.wallsLeft[side] <= 0) return best;  // sem muro pra jogar, nada a estender\n"""
new = """        if (qply >= effCap) return best;\n\n        // vNext: when the mover is out of walls, qsearch used to stop here\n        // immediately. That creates a horizon exactly at the wall-depletion\n        // boundary: a forcing pawn advance (including a jump) is invisible,\n        // as is the opponent's critical wall reply one ply later. Extend only\n        // pawn moves that STRICTLY reduce this mover's shortest-path length.\n        // The monotone condition excludes lateral/backward shuffles and keeps\n        // this continuation bounded by the normal qsearch ply cap.\n        if (s.wallsLeft[side] <= 0) {\n            const int sideDistBefore = cachedShortestPathLen(sideCache);\n            MoveList pawnMoves;\n            pawnStepMoves(s, side, pawnMoves);\n            for (size_t i = 0; i < pawnMoves.size(); i++) {\n                const Move& m = pawnMoves[i];\n                State ns = applyMove(s, m);\n                PlayerPathCache sideCacheAfter;\n                computeDistCached(ns.wallsH, ns.wallsV, ns.pawn[side], side,\n                                  &xdistCache, sideCacheAfter);\n                if (cachedShortestPathLen(sideCacheAfter) >= sideDistBefore) continue;\n\n                AccPair* childAcc = nullptr;\n                if (curAcc) {\n                    childAcc = curAcc + 1;\n                    makeChildAccPair(*curAcc, *childAcc, s, m, &xdistCache);\n                }\n                reptbl.push(ns.hash, /*irreversible=*/false);\n                int score = -quiescence(ns, -beta, -localAlpha, qply + 1,\n                                        stats, reptbl, !rootParity, childAcc);\n                reptbl.pop();\n                if (stopped) return 0;\n                if (score > best) best = score;\n                if (score > localAlpha) localAlpha = score;\n                if (localAlpha >= beta) return best;\n            }\n            return best;\n        }\n"""
if old not in s:
    raise SystemExit("empty-hand qsearch insertion point not found; source drifted")
s2 = s.replace(old, new, 1)
if s2 == s:
    raise SystemExit("patch made no change")
p.write_text(s2)
print("patched src/search.hpp with empty-hand monotone pawn qsearch")
