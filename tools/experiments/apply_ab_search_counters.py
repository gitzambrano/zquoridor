#!/usr/bin/env python3
"""Instrumentation-only patch for Zquoridor alpha-beta search.

Adds counters to SearchStats and prints them in the external adapter. The patch
must not change search decisions; it only increments uint64_t counters.
"""
from pathlib import Path

sp = Path("src/search.hpp")
s = sp.read_text()

def one(old: str, new: str, label: str):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match, got {n}")
    s = s.replace(old, new)

one(
    "    uint64_t policyNodes = 0;\n",
    "    uint64_t policyNodes = 0;\n"
    "    // Diagnostic-only counters (exp/vnext search investigation).\n"
    "    uint64_t qNodes = 0;\n"
    "    uint64_t ttProbes = 0, ttHits = 0, ttCutoffs = 0;\n"
    "    uint64_t betaCutoffs = 0, firstMoveBetaCutoffs = 0;\n"
    "    uint64_t pawnMovesSearched = 0, wallMovesSearched = 0;\n"
    "    uint64_t wallGenNodes = 0, wallMovesGenerated = 0;\n"
    "    uint64_t lmrReducedMoves = 0, lmrResearches = 0, pvsResearches = 0;\n",
    "SearchStats fields")

one(
    "    int quiescence(const State& s, int alpha, int beta, int qply, SearchStats& stats, RepetitionTable& reptbl,\n"
    "                   bool rootParity, AccPair* curAcc = nullptr) {\n"
    "        stats.nodes++;\n",
    "    int quiescence(const State& s, int alpha, int beta, int qply, SearchStats& stats, RepetitionTable& reptbl,\n"
    "                   bool rootParity, AccPair* curAcc = nullptr) {\n"
    "        stats.nodes++;\n"
    "        stats.qNodes++;\n",
    "qnodes")

one(
    "        int alphaOrig = alpha;\n        TTEntry& e = probe(s.hash);\n",
    "        int alphaOrig = alpha;\n        stats.ttProbes++;\n        TTEntry& e = probe(s.hash);\n",
    "tt probes")

one(
    "        if (e.valid && e.key == s.hash) {\n            hasTTMove = true;\n",
    "        if (e.valid && e.key == s.hash) {\n            stats.ttHits++;\n            hasTTMove = true;\n",
    "tt hits")

one(
    "                if (e.flag == EXACT) return e.score;\n",
    "                if (e.flag == EXACT) { stats.ttCutoffs++; return e.score; }\n",
    "tt exact cutoff")

one(
    "                if (alpha >= beta) return e.score;\n",
    "                if (alpha >= beta) { stats.ttCutoffs++; return e.score; }\n",
    "tt bound cutoff")

one(
    "        auto tryMove = [&](const Move& m, int moveIndex, int catHeat,\n"
    "                           float polLogit = 0.f, float polMaxLogit = 0.f,\n"
    "                           bool polHave = false) -> bool {\n",
    "        auto tryMove = [&](const Move& m, int moveIndex, int catHeat,\n"
    "                           float polLogit = 0.f, float polMaxLogit = 0.f,\n"
    "                           bool polHave = false) -> bool {\n"
    "            if (m.isWall) stats.wallMovesSearched++; else stats.pawnMovesSearched++;\n",
    "searched moves")

one(
    "                        if (reduction > maxRed) reduction = maxRed;\n"
    "                    }\n"
    "                }\n"
    "                // PVS (Prioridade 8): janela nula em profundidade (talvez\n",
    "                        if (reduction > maxRed) reduction = maxRed;\n"
    "                    }\n"
    "                }\n"
    "                if (reduction > 0) stats.lmrReducedMoves++;\n"
    "                // PVS (Prioridade 8): janela nula em profundidade (talvez\n",
    "lmr reduced")

one(
    "                if (!stopped && reduction > 0 && score > alpha) {\n",
    "                if (!stopped && reduction > 0 && score > alpha) {\n"
    "                    stats.lmrResearches++;\n",
    "lmr research")

one(
    "                if (!stopped && score > alpha && score < beta) {\n",
    "                if (!stopped && score > alpha && score < beta) {\n"
    "                    stats.pvsResearches++;\n",
    "pvs research")

one(
    "            if (alpha >= beta) {\n                recordCutoff(m, ply, side, depth);\n",
    "            if (alpha >= beta) {\n"
    "                stats.betaCutoffs++;\n"
    "                if (moveIndex == 1) stats.firstMoveBetaCutoffs++;\n"
    "                recordCutoff(m, ply, side, depth);\n",
    "beta cutoffs")

one(
    "            PlayerPathCache cache0, cache1;\n"
    "            legalWallMoves(s, side, wallMoves, nullptr, nullptr, nullptr, nullptr, &cache0, &cache1, &xdistCache);\n",
    "            PlayerPathCache cache0, cache1;\n"
    "            legalWallMoves(s, side, wallMoves, nullptr, nullptr, nullptr, nullptr, &cache0, &cache1, &xdistCache);\n"
    "            stats.wallGenNodes++;\n"
    "            stats.wallMovesGenerated += wallMoves.size();\n",
    "wall generation")

sp.write_text(s)

up = Path("tools/external/zquoridor_uci.cpp")
u = up.read_text()
old = '                      << " lmrDiv=" << engine.getLmrDivisor()\n'
new = (old +
       '                      << " qNodes=" << stats.qNodes\n'
       '                      << " ttProbes=" << stats.ttProbes\n'
       '                      << " ttHits=" << stats.ttHits\n'
       '                      << " ttCuts=" << stats.ttCutoffs\n'
       '                      << " betaCuts=" << stats.betaCutoffs\n'
       '                      << " firstCuts=" << stats.firstMoveBetaCutoffs\n'
       '                      << " pawnSearched=" << stats.pawnMovesSearched\n'
       '                      << " wallSearched=" << stats.wallMovesSearched\n'
       '                      << " wallGenNodes=" << stats.wallGenNodes\n'
       '                      << " wallGenerated=" << stats.wallMovesGenerated\n'
       '                      << " lmrMoves=" << stats.lmrReducedMoves\n'
       '                      << " lmrResearch=" << stats.lmrResearches\n'
       '                      << " pvsResearch=" << stats.pvsResearches\n')
if u.count(old) != 1:
    raise SystemExit(f"adapter marker expected 1 match, got {u.count(old)}")
up.write_text(u.replace(old, new))
print("patched search.hpp and zquoridor_uci.cpp with diagnostic counters")
