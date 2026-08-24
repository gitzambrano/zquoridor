// repro_wander -- reproduces the "wandering" report: in a won near-endgame
// position the engine shuffles its pawn sideways instead of walking to the
// goal, and it keeps doing so while the opponent closes the race.
//
// The position comes from a real game in the web GUI, at move 22. Player 0
// is the engine. It holds 0 walls and needs 5 steps. Player 1 is the human
// opponent. That player holds 8 walls and needs 13 steps. Twelve horizontal
// walls are on the board and no vertical wall is. All four distances match
// the DIST counters the GUI showed, so the position is exact.
//
// The replay drives the engine through the recorded human answers
// (g7 f7 e7 d7 c7, which is column 6 down to column 2 on row 2 in engine
// coordinates). Tree reuse stays live across the whole segment, exactly as
// the web GUI keeps it. The playout runs the same position against a greedy
// opponent instead.
//
// WHAT IT SHOWS. The hybrid at production settings answers f6, g6, f6, g6
// and holds its own distance at 5 for six moves. Pure alpha-beta wanders
// too, but less. See status.md, section "MCTS endgame wandering", for the
// measured cause.
//
// Build (performance profile):
//   g++ -O3 -std=c++17 -march=native [-mavx2 -mfma] -Isrc \
//       -o bin/repro_wander benchmarks/repro_wander.cpp
// Run from the repo ROOT:
//   bin/repro_wander [timeMs] [plies]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;
using McabRunnerT = mcab::McabRunner<Negamax, State, Move, MoveList, AccPair,
                                      RepetitionTable, SearchStats>;

static void addH(State& s, int r, int c) {
    s.wallsH |= 1ull << slotIdx(r, c);
    s.hash ^= zobrist().wallHKey[slotIdx(r, c)];
}

static State reproPosition(int p0r, int p0c, int turn) {
    State s;
    s.pawn[0] = cellIdx(p0r, p0c);
    s.pawn[1] = cellIdx(2, 2);
    s.wallsH = 0; s.wallsV = 0;
    s.wallsLeft[0] = 0;
    s.wallsLeft[1] = 8;
    s.turn = turn;
    Zobrist& z = zobrist();
    s.hash = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    if (turn == 1) s.hash ^= z.turnKey;
    addH(s, 0, 0); addH(s, 0, 2);
    addH(s, 1, 1); addH(s, 1, 3); addH(s, 1, 5); addH(s, 1, 7);
    addH(s, 2, 0); addH(s, 2, 2); addH(s, 2, 4); addH(s, 2, 6);
    addH(s, 5, 4); addH(s, 5, 6);
    return s;
}

static std::string cellName(int cell) {
    int r = cell / N, c = cell % N;
    char buf[8];
    snprintf(buf, sizeof buf, "%c%d", (char)('a' + c), r + 1);
    return buf;
}
static std::string moveName(const Move& m) {
    if (!m.isWall) return cellName(m.a);
    char buf[16];
    snprintf(buf, sizeof buf, "%c%d%d%s", m.a ? 'V' : 'H', m.b, m.c, "");
    return buf;
}
static int dist(const State& s, int p) {
    return shortestPathLen(s.wallsH, s.wallsV, s.pawn[p], p);
}

// Opponent policy: always take a pawn step that shortens its own path.
static Move greedyOpponent(const State& s) {
    MoveList ms = legalMoves(s);
    int best = -1, bestD = 1 << 30;
    for (size_t i = 0; i < ms.size(); i++) {
        if (ms[i].isWall) continue;
        State ns = applyMove(s, ms[i]);
        int d = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[s.turn], s.turn);
        if (d < bestD) { bestD = d; best = (int)i; }
    }
    return ms[best];
}

static void playout(const char* label, bool mcabOn, int timeMs, int plies) {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    McabRunnerT runner;
    mcab::McabParams p = runner.params();
    p.enabled = mcabOn;
    runner.setParams(p);
    runner.resetTree();

    State s = reproPosition(4, 4, /*turn=*/0);
    RepetitionTable hist;
    hist.push(s.hash);

    printf("\n=== %s (time=%dms) ===\n", label, timeMs);
    printf("start: p0=%s d0=%d | p1=%s d1=%d\n",
           cellName(s.pawn[0]).c_str(), dist(s, 0),
           cellName(s.pawn[1]).c_str(), dist(s, 1));

    for (int ply = 0; ply < plies; ply++) {
        if (winner(s) != -1) { printf("winner: p%d\n", winner(s)); break; }
        Move m;
        if (s.turn == 0) {
            SearchStats st;
            mcab::McabStats ms{};
            m = runner.choose(eng, s, 40, timeMs, st, hist, &ms);
            State ns = applyMove(s, m);
            printf("ply %2d  ENGINE %-5s  d0=%d->%d  d1=%d  sims=%lld\n",
                   ply, moveName(m).c_str(), dist(s, 0), dist(ns, 0), dist(s, 1),
                   ms.simulations);
        } else {
            m = greedyOpponent(s);
            State ns = applyMove(s, m);
            printf("ply %2d  opp    %-5s  d1=%d->%d\n",
                   ply, moveName(m).c_str(), dist(s, 1), dist(ns, 1));
        }
        s = applyMove(s, m);
        hist.push(s.hash);
        if (hist.count(s.hash) >= 3) { printf("*** 3-fold repetition draw ***\n"); break; }
    }
    printf("end: p0=%s d0=%d | p1=%s d1=%d\n",
           cellName(s.pawn[0]).c_str(), dist(s, 0),
           cellName(s.pawn[1]).c_str(), dist(s, 1));
}

// Replays the exact segment from the user's screenshots: the engine is
// player 0, the opponent plays the human's recorded moves (g7 f7 e7 d7 c7 =
// (2,6) (2,5) (2,4) (2,3) (2,2) in engine coordinates). Tree reuse stays
// live across the whole segment, exactly like the web GUI.
static void replay(const char* label, bool mcabOn, int timeMs, bool treeReuse = true, bool clearTT = false) {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    McabRunnerT runner;
    mcab::McabParams p = runner.params();
    p.enabled = mcabOn;
    p.treeReuse = treeReuse;
    p.clearTTPerMove = clearTT;
    runner.setParams(p);
    runner.resetTree();

    // after 21...h7: engine at e5, human at h7
    State s = reproPosition(4, 4, /*turn=*/0);
    s.hash ^= zobrist().pawnKey[1][s.pawn[1]];
    s.pawn[1] = cellIdx(2, 7);
    s.hash ^= zobrist().pawnKey[1][s.pawn[1]];

    const int humanCols[] = {6, 5, 4, 3, 2, 1, 0};
    RepetitionTable hist;
    hist.push(s.hash);

    printf("\n=== REPLAY %s (time=%dms) ===\n", label, timeMs);
    int hi = 0;
    for (int mv = 22; mv <= 28; mv++) {
        if (winner(s) != -1) { printf("winner: p%d\n", winner(s)); return; }
        SearchStats st;
        mcab::McabStats ms{};
        Move m = runner.choose(eng, s, 40, timeMs, st, hist, &ms);
        State ns = applyMove(s, m);
        printf("%2d. ENGINE %-5s  d0=%d->%d   d1=%d   reused=%d nodes=%d sims=%lld\n",
               mv, moveName(m).c_str(), dist(s, 0), dist(ns, 0), dist(s, 1),
               (int)ms.treeReused, ms.reusedNodes, ms.simulations);
        s = ns;
        hist.push(s.hash);
        if (winner(s) != -1) { printf("winner: p%d\n", winner(s)); return; }
        if (hi >= (int)(sizeof humanCols / sizeof humanCols[0])) break;
        MoveList lm = legalMoves(s);
        Move hm{}; bool found = false;
        for (size_t i = 0; i < lm.size(); i++)
            if (!lm[i].isWall && lm[i].a == cellIdx(2, humanCols[hi])) { hm = lm[i]; found = true; }
        if (!found) { printf("    (recorded human move illegal here -- stopping replay)\n"); return; }
        hi++;
        s = applyMove(s, hm);
        hist.push(s.hash);
        printf("    human  %-5s  d1=%d\n", moveName(hm).c_str(), dist(s, 1));
    }
}

int main(int argc, char** argv) {
    int timeMs = argc > 1 ? atoi(argv[1]) : 300;
    int plies  = argc > 2 ? atoi(argv[2]) : 24;
    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        printf("FATAL: could not load data/nnue/nnue_weights_int8.bin\n");
        return 1;
    }
    replay("HYBRID, treeReuse=ON  (production)", true, timeMs, true, false);
    replay("HYBRID, treeReuse=OFF", true, timeMs, false, false);
    replay("HYBRID, treeReuse=ON, clearTTPerMove=ON", true, timeMs, true, true);
    replay("PURE ALPHA-BETA", false, timeMs);
    playout("HYBRID MCTS (production)", true, timeMs, plies);
    playout("PURE ALPHA-BETA", false, timeMs, plies);
    return 0;
}
