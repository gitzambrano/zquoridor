// test_mcab_endgame_leaf.cpp -- pins the endgame leaf rule of src/mcab.hpp
// (McabParams::endgameMoverWallThreshold, see inv/endgame-wander).
//
// The rule exists because the WL head is almost blind to the pawn race after
// a side spends its walls, so a tree of static leaves has no gradient toward
// the goal and the engine shuffles its pawn. The rule gives the leaves a
// real alpha-beta search in that regime only.
//
// The test checks three things:
//   1. Production stays bit-identical: the rule is off by default.
//   2. The gate reads the wall stock of the side to move at the root, and it
//      stays off above the threshold.
//   3. With the rule on, the engine walks the won race of
//      benchmarks/repro_wander.cpp to the goal instead of shuffling.
//
// Build (correctness profile, no -march=native):
//   g++ -O2 -std=c++17 -Isrc -o bin/test_mcab_endgame_leaf \
//       tests/test_mcab_endgame_leaf.cpp
// Run from the repo ROOT (it loads data/nnue/nnue_weights_int8.bin).
#include <cstdio>
#include <cassert>
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;
using McabRunnerT = mcab::McabRunner<Negamax, State, Move, MoveList, AccPair,
                                      RepetitionTable, SearchStats>;

namespace {

void addWallH(State& s, int r, int c) {
    s.wallsH |= 1ull << slotIdx(r, c);
    s.hash ^= zobrist().wallHKey[slotIdx(r, c)];
}

// The position of benchmarks/repro_wander.cpp at move 22. Player 0 is the
// engine: 0 walls left, 5 steps from the goal. Player 1 holds 8 walls and
// needs 13 steps. Player 0 moves first.
State reproPosition() {
    State s;
    s.pawn[0] = cellIdx(4, 4);
    s.pawn[1] = cellIdx(2, 7);
    s.wallsH = 0;
    s.wallsV = 0;
    s.wallsLeft[0] = 0;
    s.wallsLeft[1] = 8;
    s.turn = 0;
    Zobrist& z = zobrist();
    s.hash = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    addWallH(s, 0, 0); addWallH(s, 0, 2);
    addWallH(s, 1, 1); addWallH(s, 1, 3); addWallH(s, 1, 5); addWallH(s, 1, 7);
    addWallH(s, 2, 0); addWallH(s, 2, 2); addWallH(s, 2, 4); addWallH(s, 2, 6);
    addWallH(s, 5, 4); addWallH(s, 5, 6);
    return s;
}

int distance(const State& s, int player) {
    return shortestPathLen(s.wallsH, s.wallsV, s.pawn[player], player);
}

// The opponent always takes the legal pawn move that most shortens its own
// path. It never places a wall, so the playout stays deterministic.
Move greedyOpponent(const State& s) {
    MoveList moves = legalMoves(s);
    int best = -1, bestDist = 1 << 30;
    for (size_t i = 0; i < moves.size(); i++) {
        if (moves[i].isWall) continue;
        State next = applyMove(s, moves[i]);
        int d = shortestPathLen(next.wallsH, next.wallsV, next.pawn[s.turn], s.turn);
        if (d < bestDist) { bestDist = d; best = (int)i; }
    }
    assert(best >= 0 && "the opponent always has at least one legal pawn move here");
    return moves[(size_t)best];
}

// Plays `moveCap` engine moves from the repro position and returns how many
// steps the engine cut off its own shortest path.
int playProgress(const mcab::McabParams& params, int timeMs, int moveCap) {
    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    McabRunnerT runner;
    runner.setParams(params);
    runner.resetTree();

    State s = reproPosition();
    RepetitionTable history;
    history.push(s.hash);
    int startDist = distance(s, 0);

    for (int i = 0; i < moveCap; i++) {
        SearchStats stats;
        Move m = runner.choose(engine, s, 40, timeMs, stats, history);
        s = applyMove(s, m);
        history.push(s.hash);
        if (winner(s) != -1) return startDist;
        s = applyMove(s, greedyOpponent(s));
        history.push(s.hash);
        if (winner(s) != -1) break;
    }
    return startDist - distance(s, 0);
}

} // namespace

// ---------------------------------------------------------------------
// 1) The rule is off in production, so production stays bit-identical.
// ---------------------------------------------------------------------
void testDefaultIsOff() {
    mcab::McabParams prod;
    assert(prod.endgameMoverWallThreshold < 0 &&
           "the endgame leaf rule must stay OFF by default -- turning it on "
           "changes every production search and needs an arena match first");
    printf("[testDefaultIsOff] endgameMoverWallThreshold=%d OK\n", prod.endgameMoverWallThreshold);
}

// ---------------------------------------------------------------------
// 2) The gate reads the wall stock of the SIDE TO MOVE at the root, not the
//    combined stock. Player 0 moves first here and holds 0 walls, while
//    player 1 holds 8. Therefore a threshold of -1 must not fire and a
//    threshold of 0 must fire, even though 8 walls are still on the board.
// ---------------------------------------------------------------------
void testGateReadsRootWallStock() {
    State root = reproPosition();
    assert(root.turn == 0 && root.wallsLeft[0] == 0 && root.wallsLeft[1] == 8 &&
           "the repro position must have player 0 to move with 0 walls against 8");

    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    RepetitionTable history;

    auto leafSearches = [&](int threshold) {
        McabRunnerT runner;
        mcab::McabParams p;
        p.nodeBudget = 400;
        p.endgameMoverWallThreshold = threshold;
        p.endgameLeafDepth = 2;
        runner.setParams(p);
        runner.resetTree();
        SearchStats stats;
        mcab::McabStats mstats{};
        runner.choose(engine, root, 40, 0, stats, history, &mstats);
        return mstats.leafDepthSum;
    };

    long long off = leafSearches(-1);   // negative: rule stays off
    long long on = leafSearches(0);     // mover holds 0 walls: rule fires
    printf("[testGateReadsRootWallStock] leafDepthSum: threshold=-1 -> %lld, threshold=0 -> %lld\n",
           off, on);
    assert(off == 0 && "a negative threshold must leave the leaves static");
    assert(on > 0 && "a mover out of walls must give the leaves a real search");
    printf("[testGateReadsRootWallStock] OK\n");
}

// ---------------------------------------------------------------------
// 3) With the rule on, the engine walks the won race to the goal. Production
//    settings hold their distance at 5 for six moves in this position.
// ---------------------------------------------------------------------
void testEndgameLeafStopsWandering() {
    mcab::McabParams p;
    p.endgameMoverWallThreshold = 0;
    p.endgameLeafDepth = 2;
    int progress = playProgress(p, /*timeMs=*/300, /*moveCap=*/6);
    printf("[testEndgameLeafStopsWandering] progress=%+d (perfect play = +5)\n", progress);
    assert(progress >= 4 &&
           "with the endgame leaf rule on, the engine must walk the won race "
           "to the goal instead of shuffling its pawn");
    printf("[testEndgameLeafStopsWandering] OK\n");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        printf("FATAL: could not load data/nnue/nnue_weights_int8.bin. "
               "Run this test from the repo ROOT.\n");
        return 1;
    }
    testDefaultIsOff();
    testGateReadsRootWallStock();
    testEndgameLeafStopsWandering();
    printf("\nTODOS OS TESTES DE test_mcab_endgame_leaf PASSARAM\n");
    return 0;
}
