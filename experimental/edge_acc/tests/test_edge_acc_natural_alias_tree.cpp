#include "../src/search.hpp"
#include "../src/mcab.hpp"
#include <cstdio>
#include <cstdlib>

using namespace qr;
using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                               RepetitionTable, SearchStats>;

#ifndef EDGE_NATURAL_VARIANT
#define EDGE_NATURAL_VARIANT "unspecified"
#endif

static uint64_t recomputeHash(const State& state) {
    Zobrist& z = zobrist();
    uint64_t hash = z.pawnKey[0][state.pawn[0]] ^ z.pawnKey[1][state.pawn[1]];
    for (int slot = 0; slot < WS * WS; ++slot) {
        const uint64_t mask = 1ull << slot;
        if (state.wallsH & mask) hash ^= z.wallHKey[slot];
        if (state.wallsV & mask) hash ^= z.wallVKey[slot];
    }
    if (state.turn != 0) hash ^= z.turnKey;
    return hash;
}

static State naturalPosition() {
    State state{};
    state.pawn[0] = (uint8_t)cellIdx(6, 7);
    state.pawn[1] = (uint8_t)cellIdx(3, 4);
    state.wallsH = 0;
    state.wallsV = 0;
    state.wallsLeft[0] = 10;
    state.wallsLeft[1] = 10;
    state.turn = 0;
    state.hash = recomputeHash(state);
    return state;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <weights_int8.bin> <node_budget>\n", argv[0]);
        return 1;
    }
    if (!loadWeightsQuant(argv[1])) {
        std::fprintf(stderr, "failed to load weights: %s\n", argv[1]);
        return 2;
    }
    const int nodes = std::atoi(argv[2]);
    if (nodes <= 0) return 3;

    const State root = naturalPosition();
    const MoveList moves = legalMoves(root);
    bool hasH63 = false;
    bool hasV63 = false;
    for (const Move& move : moves) {
        hasH63 |= move.isWall && move.a == 0 && move.b == 7 && move.c == 7;
        hasV63 |= move.isWall && move.a == 1 && move.b == 7 && move.c == 7;
    }
    if (!hasH63 || !hasV63) {
        std::fprintf(stderr, "root lacks natural H63/V63 sibling moves: H=%d V=%d\n",
                     hasH63 ? 1 : 0, hasV63 ? 1 : 0);
        return 4;
    }

    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    Mcab search;
    search.params.nodeBudget = nodes;
    search.params.autoNodeBudget = false;
    search.params.leafDepth = 0;
    search.params.adaptiveLeafDepth = false;
    search.params.treeReuse = true;
    search.params.rootNoiseEnabled = false;
    search.params.adaptiveTime = false;
    RepetitionTable history;
    SearchStats stats;
    mcab::McabStats mcabStats;
    Move best = search.chooseMoveMCAB(engine, root, 40, 0, stats,
                                      history, &mcabStats);
    const auto* treeRoot = search.rootNodeForInspection();
    long long h63Visits = 0;
    long long v63Visits = 0;
    if (treeRoot) {
        for (size_t i = 0; i < treeRoot->moves.size() && i < treeRoot->N.size(); ++i) {
            const Move& move = treeRoot->moves[i];
            if (!move.isWall || move.b != 7 || move.c != 7) continue;
            if (move.a == 0) h63Visits = (long long)treeRoot->N[i];
            if (move.a == 1) v63Visits = (long long)treeRoot->N[i];
        }
    }
    std::printf(
        "EDGE_NATURAL_TREE variant=%s nodes=%d best=%u pool=%zu "
        "simulations=%lld expanded=%lld rootN=%d fp=%016llx "
        "h63=%d v63=%d h63N=%lld v63N=%lld reused=%d\n",
        EDGE_NATURAL_VARIANT, nodes, (unsigned)moveToPolicyIndex(best),
        search.poolSize(), (long long)mcabStats.simulations,
        (long long)mcabStats.nodesExpanded,
        treeRoot ? treeRoot->totalN : -1,
        (unsigned long long)search.treeFingerprintForInspection(),
        hasH63 ? 1 : 0, hasV63 ? 1 : 0, h63Visits, v63Visits,
        mcabStats.treeReused ? 1 : 0);
    return 0;
}
