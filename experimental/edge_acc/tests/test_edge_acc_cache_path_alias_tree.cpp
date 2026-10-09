#include "../src/search.hpp"
#include "../src/mcab.hpp"
#include <cstdio>
#include <cstdlib>

using namespace qr;
using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                               RepetitionTable, SearchStats>;

#ifndef EDGE_ALIAS_VARIANT
#define EDGE_ALIAS_VARIANT "unspecified"
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

static State aliasPosition() {
    State state{};
    state.pawn[0] = (uint8_t)cellIdx(6, 7);
    state.pawn[1] = (uint8_t)cellIdx(3, 4);
    state.wallsH = 0;
    state.wallsV = 1ull << 63;
    state.wallsLeft[0] = 9;
    state.wallsLeft[1] = 10;
    state.turn = 0;
    state.hash = recomputeHash(state);
    return state;
}

static void configure(Mcab& search, int nodes) {
    search.params.nodeBudget = nodes;
    search.params.autoNodeBudget = false;
    search.params.leafDepth = 0;
    search.params.adaptiveLeafDepth = false;
    search.params.treeReuse = true;
    search.params.clearTTPerMove = true;
    search.params.rootNoiseEnabled = false;
    search.params.adaptiveTime = false;
}

static void printResult(const char* phase, int nodes, const Mcab& search,
                        const mcab::McabStats& stats, Move best) {
    const auto* root = search.rootNodeForInspection();
    std::printf(
        "EDGE_ALIAS_TREE variant=%s phase=%s nodes=%d best=%u pool=%zu "
        "simulations=%lld expanded=%lld rootN=%d fp=%016llx reused=%d\n",
        EDGE_ALIAS_VARIANT, phase, nodes, (unsigned)moveToPolicyIndex(best),
        search.poolSize(), (long long)stats.simulations,
        (long long)stats.nodesExpanded, root ? root->totalN : -1,
        (unsigned long long)search.treeFingerprintForInspection(),
        stats.treeReused ? 1 : 0);
}

static int runCase(Negamax& engine, const State& root, int nodes) {
    Mcab search;
    configure(search, nodes);
    RepetitionTable history;
    SearchStats warmStats;
    mcab::McabStats warmMcabStats;
    Move warmBest = search.chooseMoveMCAB(engine, root, 40, 0, warmStats,
                                           history, &warmMcabStats);
    printResult("warm", nodes, search, warmMcabStats, warmBest);

    // These entries describe legal H-wall geometry. The legacy path-cache key
    // aliases each H-wall query with its V-wall transpose, so the second search
    // receives valid data for the wrong wall orientation.
    PlayerPathCacheTable* paths = engine.pathCache();
    if (!paths) return 10;
    paths->clear();
    constexpr uint64_t aliasH = 1ull << 63;
    for (int cell = 0; cell < N * N; ++cell) {
        for (int player = 0; player < 2; ++player) {
            PlayerPathCache data;
            computeDistFull(aliasH, 0, cell, player, data);
            paths->put(aliasH, 0, cell, player, data);
        }
    }

    // Verify that the preloaded history yields a different valid BFS result
    // through the aliased V-wall query at the adversarial root.
    PlayerPathCache aliased, expected;
    computeDistFull(0, aliasH, root.pawn[0], 0, expected);
    if (!paths->get(0, aliasH, root.pawn[0], 0, aliased) ||
        aliased.distToGoal == expected.distToGoal) {
        std::fprintf(stderr, "adversarial BFS alias did not produce a distinct distance\n");
        return 10;
    }

    // Clear only the tree. Keep EdgeAcc entries from the warm search. The
    // second call then builds and retains the contaminated tree for inspection.
    search.resetTree();
    SearchStats aliasStats;
    mcab::McabStats aliasMcabStats;
    Move aliasBest = search.chooseMoveMCAB(engine, root, 40, 0, aliasStats,
                                            history, &aliasMcabStats);
    if (aliasMcabStats.treeReused) {
        std::fprintf(stderr, "unexpected tree reuse at node budget %d\n", nodes);
        return 11;
    }

    printResult("alias", nodes, search, aliasMcabStats, aliasBest);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights_int8.bin>\n", argv[0]);
        return 1;
    }
    if (!loadWeightsQuant(argv[1])) {
        std::fprintf(stderr, "failed to load weights: %s\n", argv[1]);
        return 2;
    }

    const State root = aliasPosition();
    const MoveList moves = legalMoves(root);
    if (moves.empty() || winner(root) != -1) {
        std::fprintf(stderr, "adversarial root is not searchable\n");
        return 3;
    }

    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    for (int nodes : {1000, 4000}) {
        const int result = runCase(engine, root, nodes);
        if (result != 0) return result;
    }
    return 0;
}
