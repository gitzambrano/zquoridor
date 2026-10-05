#include "../src/search.hpp"
#include "../src/mcab.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace qr;
using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                               RepetitionTable, SearchStats>;

static State makePosition(uint64_t seed, int plies) {
    std::mt19937_64 rng(seed);
    State s = initialState();
    for (int ply = 0; ply < plies && winner(s) == -1; ++ply) {
        MoveList moves = legalMoves(s);
        if (moves.empty()) break;
        std::vector<size_t> pawn, wall;
        for (size_t i = 0; i < moves.size(); ++i) {
            (moves[i].isWall ? wall : pawn).push_back(i);
        }

        size_t pick = 0;
        // Keep many walls on the board without exhausting stock: roughly one wall
        // every three plies, otherwise a pawn move.
        if ((ply % 3) == 0 && !wall.empty()) {
            pick = wall[(size_t)(rng() % wall.size())];
        } else if (!pawn.empty()) {
            pick = pawn[(size_t)(rng() % pawn.size())];
        } else {
            pick = (size_t)(rng() % moves.size());
        }
        s = applyMove(s, moves[pick]);
    }
    return s;
}

static void configure(Mcab& m, int nodes) {
    m.params.nodeBudget = nodes;
    m.params.autoNodeBudget = false;
    m.params.leafDepth = 0;
    m.params.adaptiveLeafDepth = false;
    m.params.treeReuse = true;
    m.params.rootNoiseEnabled = false;
    m.params.adaptiveTime = false;
}

static void runIndependent(const char* label, const State& s, int nodes) {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    Mcab m;
    configure(m, nodes);
    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats ms;
    Move best = m.chooseMoveMCAB(eng, s, 40, 0, stats, hist, &ms);
    const auto* root = m.rootNodeForInspection();
    std::printf(
        "CASE %s nodes=%d best=%u pool=%zu sims=%lld expanded=%lld "
        "hits=%lld misses=%lld fp=%016llx rootN=%d\n",
        label, nodes, (unsigned)moveToPolicyIndex(best), m.poolSize(),
        (long long)ms.simulations, (long long)ms.nodesExpanded,
        (long long)ms.edgeAccCacheHits, (long long)ms.edgeAccCacheMisses,
        (unsigned long long)m.treeFingerprintForInspection(),
        root ? root->totalN : -1);
}

static void runReuse(const State& start, int nodes, int steps) {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    Mcab m;
    configure(m, nodes);
    RepetitionTable hist;
    State s = start;

    for (int step = 0; step < steps && winner(s) == -1; ++step) {
        SearchStats stats;
        mcab::McabStats ms;
        Move best = m.chooseMoveMCAB(eng, s, 40, 0, stats, hist, &ms);
        const auto* root = m.rootNodeForInspection();
        std::printf(
            "REUSE step=%d nodes=%d best=%u pool=%zu sims=%lld expanded=%lld "
            "treeHit=%d reused=%lld hits=%lld misses=%lld fp=%016llx rootN=%d\n",
            step, nodes, (unsigned)moveToPolicyIndex(best), m.poolSize(),
            (long long)ms.simulations, (long long)ms.nodesExpanded,
            ms.treeReused ? 1 : 0, (long long)ms.reusedNodes,
            (long long)ms.edgeAccCacheHits, (long long)ms.edgeAccCacheMisses,
            (unsigned long long)m.treeFingerprintForInspection(),
            root ? root->totalN : -1);

        s = applyMove(s, best);
    }
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

    // Independent large trees from several phases of play.
    State p0 = initialState();
    State p1 = makePosition(0x1001, 8);
    State p2 = makePosition(0x2002, 14);
    State p3 = makePosition(0x3003, 20);

    runIndependent("initial-100k", p0, 100000);
    runIndependent("mid8-100k", p1, 100000);
    runIndependent("mid14-100k", p2, 100000);
    runIndependent("mid20-100k", p3, 100000);

    // One substantially larger tree.
    runIndependent("mid14-300k", p2, 300000);

    // Re-rooting stress: reuse the same MCAB tree across successive chosen moves.
    runReuse(makePosition(0x515151, 12), 100000, 5);
    return 0;
}
