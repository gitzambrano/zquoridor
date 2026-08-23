// map_compute.cpp -- investigation benchmark: where does compute go today?
//
// Three blocks, all on one deterministic position corpus (opening/midgame/
// late/race mix, generated from fixed seeds):
//
// A) Production MCAB (McabParams defaults, leafDepth=0, treeReuse off --
//    fresh tree per position so counts are comparable across positions) at
//    40ms and 200ms per move. Reports McabStats (simulations, expanded
//    nodes, leaf searches) plus wall time. Time attribution is derived
//    from unit costs measured in block C: policy passes == nodesExpanded,
//    leaf evals == leafSearches.
//
// B) Pure alpha-beta chooseMove at 40ms and 200ms per move: nodes,
//    reached depth, nps. Reported twice: policy ordering gated at
//    minDepth=3 (today's production AB) and ordering without any policy
//    pass (minDepth=99).
//
// C) Unit costs on this machine: forwardPolicyQuant, nnueEvalInt,
//    legalMoves, makeChildAccPair, buildAccPairRoot -- ns/call, so the
//    block A counts convert into approximate % of wall time.
//
// Run from the repo root (loads data/nnue/nnue_weights_int8.bin).
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;
using mcab::MCABSearch;
using mcab::McabStats;
using Mcab = MCABSearch<Negamax, State, Move, MoveList, AccPair,
                        RepetitionTable, SearchStats>;

namespace {

std::vector<State> stratifiedPositions() {
    std::mt19937 rng(20260823);
    std::vector<State> out;
    auto playRandom = [&](int plies) {
        State s = initialState();
        for (int p = 0; p < plies; p++) {
            if (winner(s) != -1) return State();
            auto moves = legalMoves(s);
            if (moves.empty()) return State();
            std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
            s = applyMove(s, moves[d(rng)]);
        }
        return s;
    };
    // 4 opening (4..8 plies), 6 midgame (14..24), 4 late (30..40),
    // 2 near-race (many plies). Reject finished games.
    const int wants[] = {4, 6, 4, 2};
    const int lo[]    = {4, 14, 30, 44};
    const int hi[]    = {8, 24, 40, 58};
    for (int k = 0; k < 4; k++) {
        int got = 0, tries = 0;
        while (got < wants[k] && tries < 4000) {
            tries++;
            int plies = lo[k] + (int)(rng() % (unsigned)(hi[k] - lo[k] + 1));
            State s = playRandom(plies);
            if (winner(s) != -1) continue;
            out.push_back(s);
            got++;
        }
    }
    return out;
}

double secondsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <typename F>
double benchUnit(int reps, F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) f(i);
    return secondsSince(t0) / reps * 1e9;  // ns/call
}

}  // namespace

int main(int argc, char** argv) {
    const char* weightsPath = (argc > 1) ? argv[1] : "data/nnue/nnue_weights_int8.bin";
    if (!loadWeightsQuant(weightsPath)) {
        fprintf(stderr, "[map] failed to load '%s'\n", weightsPath);
        return 1;
    }
    std::vector<State> pos = stratifiedPositions();
    printf("[map] corpus=%zu positions\n", pos.size());

    // ---------------- C) unit costs first (feed attribution) -------------
    Negamax ueng;
    ueng.setEvalMode(Negamax::EvalMode::NNUE);
    const State& mid = pos[pos.size() / 2];
    AccPair rootAcc = buildAccPairRoot(mid, ueng.pathCache());
    std::array<float, POLICY_OUT> pol{};
    volatile float polSink = 0.f;
    double nsPol = benchUnit(20000, [&](int) { forwardPolicyQuant(rootAcc.acc[mid.turn], pol); polSink += pol[7]; });
    double nsEval = benchUnit(200000, [&](int) { volatile int v = nnueEvalInt(rootAcc, mid.turn); (void)v; });
    MoveList ml;
    double nsLegal = benchUnit(50000, [&](int) { ml = legalMoves(mid); });
    Move m0 = ml.size() ? ml[ml.size() / 2] : Move::pawn(0);
    AccPair childAcc;
    double nsChildAcc = benchUnit(100000, [&](int) { makeChildAccPair(rootAcc, childAcc, mid, m0, ueng.pathCache()); });
    double nsBuildRoot = benchUnit(20000, [&](int) { rootAcc = buildAccPairRoot(mid, ueng.pathCache()); });
    SearchStats ust;
    RepetitionTable urep;
    double nsLeafD5 = benchUnit(3000, [&](int) { ueng.searchLeaf(mid, 5, ust, urep, &rootAcc); });
    printf("[map-unit] policyPass_ns=%.0f nnueEval_ns=%.0f legalMoves_ns=%.0f makeChildAcc_ns=%.0f buildRootAcc_ns=%.0f searchLeafD5_ns=%.0f\n",
           nsPol, nsEval, nsLegal, nsChildAcc, nsBuildRoot, nsLeafD5);

    // ---------------- A) MCAB production shape ---------------------------
    for (int timeMs : {40, 200}) {
        long long totSims = 0, totExp = 0, totLeaves = 0, totTrunc = 0;
        double totalTime = 0.0;
        int searched = 0;
        for (size_t pi = 0; pi < pos.size(); pi++) {
            const State& s = pos[pi];
            if (s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0) continue;
            Negamax eng;
            eng.setEvalMode(Negamax::EvalMode::NNUE);
            Mcab mcab;
            mcab.params = mcab::McabParams{};      // production values
            mcab.params.treeReuse = false;          // fresh tree per position
            mcab.params.rootNoiseEnabled = false;   // arena-style, no noise
            SearchStats st;
            RepetitionTable hist;
            McabStats ms;
            auto t0 = std::chrono::steady_clock::now();
            mcab.chooseMoveMCAB(eng, s, 40, timeMs, st, hist, &ms);
            double dtMs = secondsSince(t0) * 1000.0;
            totalTime += secondsSince(t0);
            totSims += ms.simulations;
            totExp += ms.nodesExpanded;
            totLeaves += ms.leafSearches;
            totTrunc += ms.leafTruncated;
            searched++;
            printf("[map-mcab-%dms-pos%02zu] walls=%d/%d sims=%lld exp=%lld ms=%.1f\n",
                   timeMs, (unsigned long)pi, (int)s.wallsLeft[0], (int)s.wallsLeft[1],
                   ms.simulations, ms.nodesExpanded, dtMs);
        }
        int nMoves = searched;
        printf("[map-mcab-%dms] searched_positions=%d sims_per_move=%.0f expanded_per_move=%.0f leaves_per_move=%.0f truncated_total=%lld wall_s_per_move_ms=%.1f\n",
               timeMs, nMoves, (double)totSims / nMoves, (double)totExp / nMoves,
               (double)totLeaves / nMoves, totTrunc, totalTime * 1000.0 / nMoves);
        double policySec = totExp * nsPol * 1e-9;
        double evalSec = totLeaves * nsEval * 1e-9;
        printf("[map-mcab-%dms] approx_policy_s=%.3f approx_leaf_eval_s=%.3f total_s=%.2f\n",
               timeMs, policySec, evalSec, totalTime);
    }

    // ---------------- B) pure alpha-beta shapes --------------------------
    for (int timeMs : {40, 200}) {
        for (int minDepth : {3, 99, 0}) {
            uint64_t totNodes = 0;
            int depthSum = 0;
            double totalTime = 0.0;
            for (auto& s : pos) {
                Negamax eng;
                eng.setEvalMode(Negamax::EvalMode::NNUE);
                eng.setPolicyOrderingEnabled(true);
                eng.setPolicyOrderingMinDepth(minDepth);
                SearchStats st;
                auto t0 = std::chrono::steady_clock::now();
                eng.chooseMove(s, 40, timeMs, st);
                totalTime += secondsSince(t0);
                totNodes += st.nodes;
                depthSum += st.reachedDepth;
            }
            int nPos = (int)pos.size();
            printf("[map-ab-%dms-mindepth%d] positions=%d nodes_per_move=%.0f avg_depth=%.2f nps=%.0f\n",
                   timeMs, minDepth, nPos, (double)totNodes / nPos,
                   (double)depthSum / nPos, totNodes / totalTime);
        }
    }
    return 0;
}
