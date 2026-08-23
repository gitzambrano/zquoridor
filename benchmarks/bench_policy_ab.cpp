// bench_policy_ab.cpp -- inv/ab-policy perf instrument (2026-08-23).
//
// Measures, on one deterministic stratified corpus (opening/midgame/late/
// race mix), for each direction-B/C/D variant and the all-on stress:
//   1. NODES TO FIXED DEPTH: searchShallow at depth 8 summed over the
//      corpus, plus wall time and nps. This is the repo's canonical
//      "nodes-to-equal-depth" metric.
//   2. TIME TO EQUAL BUDGET: chooseMove with a 200ms budget per position,
//      reporting the average reached depth.
//
// All variants run in NNUE mode with policy ordering ON at minDepth 3
// (production). Run from the repo root (loads data/nnue/nnue_weights_int8.bin).
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

namespace {

std::vector<State> corpus() {
    std::mt19937 rng(20260823);
    std::vector<State> out;
    const int wants[] = {3, 5, 3, 2};
    const int lo[]    = {4, 14, 30, 44};
    const int hi[]    = {8, 24, 40, 58};
    for (int k = 0; k < 4; k++) {
        int got = 0, tries = 0;
        while (got < wants[k] && tries < 4000) {
            tries++;
            int plies = lo[k] + (int)(rng() % (unsigned)(hi[k] - lo[k] + 1));
            State s = initialState();
            bool dead = false;
            for (int p = 0; p < plies; p++) {
                if (winner(s) != -1) { dead = true; break; }
                auto moves = legalMoves(s);
                if (moves.empty()) { dead = true; break; }
                std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
                s = applyMove(s, moves[d(rng)]);
            }
            if (dead || winner(s) != -1) continue;
            out.push_back(s);
            got++;
        }
    }
    return out;
}

double secondsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void configure(Negamax& eng, int variant) {
    // bitmask: 1 = history seed(B), 2 = policy LMR(C), 4 = policy LMP(D),
    // 8 = LMP base mass 0.15 instead of 0.05.
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    eng.setPolicyHistorySeedEnabled(variant & 1);
    eng.setPolicyLmrEnabled(variant & 2);
    eng.setPolicyLmpEnabled(variant & 4);
    if (variant & 8) eng.setPolicyLmpBaseMass(0.15);
}

const char* name(int variant) {
    switch (variant) {
        case 0:  return "off";
        case 1:  return "B-history";
        case 2:  return "C-lmr";
        case 4:  return "D-lmp.05";
        case 12: return "D-lmp.15";
        case 7:  return "all-BCD";
        default: return "?";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* weightsPath = (argc > 1) ? argv[1] : "data/nnue/nnue_weights_int8.bin";
    if (!loadWeightsQuant(weightsPath)) {
        fprintf(stderr, "[bpab] failed to load '%s'\n", weightsPath);
        return 1;
    }
    std::vector<State> pos = corpus();
    printf("[bpab] corpus=%zu positions\n", pos.size());
    printf("%-10s %14s %10s %9s | %12s %10s\n",
           "variant", "nodes_d8", "ms", "nps", "depth@200ms", "ms");

    const int variants[] = {0, 1, 2, 4, 12, 7};
    for (int v : variants) {
        // --- fixed depth ---
        uint64_t nodes = 0;
        double sec = 0.0;
        for (auto& s : pos) {
            Negamax eng;
            configure(eng, v);
            SearchStats st;
            auto t0 = std::chrono::steady_clock::now();
            // searchShallow does not reset ordering state and does not run
            // the direction-B hook (only chooseMove does): do both here so
            // every variant starts cold and B actually acts.
            eng.resetOrderingState();
            if (v & 1) eng.seedPolicyHistoryFromRoot(s);
            eng.searchShallow(s, 8, st);
            sec += secondsSince(t0);
            nodes += st.nodes;
        }
        double nps = sec > 0 ? nodes / sec : 0.0;

        // --- fixed time ---
        double depthSum = 0, tsec = 0.0;
        for (auto& s : pos) {
            Negamax eng;
            configure(eng, v);
            SearchStats st;
            auto t0 = std::chrono::steady_clock::now();
            eng.chooseMove(s, 64, 200, st);
            tsec += secondsSince(t0);
            depthSum += st.reachedDepth;
        }
        printf("%-10s %14llu %10.1f %9.0f | %12.2f %10.1f\n",
               name(v), (unsigned long long)nodes, sec * 1000.0, nps,
               depthSum / pos.size(), tsec * 1000.0 / pos.size());
    }
    return 0;
}
