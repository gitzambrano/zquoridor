// Benchmarks the active zQuoridor 4.10 MCGS and quantized NNUE.
// No production search logic changes. All fixtures and seeds are deterministic.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "mcab.hpp"
#include "time_manager.hpp"

using Mcab = mcab::MCABSearch<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                              qr::AccPair, qr::RepetitionTable, qr::SearchStats>;
using Clock = std::chrono::steady_clock;

struct Fixture {
    qr::State state;
    qr::RepetitionTable history;
    int plies;
};

static std::vector<Fixture> fixtures() {
    std::vector<Fixture> out;
    const int lengths[] = {6, 9, 12, 16, 20, 24, 29, 35, 42};
    for (int i = 0; i < 9; ++i) {
        std::mt19937 rng(20261104u + (uint32_t)i * 719u);
        qr::State s = qr::initialState();
        qr::RepetitionTable history;
        for (int ply = 0; ply < lengths[i] && qr::winner(s) < 0; ++ply) {
            qr::MoveList legal = qr::legalMoves(s);
            if (legal.empty()) break;
            qr::MoveList pawns, walls;
            for (const auto& m : legal) {
                if (m.isWall) walls.push_back(m);
                else pawns.push_back(m);
            }
            const bool chooseWall = !walls.empty() && (ply % 3 == 0 || (rng() % 5) == 0);
            const auto& options = chooseWall ? walls : (!pawns.empty() ? pawns : legal);
            const qr::Move m = options[(size_t)(rng() % options.size())];
            history.push(s.hash, m.isWall);
            s = qr::applyMove(s, m);
        }
        if (qr::winner(s) < 0) out.push_back({s, history, lengths[i]});
    }
    return out;
}

int main(int argc, char** argv) {
    int nodeBudget = 256;
    int timeMs = 0;
    int count = 9;
    bool evalCache = true, graph = true, lazy = true, adaptive = false, uncap = false;
    std::string label = "baseline";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing argument for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--nodes") nodeBudget = std::stoi(next());
        else if (a == "--ms") timeMs = std::stoi(next());
        else if (a == "--count") count = std::stoi(next());
        else if (a == "--cache") evalCache = std::stoi(next()) != 0;
        else if (a == "--graph") graph = std::stoi(next()) != 0;
        else if (a == "--lazy") lazy = std::stoi(next()) != 0;
        else if (a == "--adaptive") adaptive = std::stoi(next()) != 0;
        else if (a == "--uncap") uncap = std::stoi(next()) != 0;
        else if (a == "--label") label = next();
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (!qr::loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        std::fprintf(stderr, "Production NNUE weights did not load\n");
        return 3;
    }
    const auto states = fixtures();
    count = std::min(count, (int)states.size());
    if (count == 0) return 4;
    std::printf("HEADER label=%s count=%d nodes=%d ms=%d cache=%d graph=%d lazy=%d adaptive=%d uncap=%d\n",
                label.c_str(), count, nodeBudget, timeMs, evalCache, graph, lazy, adaptive, uncap);
    std::fflush(stdout);
    double totalMs = 0.0;
    long long totalExp = 0, totalSim = 0, policyHits = 0, policyMiss = 0;
    long long valueHits = 0, valueMiss = 0, transHits = 0, transLookups = 0;
    double maxTreeMiB = 0.0;
    for (int k = 0; k < count; ++k) {
        qr::Negamax engine;
        engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
        engine.setPolicyOrderingEnabled(true);
        Mcab search;
        search.params.evalCache = evalCache;
        search.params.transpositionGraph = graph;
        search.params.graphQCorrection = graph;
        search.params.lazyExpansion = lazy;
        search.params.treeReuse = true;
        search.params.nodeBudget = nodeBudget;
        search.params.autoNodeBudget = timeMs > 0 || adaptive;
        if (uncap) {
            search.params.autoNodeBudgetPerMs = 256;
            search.params.autoNodeBudgetCeiling = 2000000;
        }
        search.params.leafDepth = 0;
        search.params.adaptiveTime = adaptive;
        int effectiveMs = timeMs;
        if (adaptive) {
            const auto clock = zqtime::allocate(
                zqtime::TimeControl{180000, 2000, states[k].plies, 0, 20});
            search.params.adaptiveOptimumMs = clock.optimumMs;
            effectiveMs = clock.maximumMs;
        }
        qr::SearchStats stats{};
        mcab::McabStats ms{};
        auto begin = Clock::now();
        qr::Move result = search.chooseMoveMCAB(engine, states[k].state, 40,
                                                 effectiveMs, stats, states[k].history, &ms);
        const double elapsed = std::chrono::duration<double, std::milli>(
            Clock::now() - begin).count();
        const double treeMiB = (double)search.approxTreeBytes() / 1048576.0;
        maxTreeMiB = std::max(maxTreeMiB, treeMiB);
        totalMs += elapsed;
        totalExp += ms.nodesExpanded;
        totalSim += ms.simulations;
        policyHits += ms.evalCachePolicyHits;
        policyMiss += ms.evalCachePolicyMisses;
        valueHits += ms.evalCacheValueHits;
        valueMiss += ms.evalCacheValueMisses;
        transHits += ms.transpositionHits;
        transLookups += ms.transpositionLookups;
        std::printf("ROW label=%s case=%d plies=%d best=%d ms=%.3f expanded=%lld simulations=%lld "
                    "tree_mib=%.3f policy_hit=%lld policy_miss=%lld value_hit=%lld value_miss=%lld "
                    "graph_hit=%lld graph_lookup=%lld adaptive_stop=%d node_cap=%d\n",
                    label.c_str(), k, states[k].plies, (int)qr::moveToPolicyIndex(result),
                    elapsed, ms.nodesExpanded, ms.simulations, treeMiB,
                    ms.evalCachePolicyHits, ms.evalCachePolicyMisses,
                    ms.evalCacheValueHits, ms.evalCacheValueMisses,
                    ms.transpositionHits, ms.transpositionLookups, (int)ms.adaptiveTimeStop,
                    ms.effectiveNodeBudget);
        std::fflush(stdout);
    }
    const double seconds = totalMs / 1000.0;
    std::printf("SUMMARY label=%s elapsed_ms=%.3f expanded=%lld simulations=%lld expanded_s=%.3f "
                "sim_s=%.3f max_tree_mib=%.3f policy_hit=%lld policy_miss=%lld "
                "value_hit=%lld value_miss=%lld graph_hit=%lld graph_lookup=%lld\n",
                label.c_str(), totalMs, totalExp, totalSim,
                seconds > 0 ? totalExp / seconds : 0, seconds > 0 ? totalSim / seconds : 0,
                maxTreeMiB, policyHits, policyMiss, valueHits, valueMiss,
                transHits, transLookups);
    return 0;
}
