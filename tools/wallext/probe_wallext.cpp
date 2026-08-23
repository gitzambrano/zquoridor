// probe_wallext.cpp -- timing probe for the experiment tools.
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <chrono>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../tools/wallext/corpus.hpp"

using namespace qr;
using namespace wallext;
using clockT = std::chrono::steady_clock;

int main() {
    auto t0 = clockT::now();
    std::vector<CorpusEntry> full = buildCorpus();
    double genMs = std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
    printf("buildCorpus: %.0f ms, %zu posicoes\n", genMs, full.size());
    int hist[11] = {};
    for (const CorpusEntry& e : full) hist[e.s.wallsLeft[0] + e.s.wallsLeft[1]]++;
    for (int t = CORPUS_MIN_TOTAL; t <= 5; t++) printf("  total %d: %d posicoes\n", t, hist[t]);

    // First few low-wall positions: single depth-8 search, with and
    // without the exact race solver, to attribute the per-node cost.
    int tried = 0;
    for (size_t i = 0; i < full.size() && tried < 3; i++) {
        const State& s = full[i].s;
        if (s.wallsLeft[0] + s.wallsLeft[1] != 3) continue;
        Negamax eng;
        SearchStats st;
        auto a = clockT::now();
        eng.testFixedDepthFullWindowLmr(s, 8, st);
        double ms8 = std::chrono::duration<double, std::milli>(clockT::now() - a).count();
        qr::g_raceExactBudgetUs = 0.0;  // solver off: heuristic fall-through everywhere
        SearchStats st2;
        auto b = clockT::now();
        eng.testFixedDepthFullWindowLmr(s, 8, st2);
        double ms8off = std::chrono::duration<double, std::milli>(clockT::now() - b).count();
        qr::g_raceExactBudgetUs = 1e18;
        printf("pos %zu (total=3): d8 COM race nos=%lld ms=%.0f (%.0f nps) | SEM race nos=%lld ms=%.0f (%.0f nps)\n",
               i, (long long)st.nodes, ms8, st.nodes / (ms8 / 1000.0),
               (long long)st2.nodes, ms8off, st2.nodes / (ms8off / 1000.0));
        tried++;
    }
    return 0;
}
