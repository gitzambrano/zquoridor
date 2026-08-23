// knob_check.cpp -- does the low-walls rule change anything at all?
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../tools/wallext/corpus.hpp"

using namespace qr;
using namespace wallext;

int main() {
    std::vector<CorpusEntry> corpus = buildCorpus();
    int changed = 0, same = 0, tried = 0;
    for (size_t i = 0; i < corpus.size(); i += 7) {
        const State& s = corpus[i].s;
        int total = s.wallsLeft[0] + s.wallsLeft[1];
        if (total > 6) continue;
        Negamax base;
        SearchStats stB;
        int scB = base.testFixedDepthFullWindowLmr(s, 6, stB);
        Negamax var;
        var.setQsLowWallsBonus(2);
        var.setQsLowWallsThreshold(6);
        SearchStats stV;
        int scV = var.testFixedDepthFullWindowLmr(s, 6, stV);
        tried++;
        if (stV.nodes != stB.nodes || scV != scB) {
            changed++;
            if (changed <= 5) {
                printf("pos %zu total=%d: nos %lld -> %lld, score %d -> %d\n",
                       i, total, (long long)stB.nodes, (long long)stV.nodes, scB, scV);
            }
        } else {
            same++;
        }
    }
    printf("tried=%d mudou=%d igual=%d\n", tried, changed, same);

    // Also check how often quiescence even reaches qply >= 1 / >= 2 here:
    // instrument by comparing caps 0 vs 1 vs 2 vs 4 on one hot position.
    size_t best = 0;
    long long bestDelta = -1;
    for (size_t i = 0; i < corpus.size(); i += 3) {
        const State& s = corpus[i].s;
        Negamax e0;
        SearchStats s0;
        e0.testFixedDepthFullWindowLmr(s, 6, s0);
        Negamax e2;
        e2.setQsMaxExtraPlies(0);
        SearchStats s2;
        e2.testFixedDepthFullWindowLmr(s, 6, s2);
        long long d = (long long)s0.nodes - (long long)s2.nodes;
        if (d > bestDelta) { bestDelta = d; best = i; }
    }
    printf("posicao com maior ganho de extensao: %zu (delta=%lld nos)\n", best, bestDelta);
    {
        const State& s = corpus[best].s;
        for (int cap : {0, 1, 2, 3, 4}) {
            Negamax e;
            if (cap != 2) e.setQsMaxExtraPlies(cap);
            SearchStats st;
            int sc = e.testFixedDepthFullWindowLmr(s, 6, st);
            printf("  cap=%d: nos=%lld score=%d\n", cap, (long long)st.nodes, sc);
        }
        // Low-walls rule on that position, thresholds sweeping.
        for (int thr : {2, 4, 6}) {
            for (int b : {1, 2, 3}) {
                Negamax e;
                e.setQsLowWallsBonus(b);
                e.setQsLowWallsThreshold(thr);
                SearchStats st;
                int sc = e.testFixedDepthFullWindowLmr(s, 6, st);
                printf("  b=%d,t=%d: nos=%lld score=%d\n", b, thr, (long long)st.nodes, sc);
            }
        }
    }
    return 0;
}
