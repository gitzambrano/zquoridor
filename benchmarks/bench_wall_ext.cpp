// bench_wall_ext.cpp -- nodes-to-fixed-depth for the variable wall
// quiescence extension (inv/qsendgame-ext, 2026-08).
//
// Sweeps the low-walls rule over bonus in {1,2,3} x threshold in {2,4,6}
// against the default configuration (bonus 0 = rule off) on a fixed slice
// of the low-wall corpus, at depths 8 and 10. Methodology follows
// bench_quiescence_toggle.cpp: fixed positions, fixed depth cap, huge
// time budget so nothing stops early, one fresh engine per position so
// no TT state leaks between rows of the comparison. Every number is
// therefore attributable to the parameter change alone, and the node
// counts are exact (no time-based noise).
//
// Deeper extensions explore more nodes by definition; this benchmark
// quantifies that cost per configuration so an Elo decision can weigh
// strength against node growth. Build (from the repo root):
//   g++ -O3 -std=c++17 -march=native -mavx2 -mfma -Isrc ^
//       -o bin\bench_wall_ext.exe benchmarks\bench_wall_ext.cpp
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../tools/wallext/corpus.hpp"

using namespace qr;
using namespace wallext;
using std::vector;
using clockT = std::chrono::steady_clock;

struct Row {
    int bonus, threshold;
    uint64_t nodes[2] = {0, 0};
    double ms[2] = {0.0, 0.0};
    int positions = 0;
};

int main(int argc, char** argv) {
    const int DEPTHS[2] = {8, 10};
    // Fixed subset of the lowest-wall buckets: totals 2..5, where the
    // rule under test actually fires. One position per STRIDE keeps a
    // full sweep inside minutes.
    std::vector<CorpusEntry> full = wallext::buildCorpus();
    std::vector<CorpusEntry> lowWall;
    for (const CorpusEntry& e : full) {
        if (e.s.wallsLeft[0] + e.s.wallsLeft[1] <= 5) lowWall.push_back(e);
    }
    const int STRIDE = argv[1] ? std::atoi(argv[1]) : 3;
    std::vector<CorpusEntry> subset;
    for (size_t i = 0; i < lowWall.size(); i += (size_t)STRIDE) subset.push_back(lowWall[i]);

    printf("corpus total=%zu, fatia total<=5=%zu, subconjunto medido=%zu, depths=%d/%d\n",
           full.size(), lowWall.size(), subset.size(), DEPTHS[0], DEPTHS[1]);
    printf("posicoes do subconjunto por total de muros:");
    for (int t = wallext::CORPUS_MIN_TOTAL; t <= 5; t++) {
        int n = 0;
        for (const CorpusEntry& e : subset) {
            if (e.s.wallsLeft[0] + e.s.wallsLeft[1] == t) n++;
        }
        printf(" t%d=%d", t, n);
    }
    printf("\n\n");

    struct Config { int bonus, threshold; };
    const Config CONFIGS[] = {
        {0, 0},                                  // baseline (rule off)
        {1, 2}, {1, 4}, {1, 6},
        {2, 2}, {2, 4}, {2, 6},
        {3, 2}, {3, 4}, {3, 6},
    };
    const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
    Row rows[NUM_CONFIGS];

    for (int c = 0; c < NUM_CONFIGS; c++) {
        rows[c].bonus = CONFIGS[c].bonus;
        rows[c].threshold = CONFIGS[c].threshold;
        for (size_t i = 0; i < subset.size(); i++) {
            const State& s = subset[i].s;
            Negamax eng;  // heuristic mode: cheap, fully deterministic
            if (CONFIGS[c].bonus > 0) {
                eng.setQsLowWallsBonus(CONFIGS[c].bonus);
                eng.setQsLowWallsThreshold(CONFIGS[c].threshold);
            }
            for (int d = 0; d < 2; d++) {
                SearchStats st;
                auto t0 = clockT::now();
                eng.testFixedDepthFullWindowLmr(s, DEPTHS[d], st);
                double ms = std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
                rows[c].nodes[d] += st.nodes;
                rows[c].ms[d] += ms;
            }
        }
        rows[c].positions = (int)subset.size();
        printf("feito config %d/%d (bonus=%d thr=%d)\n", c + 1, NUM_CONFIGS,
               CONFIGS[c].bonus, CONFIGS[c].threshold);
        fflush(stdout);
    }

    printf("\n=== nos ate profundidade fixa, somados sobre o subconjunto ===\n");
    printf("%-14s %14s %7s %9s %9s | %14s %7s %9s %9s\n",
           "config", "nos d8", "razao", "ms d8", "nps d8",
           "nos d10", "razao", "ms d10", "nps d10");
    for (int c = 0; c < NUM_CONFIGS; c++) {
        const Row& r = rows[c];
        char label[32];
        if (r.bonus == 0) snprintf(label, sizeof(label), "baseline");
        else snprintf(label, sizeof(label), "b=%d,t=%d", r.bonus, r.threshold);
        double ratio8 = (double)r.nodes[0] / (double)rows[0].nodes[0];
        double ratio10 = (double)r.nodes[1] / (double)rows[0].nodes[1];
        double nps8 = r.nodes[0] / (r.ms[0] / 1000.0);
        double nps10 = r.nodes[1] / (r.ms[1] / 1000.0);
        printf("%-14s %14llu %7.3f %9.0f %9.0f | %14llu %7.3f %9.0f %9.0f\n",
               label,
               (unsigned long long)r.nodes[0], ratio8, r.ms[0], nps8,
               (unsigned long long)r.nodes[1], ratio10, r.ms[1], nps10);
    }

    printf("\nleituras: razao > 1 significa que a regra explora mais nos que o baseline\n"
           "(esperado -- extensao mais funda); nps mostra o custo por no da extensao.\n");
    return 0;
}
