// bench_race_regression.cpp -- reproduz o cenario relatado: nos/s em
// posicoes REAIS que atravessam a fronteira de "muros esgotados" durante
// uma busca de verdade (chooseMove, orcamento de tempo), nao so chamadas
// repetidas com a MESMA topologia (o que o teste unitario ja cobre bem).
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <chrono>
#include <random>
#include "rules.hpp"
#include "search.hpp"
#include "endgame_race.hpp"
using namespace qr;

int main() {
    std::mt19937 rng(2026);
    std::vector<State> positions;
    // Joga partidas aleatorias ate os dois lados ficarem com POUCOS muros
    // (mas ainda >0 para pelo menos um, para pegar a TRANSICAO real para
    // wallsLeft==0,0 durante a busca, nao so posicoes ja no final).
    for (int game = 0; game < 30 && (int)positions.size() < 12; game++) {
        State s = initialState();
        for (int p = 0; p < 300; p++) {
            if (winner(s) != -1) break;
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
            s = applyMove(s, moves[d(rng)]);
            if (s.wallsLeft[0] <= 1 && s.wallsLeft[1] <= 1 && winner(s) == -1) {
                positions.push_back(s);
                break;
            }
        }
    }
    printf("posicoes de teste (poucos muros restantes): %zu\n", positions.size());

    uint64_t totalNodes = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (auto& s : positions) {
        Negamax eng;
        SearchStats st;
        Move mv = eng.chooseMove(s, 20, 300, st);  // mesmo orcamento tipico de arena/selfplay
        totalNodes += st.nodes;
        (void)mv;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("nos totais=%llu tempo=%.3fs nos/seg=%.0f\n",
           (unsigned long long)totalNodes, secs, totalNodes / secs);
    printf("race cache: hits=%llu misses=%llu (razao hit=%.4f)\n",
           (unsigned long long)g_raceCacheHits, (unsigned long long)g_raceCacheMisses,
           (double)g_raceCacheHits / (double)(g_raceCacheHits + g_raceCacheMisses + 1));
    return 0;
}
