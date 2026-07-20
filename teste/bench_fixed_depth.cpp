// bench_fixed_depth.cpp -- nos/s medido em trabalho FIXO (profundidade
// fixa, janela cheia, TT limpa, 10 posições fixas via seed fixa) em vez de
// orçamento de tempo. Existe porque o benchmark de self-play do
// main.cpp (200ms/lance) mede "quanto trabalho cabe no tempo", que
// depende do quão carregada a máquina está no momento -- em ambientes
// compartilhados isso pode variar 2x+ de uma rodada pra outra mesmo sem
// nenhuma mudança de código (visto na prática ao medir a otimização de
// nós/s da Fase 4.2.10, achado por profiling com gprof). Este benchmark
// garante a MESMA contagem de nós em toda rodada (confirma primeiro que o
// resultado da busca não mudou, só a velocidade), then mede só o tempo de
// parede pra fazer aquele trabalho fixo -- comparação antes/depois mais
// limpa. Rode várias vezes e compare médias, não uma rodada só (ruído de
// CPU do ambiente ainda afeta o tempo absoluto, só não afeta mais a
// contagem de nós).
//
// Uso: g++ -O3 -std=c++17 -march=native -o bench_fixed_depth bench_fixed_depth.cpp && ./bench_fixed_depth
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <chrono>
#include <random>
#include "rules.hpp"
#include "search.hpp"
using namespace qr;

int main() {
    std::mt19937 rng(2026);
    std::vector<State> positions;
    for (int i = 0; i < 10; i++) {
        State s = initialState();
        int plies = 4 + (int)(rng() % 20);
        for (int p = 0; p < plies; p++) {
            if (winner(s) != -1) break;
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
            s = applyMove(s, moves[d(rng)]);
        }
        if (winner(s) == -1) positions.push_back(s);
    }

    uint64_t totalNodes = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (auto& s : positions) {
        Negamax eng;
        SearchStats st;
        eng.testFixedDepthFullWindow(s, 5, st);  // profundidade FIXA -- mesmo trabalho sempre
        totalNodes += st.nodes;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("nos totais=%llu tempo=%.3fs nos/seg=%.0f\n",
           (unsigned long long)totalNodes, secs, totalNodes / secs);
    return 0;
}
