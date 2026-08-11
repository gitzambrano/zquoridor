// bench_lmr_pvs.cpp -- benchmark ad-hoc (não faz parte da suíte
// persistida), mesmo espírito de bench_wall_touch_bonus.cpp: mede se
// LMR+PVS+RFP+LMP (plano-additional.md, Prioridades 3+3b+3c+8) realmente
// ajudam, de duas formas independentes:
//   1. nós/s e profundidade média em posições fixas, orçamento de tempo
//      fixo -- proxy de eficiência de busca.
//   2. partidas diretas engine-vs-engine (as quatro heurísticas ligadas
//      vs desligadas, cores alternadas) -- resposta mais direta à
//      pergunta "joga melhor".
//
// Diferente de bench_wall_touch_bonus.cpp (que precisou de uma cópia
// congelada da versão antiga, porque CAT não tinha toggle em runtime),
// aqui basta os 3 setters (setLmrPvsEnabled/setRfpEnabled/setLmpEnabled)
// na MESMA classe Negamax -- todas nasceram com toggle (ver search.hpp)
// exatamente para permitir este tipo de A/B sem duplicar código.
#include <cstdio>
#include <chrono>
#include <random>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;
using clockT = std::chrono::steady_clock;
static double msSince(clockT::time_point t0) {
    return std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
}

static void setAll(Negamax& e, bool enabled) {
    e.setLmrPvsEnabled(enabled);
    e.setRfpEnabled(enabled);
    e.setLmpEnabled(enabled);
}

static std::vector<State> fixedPositions(int n, int plyMin, int plyMax) {
    std::mt19937 rng(777);
    std::vector<State> out;
    while ((int)out.size() < n) {
        State s = initialState();
        int targetPly = plyMin + (int)(rng() % (plyMax - plyMin + 1));
        for (int p = 0; p < targetPly; p++) {
            if (winner(s) != -1) break;
            MoveList moves = legalMoves(s);
            if (moves.empty()) break;
            s = applyMove(s, moves[rng() % moves.size()]);
        }
        if (winner(s) == -1) out.push_back(s);
    }
    return out;
}

static void benchNodeEfficiency() {
    printf("=== nos/s e profundidade -- 40 posicoes fixas, 200ms/lance ===\n");
    auto positions = fixedPositions(40, 5, 40);
    {
        Negamax engine;  // default: LMR+PVS+RFP+LMP ligados
        uint64_t totalNodes = 0; int depthSum = 0;
        for (auto& s : positions) {
            SearchStats st;
            engine.chooseMove(s, 40, 200, st);
            totalNodes += st.nodes; depthSum += st.reachedDepth;
        }
        printf("COM LMR+PVS+RFP+LMP: nos totais=%llu, profundidade media=%.2f\n",
               (unsigned long long)totalNodes, depthSum / (double)positions.size());
    }
    {
        Negamax engine;
        setAll(engine, false);
        uint64_t totalNodes = 0; int depthSum = 0;
        for (auto& s : positions) {
            SearchStats st;
            engine.chooseMove(s, 40, 200, st);
            totalNodes += st.nodes; depthSum += st.reachedDepth;
        }
        printf("SEM LMR+PVS+RFP+LMP: nos totais=%llu, profundidade media=%.2f\n",
               (unsigned long long)totalNodes, depthSum / (double)positions.size());
    }
}

static int playGame(Negamax& e0, Negamax& e1, int movetimeMs, int maxPlies) {
    State s = initialState();
    for (int ply = 0; ply < maxPlies; ply++) {
        int w = winner(s);
        if (w != -1) return w;
        SearchStats st;
        Move m = (s.turn == 0) ? e0.chooseMove(s, 40, movetimeMs, st)
                                : e1.chooseMove(s, 40, movetimeMs, st);
        s = applyMove(s, m);
    }
    return -1;  // nao terminou
}

static void benchHeadToHead() {
    printf("\n=== partidas diretas: COM LMR+PVS+RFP+LMP vs SEM, 150ms/lance ===\n");
    int comWins = 0, semWins = 0, indecisas = 0;
    const int NUM_GAMES = 10;
    for (int g = 0; g < NUM_GAMES; g++) {
        bool comIsPlayer0 = (g % 2 == 0);
        auto t0 = clockT::now();
        Negamax com;                 // default: tudo ligado
        Negamax sem; setAll(sem, false);
        int w = comIsPlayer0 ? playGame(com, sem, 150, 200) : playGame(sem, com, 150, 200);
        double secs = msSince(t0) / 1000.0;
        int vencedorCom = comIsPlayer0 ? 0 : 1;
        if (w == -1) { indecisas++; printf("jogo %d: indecisa (%.1fs)\n", g, secs); }
        else if (w == vencedorCom) { comWins++; printf("jogo %d: COM venceu (%.1fs)\n", g, secs); }
        else { semWins++; printf("jogo %d: SEM venceu (%.1fs)\n", g, secs); }
    }
    printf("placar: COM %d - %d SEM (%d indecisas/%d jogos)\n", comWins, semWins, indecisas, NUM_GAMES);
}

int main() {
    benchNodeEfficiency();
    benchHeadToHead();
    return 0;
}

