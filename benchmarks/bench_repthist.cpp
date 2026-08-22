// bench_repthist -- nps com historico de repeticao REALISTA: joga N lances
// deterministicos pra encher o reptbl (como um jogo real de selfplay), e
// so entao mede buscas de profundidade fixa com esse historico carregado.
// Metrica estavel entre builds (mesmas posicoes, mesmo historico).
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;
using clockT = std::chrono::steady_clock;

// compat com refs antigos: usa push(h, irreversible) se existir
template <class T>
auto pushCompat(T& t, uint64_t h, bool irrev, int) -> decltype(t.push(h, irrev), void()) { t.push(h, irrev); }
template <class T>
void pushCompat(T& t, uint64_t h, bool, long) { t.push(h); }

int main() {
    // carrega pesos quantizados reais pro modo NNUE (se existirem)
    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin"))
        fprintf(stderr, "[aviso] pesos NNUE nao carregados -- so heuristico\n");
    // gera 6 posicoes "de meio de jogo" com historicos longos
    std::mt19937 rng(20260822);
    double totalNps = 0;
    int samples = 0;
    uint64_t totalNodes = 0;
    for (int g = 0; g < 6; g++) {
        State s = initialState();
        RepetitionTable hist;
        hist.markRoot();
        int warm = 24 + 8 * g;
        for (int p = 0; p < warm; p++) {
            if (winner(s) != -1) break;
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            // prioriza lances de PEAO que nao criam repeticao -- mantem a
            // posicao no meio-jogo (muros sobrando), que e o regime caro
            MoveList candPawn, candAny;
            for (size_t i = 0; i < moves.size(); i++) {
                State ns = applyMove(s, moves[i]);
                pushCompat(hist, ns.hash, false, 0);
                bool rep = hist.isRepetitionDraw(ns.hash);
                hist.pop();
                if (rep) continue;
                candAny.push_back(moves[i]);
                if (!moves[i].isWall) candPawn.push_back(moves[i]);
            }
            const MoveList& src = !candPawn.empty() ? candPawn : (!candAny.empty() ? candAny : moves);
            State ns = applyMove(s, src[rng() % src.size()]);
            pushCompat(hist, ns.hash,
                       s.wallsLeft[0] + s.wallsLeft[1] < WALLS_PER_PLAYER * 2, 0);
            s = ns;
        }
        if (winner(s) != -1) continue;
        Negamax eng;
        SearchStats st;
        RepetitionTable rt = hist;
        rt.markRoot();
        auto t0 = clockT::now();
        int score = eng.testNegamaxKeepTT(s, 7, -SCORE_INF, SCORE_INF, st);
        (void)score;
        double ms = std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
        totalNodes += st.nodes;
        printf("jogo %d [heur]: hist=%3d nos=%9llu tempo=%7.1fms nps=%9.0f\n", g, (int)rt.size,
               (unsigned long long)st.nodes, ms, st.nodes / (ms / 1000.0));
        totalNps += st.nodes / (ms / 1000.0);
        samples++;

        // mesmo posicao/historico em modo NNUE (producao)
        if (nnueWeightsLoaded()) {
            eng.setEvalMode(Negamax::EvalMode::NNUE);
            eng.setPolicyOrderingEnabled(true);
            eng.clearTT();  // TT já populada pela busca heurística acima devolveria cutoff imediato na raiz
            SearchStats st2;
            auto t1 = clockT::now();
            int score2 = eng.testNegamaxKeepTT(s, 7, -SCORE_INF, SCORE_INF, st2);
            (void)score2;
            double ms2 = std::chrono::duration<double, std::milli>(clockT::now() - t1).count();
            printf("jogo %d [nnue]: hist=%3d nos=%9llu tempo=%7.1fms nps=%9.0f\n", g, (int)rt.size,
                   (unsigned long long)st2.nodes, ms2, st2.nodes / (ms2 / 1000.0));
            totalNps += st2.nodes / (ms2 / 1000.0);
            samples++;
        }
    }
    printf("\nMEDIA: %.0f nos/seg | nos totais=%llu\n", totalNps / samples, (unsigned long long)totalNodes);
    return 0;
}
