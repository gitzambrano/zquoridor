// bench_mcab_equivalence.cpp -- Fase 3 do plano plan-hybrid-mc-ab.md:
// validação de CORRETUDE do híbrido MCαβ antes de gastar qualquer ciclo de
// arena real (Seção 6).
//
// O modo equivalência (`mcabNodeBudget <= 1`) reduz chooseMoveMCAB a: para
// cada lance legal da raiz, uma chamada `searchLeaf(filho, leafDepth)`,
// convertida por scoreToQ, escolhendo o maior Q. Sem árvore, sem PUCT. Ou
// seja, é exatamente um minimax de raiz em profundidade `leafDepth + 1`,
// só que atravessando toda a cadeia nova (searchLeaf, seedAcc incremental,
// conversão score->Q, comparação em espaço de probabilidade). Se o sinal,
// a perspectiva ou a conversão estiverem errados em qualquer ponto dessa
// cadeia, o lance escolhido diverge do que o AB puro escolheria na MESMA
// profundidade -- é isso que este benchmark mede.
//
// Referência de comparação: `Negamax::chooseMove(root, leafDepth+1, ...)`
// com orçamento de tempo folgado, para que a busca chegue de fato à
// profundidade nominal.
//
// IMPORTANTE sobre o critério de aprovação: divergência de LANCE sozinha
// não é bug. O AB puro usa aprofundamento iterativo, TT, aspiração e
// ordenação -- quando dois lances empatam em score, qual dos dois sai como
// "melhor" é um detalhe de ordem de visita, e os dois caminhos podem
// legitimamente escolher lances diferentes de MESMO valor. O que seria bug
// é divergência de SCORE: o lance do MCAB valendo menos que o lance do AB
// medido pela mesma régua. Por isso o benchmark reporta as duas coisas
// separadamente, e só o delta de score conta como falha.
//
// Uso: bin\bench_mcab_equivalence.exe [leafDepth] [numPosicoes]
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../tools/common/mcab.hpp"

using namespace qr;

using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                               RepetitionTable, SearchStats>;

namespace {

// Régua comum: score do lance `m` a partir de `root`, do ponto de vista de
// quem joga em `root`, medido por uma busca AB limpa de profundidade
// `leafDepth` no filho (TT/killers/history zerados a cada medição, para
// que a ordem das medições não influencie o resultado).
int scoreOfMove(const State& root, const Move& m, int leafDepth) {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    eng.clearTT();
    eng.resetOrderingState();
    SearchStats st;
    RepetitionTable reptbl;
    reptbl.markRoot();  // mesma convenção de chooseMove/chooseMoveMCAB: tudo
                        // antes daqui seria histórico real de partida
    State child = applyMove(root, m);
    if (winner(child) != -1) {
        // Lance que já termina o jogo: valor exato, sem busca.
        return (winner(child) == root.turn) ? (SCORE_INF - 1) : -(SCORE_INF - 1);
    }
    AccPair seed = buildAccPairRoot(child, nullptr);
    return -eng.searchLeaf(child, leafDepth, st, reptbl, &seed);
}

std::string moveStr(const Move& m) {
    char buf[64];
    if (m.isWall) {
        std::snprintf(buf, sizeof(buf), "muro%c(r%d,c%d)", m.a == 0 ? 'H' : 'V', (int)m.b, (int)m.c);
    } else {
        std::snprintf(buf, sizeof(buf), "peao(r%d,c%d)", (int)m.a / N, (int)m.a % N);
    }
    return std::string(buf);
}

}  // namespace

int main(int argc, char** argv) {
    int leafDepth = (argc > 1) ? std::atoi(argv[1]) : 4;
    int numPositions = (argc > 2) ? std::atoi(argv[2]) : 20;

    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        printf("[AVISO] nao consegui carregar data/nnue/nnue_weights_int8.bin --\n"
               "        rode a partir da RAIZ do repo. Seguindo com pesos zerados:\n"
               "        o teste de equivalencia continua valido (mede a cadeia\n"
               "        searchLeaf->scoreToQ, nao a qualidade da rede), mas as\n"
               "        posicoes ficam todas empatadas e o resultado e menos\n"
               "        informativo.\n\n");
    }

    // Mesmo gerador de posições do bench_fixed_depth.cpp (seed fixa =
    // conjunto de posições reproduzível entre rodadas).
    std::mt19937 rng(2026);
    std::vector<State> positions;
    while ((int)positions.size() < numPositions) {
        State s = initialState();
        int plies = 4 + (int)(rng() % 24);
        for (int p = 0; p < plies; p++) {
            if (winner(s) != -1) break;
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
            s = applyMove(s, moves[d(rng)]);
        }
        // Posições de "mãos vazias" são delegadas ao solver exato pelo
        // próprio chooseMoveMCAB (Seção 5, passo 1) -- não exercitam a
        // cadeia que este benchmark quer validar.
        if (winner(s) == -1 && (s.wallsLeft[0] > 0 || s.wallsLeft[1] > 0)) {
            positions.push_back(s);
        }
    }

    printf("=== Fase 3: modo equivalencia (mcabNodeBudget=0) vs AB puro ===\n");
    printf("leafDepth=%d  (AB de referencia roda em profundidade %d)  posicoes=%d\n\n",
           leafDepth, leafDepth + 1, numPositions);

    int sameMove = 0;
    int sameScore = 0;
    int worseScore = 0;
    long long absScoreDeltaSum = 0;
    int maxAbsScoreDelta = 0;

    for (size_t i = 0; i < positions.size(); i++) {
        const State& root = positions[i];

        // --- Caminho A: AB puro na mesma profundidade nominal ----------
        Negamax engAB;
        engAB.setEvalMode(Negamax::EvalMode::NNUE);
        SearchStats stAB;
        RepetitionTable histAB;
        Move mAB = engAB.chooseMove(root, leafDepth + 1, /*timeBudgetMs=*/600000, stAB, histAB);

        // --- Caminho B: MCAB em modo equivalencia ----------------------
        Negamax engMC;
        engMC.setEvalMode(Negamax::EvalMode::NNUE);
        Mcab mc;
        mc.params.enabled = true;
        mc.params.nodeBudget = 0;  // Secao 6 -- modo equivalencia
        mc.params.leafDepth = leafDepth;
        SearchStats stMC;
        RepetitionTable histMC;
        mcab::McabStats mstats;
        Move mMC = mc.chooseMoveMCAB(engMC, root, leafDepth + 1, /*timeBudgetMs=*/600000,
                                      stMC, histMC, &mstats);

        // --- Comparacao pela regua comum -------------------------------
        int scAB = scoreOfMove(root, mAB, leafDepth);
        int scMC = scoreOfMove(root, mMC, leafDepth);
        int delta = scMC - scAB;  // negativo = MCAB escolheu um lance PIOR

        bool moveEq = (mAB == mMC);
        if (moveEq) sameMove++;
        if (delta == 0) sameScore++;
        if (delta < 0) worseScore++;
        absScoreDeltaSum += (delta < 0 ? -delta : delta);
        if ((delta < 0 ? -delta : delta) > maxAbsScoreDelta) maxAbsScoreDelta = (delta < 0 ? -delta : delta);

        printf("pos %2zu | AB=%-16s (%6d) | MCAB=%-16s (%6d) | delta=%+5d %s%s\n",
               i, moveStr(mAB).c_str(), scAB, moveStr(mMC).c_str(), scMC, delta,
               moveEq ? "[lance igual]" : "[lance diferente]",
               delta < 0 ? " <-- MCAB PIOR" : "");
    }

    int n = (int)positions.size();
    printf("\n--- Resumo ---\n");
    printf("lance identico ao AB puro : %d/%d (%.1f%%)  [informativo -- empates de score sao legitimos]\n",
           sameMove, n, 100.0 * sameMove / n);
    printf("score identico ao AB puro : %d/%d (%.1f%%)\n", sameScore, n, 100.0 * sameScore / n);
    printf("MCAB escolheu lance PIOR  : %d/%d  <-- CRITERIO DE FALHA DA FASE 3\n", worseScore, n);
    printf("delta de score |medio|    : %.1f   maximo: %d\n",
           (double)absScoreDeltaSum / n, maxAbsScoreDelta);

    if (worseScore == 0) {
        printf("\nOK -- Fase 3 aprovada: o modo equivalencia nunca escolheu um lance\n"
               "de score inferior ao do AB puro na mesma profundidade. A cadeia\n"
               "searchLeaf/seedAcc/scoreToQ/comparacao esta com sinal e perspectiva\n"
               "corretos.\n");
        return 0;
    }
    printf("\nFALHA -- ha divergencia de SCORE (nao so de lance): bug de sinal,\n"
           "perspectiva ou conversao score->Q na integracao. Ver Secao 6/Fase 3\n"
           "do plano antes de seguir para a Fase 4.\n");
    return 1;
}
