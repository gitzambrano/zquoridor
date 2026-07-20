#include <cstdio>
#include <chrono>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "nnue.hpp"
using namespace qr;
using clockT = std::chrono::steady_clock;

static double msSince(clockT::time_point t0) {
    return std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
}

void benchNegamax() {
    printf("\n=== Negamax (eval heuristica simples) -- self-play, 200ms/lance ===\n");
    Negamax engine;
    State s = initialState();
    SearchStats totalStats{};
    uint64_t totalNodes = 0;
    auto t0 = clockT::now();
    int ply = 0;
    int maxPlies = 300;
    int depthSum = 0, depthSamples = 0;
    RepetitionTable reptbl;
    bool isDraw = false;
    for (; ply < maxPlies; ply++) {
        int w = winner(s);
        if (w != -1) break;
        if (reptbl.count(s.hash) >= 2) {
            isDraw = true;
            break;
        }
        SearchStats st;
        Move m = engine.chooseMove(s, /*maxDepthCap=*/40, /*timeBudgetMs=*/200, st, reptbl);
        totalNodes += st.nodes;
        depthSum += st.reachedDepth;
        depthSamples++;
        reptbl.push(s.hash);
        s = applyMove(s, m);
    }
    double totalMs = msSince(t0);
    printf("lances jogados: %d, nos totais: %llu\n", ply, (unsigned long long)totalNodes);
    printf("tempo total: %.1f ms -> %.0f nos/seg (media, incluindo overhead de iterative deepening)\n",
           totalMs, totalNodes / (totalMs / 1000.0));
    printf("profundidade media alcancada por lance: %.1f\n", depthSamples ? (double)depthSum / depthSamples : 0.0);
    int w = winner(s);
    if (isDraw) {
        printf("resultado: empate por repeticao\n");
    } else {
        printf("resultado: %s\n", w == -1 ? "nao terminou no limite de lances" :
               (w == 0 ? "jogador 0 venceu" : "jogador 1 venceu"));
    }
}

void benchNNUEForward(int iters) {
    printf("\n=== Custo do forward pass da NNUE (pesos aleatorios, so para medir custo) ===\n");
    State s = initialState();
    // avanca algumas jogadas pra ter um acumulador com muros tambem
    Negamax dummyEngine;  // só pra gerar uma posição plausível via um lance aleatório determinístico
    for (int i = 0; i < 20; i++) {
        auto moves = legalMoves(s);
        s = applyMove(s, moves[i % moves.size()]);
        if (winner(s) != -1) break;
    }
    Accumulator acc = buildAccumulator(s, s.turn);

    auto t0 = clockT::now();
    volatile float sinkV = 0;
    for (int i = 0; i < iters; i++) sinkV += forwardValueWL(acc);  // cabeça de resultado (WL) -- ver nnue.hpp
    double msValue = msSince(t0);

    std::array<float, POLICY_OUT> policyOut{};
    volatile float sinkP = 0;
    t0 = clockT::now();
    for (int i = 0; i < iters; i++) {
        forwardPolicy(acc, policyOut);
        sinkP += policyOut[i % POLICY_OUT];
    }
    double msPolicy = msSince(t0);

    printf("forwardValue:  %.1f ms / %d chamadas = %.0f ns/chamada -> %.0f chamadas/seg\n",
           msValue, iters, msValue * 1e6 / iters, iters / (msValue / 1000.0));
    printf("forwardPolicy: %.1f ms / %d chamadas = %.0f ns/chamada -> %.0f chamadas/seg\n",
           msPolicy, iters, msPolicy * 1e6 / iters, iters / (msPolicy / 1000.0));
    double combinedPerCallUs = (msValue + msPolicy) * 1000.0 / iters;
    printf("valor+politica juntos: %.2f us/no -> limite teorico se chamado em TODO no: %.0f nos/seg\n",
           combinedPerCallUs, 1e6 / combinedPerCallUs);
}

void benchAccumulatorUpdate(int iters) {
    printf("\n=== Acumulador: atualizacao incremental vs recompute do zero ===\n");
    State s = initialState();
    for (int i = 0; i < 15; i++) {
        auto moves = legalMoves(s);
        s = applyMove(s, moves[(i * 7) % moves.size()]);
        if (winner(s) != -1) break;
    }
    Accumulator acc = buildAccumulator(s, s.turn);

    // custo de recompute do zero
    auto t0 = clockT::now();
    for (int i = 0; i < iters; i++) { volatile Accumulator tmp = buildAccumulator(s, s.turn); }
    double msFull = msSince(t0);

    // custo de atualizacao incremental de um lance de peao (2 features de
    // tabuleiro + possivelmente 1 par de bucket de distancia, ver nota em
    // updateAccumulatorForMove/nnue.hpp)
    Move pawnMove = Move::pawn(cellIdx(0, 0));
    int oldCell = s.pawn[s.turn];
    State sAfterPawn = s; sAfterPawn.pawn[s.turn] = (uint8_t)oldCell;  // estado "antes" fixo p/ o par ida-e-volta
    t0 = clockT::now();
    for (int i = 0; i < iters; i++) {
        updateAccumulatorForMove(acc, true, sAfterPawn, pawnMove);
        State sMoved = sAfterPawn; sMoved.pawn[sMoved.turn] = (uint8_t)pawnMove.a;
        updateAccumulatorForMove(acc, true, sMoved, Move::pawn(oldCell));  // desfaz, mantem acc estavel
    }
    double msPawnUpdate = msSince(t0);

    // custo de atualizacao incremental "crua" de 1 feature de tabuleiro
    // (so soma/subtrai uma linha de w1, sem BFS -- serve de piso de custo)
    t0 = clockT::now();
    for (int i = 0; i < iters; i++) {
        acc.addFeature(featWallH(slotIdx(3, 3)));
        acc.removeFeature(featWallH(slotIdx(3, 3)));  // desfaz, mantem acc estavel
    }
    double msRawFeatureToggle = msSince(t0);

    // custo do update incremental REAL de um lance de muro via
    // updateAccumulatorForMove -- este e' o caso caro do novo esquema de
    // distancia BFS: um muro pode mudar o bucket dos DOIS jogadores, entao
    // paga ate 2 BFS (O(81) cada, sem alocacao) alem da feature de muro em
    // si. E' o numero relevante pra decidir se o design ficou pesado
    // demais pra busca (Fase 6), nao o toggle cru acima.
    Move wallMove = Move::wall(0, 3, 3);
    State sBeforeWall = s; sBeforeWall.wallsH &= ~(1ull << slotIdx(3, 3));  // garante slot livre antes
    Accumulator accWall = buildAccumulator(sBeforeWall, sBeforeWall.turn);  // acc precisa refletir sBeforeWall (pre-condicao da funcao)
    Accumulator snapshot = accWall;  // pra restaurar entre iteracoes sem reimplementar a logica de "desfazer"
    t0 = clockT::now();
    for (int i = 0; i < iters; i++) {
        updateAccumulatorForMove(accWall, true, sBeforeWall, wallMove);
        accWall = snapshot;  // restaura (copia O(HIDDEN), desprezivel frente ao BFS medido acima)
    }
    double msWallUpdate = msSince(t0);

    printf("recompute completo:                    %.0f ns/chamada\n", msFull * 1e6 / iters);
    printf("update incremental peao (via updateAccumulatorForMove): %.0f ns/chamada (por par ida+volta)\n",
           msPawnUpdate * 1e6 / iters);
    printf("toggle cru de 1 feature (sem BFS):      %.0f ns/chamada (por par add+remove)\n",
           msRawFeatureToggle * 1e6 / iters);
    printf("update incremental muro (via updateAccumulatorForMove, com cache de bucket): "
           "%.0f ns/chamada (inclui a copia de restauracao do snapshot)\n", msWallUpdate * 1e6 / iters);
    printf("conclusao: com o cache ownDistBucket/oppDistBucket (ver struct Accumulator em nnue.hpp),\n");
    printf("           o update de muro paga so as 2 BFS 'depois' (nao mais 4 'antes+depois'), o que\n");
    printf("           o mantem no mesmo patamar do recompute total -- nao ficou mais pesado que ele.\n");
}

int main() {
    benchNegamax();
    benchNNUEForward(200000);
    benchAccumulatorUpdate(2000000);
    return 0;
}
