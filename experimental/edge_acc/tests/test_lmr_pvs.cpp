// test_lmr_pvs.cpp -- validação de LMR (Late Move Reduction, Prioridade 3
// do plano-additional.md) + PVS (Principal Variation Search, Prioridade 8)
// + RFP (Reverse Futility Pruning, Prioridade 3b) + LMP (Late Move
// Pruning, Prioridade 3c). As quatro nasceram com toggle em runtime
// (setLmrPvsEnabled/setRfpEnabled/setLmpEnabled) e testFixedDepthFullWindow
// desliga as quatro de uma vez -- então testFixedDepthFullWindowLmr usada
// abaixo (que NÃO desliga nada, usa os defaults de produção) já exercita
// as quatro juntas, não só LMR/PVS isoladas.
//
// Diferente de test_search_staging.cpp (que exige "0 divergências de
// score" porque compara duas implementações que deveriam ser
// MATEMATICAMENTE idênticas -- staging é só um refactor), LMR é uma
// aproximação heurística POR DESENHO: reduz a profundidade de lances
// tardios e só reverifica em profundidade cheia se o resultado reduzido
// "vazar" acima de alpha. Isso significa que, em casos raros, uma busca
// LMR+PVS pode divergir do valor de uma busca de janela cheia na MESMA
// profundidade nominal (instabilidade de busca aceita -- todo motor forte
// tem isso). O que este arquivo valida, então, não é "bate sempre", e sim:
//   1. Nunca crasha, nunca devolve lance ilegal.
//   2. Taxa de concordância de score com a referência de janela cheia é
//      ALTA (o oposto seria sinal de bug real, não de instabilidade
//      normal) -- piso de regressão, não uma prova de corretude.
//   3. Em posições onde a referência já decidiu com folga clara (score
//      fortemente positivo/negativo -- candidatas a "gargalo tático", o
//      caso que o plano pede pra checar com cuidado extra antes de
//      aceitar LMR), a concordância deve ser ainda mais alta que a média.
//   4. Reduz nós-para-mesma-profundidade de forma real (evidência de que
//      "melhora profundidade", o motivo de implementar isto).
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

int main() {
    std::mt19937 rng(2027);
    const int NUM_POSITIONS = 100;
    const int DEPTH = 4;  // mesmo criterio de test_search_staging.cpp -- depth 5
                           // deixava o lado SEM LMR (referencia de janela cheia,
                           // sem a poda que justamente estamos medindo) lento
                           // demais para rodar como parte da suite de rotina.

    int checked = 0;
    int matches = 0;
    int decisiveChecked = 0;   // |scoreOff| >= DECISIVE_THRESHOLD
    int decisiveMatches = 0;
    long long nodesOff = 0, nodesLmr = 0;
    int illegalMoves = 0;
    const int DECISIVE_THRESHOLD = 300;  // score bem acima de ruído de eval heurística

    for (int game = 0; game < NUM_POSITIONS; game++) {
        State s = initialState();
        int stopPly = 5 + (int)(rng() % 35);
        bool reachedEnd = false;
        for (int ply = 0; ply < stopPly; ply++) {
            auto moves = legalMoves(s);
            if (winner(s) != -1 || moves.empty()) { reachedEnd = true; break; }
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            s = applyMove(s, moves[dist(rng)]);
        }
        if (reachedEnd || winner(s) != -1) continue;

        Negamax offEngine;
        SearchStats statsOff;
        int scoreOff = offEngine.testFixedDepthFullWindow(s, DEPTH, statsOff);  // LMR/PVS desligado (referencia)

        Negamax lmrEngine;
        SearchStats statsLmr;
        int scoreLmr = lmrEngine.testFixedDepthFullWindowLmr(s, DEPTH, statsLmr);  // LMR/PVS ligado (producao)

        checked++;
        nodesOff += statsOff.nodes;
        nodesLmr += statsLmr.nodes;
        bool same = (scoreOff == scoreLmr);
        if (same) matches++;

        if (std::abs(scoreOff) >= DECISIVE_THRESHOLD) {
            decisiveChecked++;
            if (same) decisiveMatches++;
        }

        // smoke check: chooseMove (LMR/PVS ligado, caminho de producao
        // de verdade -- aspiration/iterative deepening) sempre devolve
        // lance legal, mesmo quando o score bruto tem a fragilidade de
        // aspiration+TT-entre-iteracoes ja documentada em
        // test_search_staging.cpp.
        Negamax prodEngine;
        SearchStats statsChoose;
        Move mv = prodEngine.chooseMove(s, DEPTH, 200, statsChoose);
        auto legal = legalMoves(s);
        bool mvLegal = std::find(legal.begin(), legal.end(), mv) != legal.end();
        if (!mvLegal) {
            illegalMoves++;
            printf("FALHOU: chooseMove (LMR+PVS ligado) devolveu lance ilegal na posicao %d\n", game);
            return 1;
        }
    }

    double agreementRate = checked > 0 ? 100.0 * matches / checked : 0.0;
    double decisiveRate = decisiveChecked > 0 ? 100.0 * decisiveMatches / decisiveChecked : 100.0;
    double nodeRatio = nodesOff > 0 ? (double)nodesLmr / nodesOff : 0.0;

    printf("posicoes checadas=%d, lances ilegais=%d\n", checked, illegalMoves);
    printf("concordancia de score (LMR+PVS vs janela cheia, depth=%d): %d/%d (%.1f%%)\n",
           DEPTH, matches, checked, agreementRate);
    printf("concordancia em posicoes decisivas (|score|>=%d): %d/%d (%.1f%%)\n",
           DECISIVE_THRESHOLD, decisiveMatches, decisiveChecked, decisiveRate);
    printf("nos totais -- sem LMR/PVS: %lld, com LMR/PVS: %lld (razao %.3fx)\n",
           nodesOff, nodesLmr, nodeRatio);

    // Pisos de regressao (nao sao prova de corretude -- LMR e heuristica
    // por desenho, ver comentario no topo do arquivo). Valores escolhidos
    // com folga generosa sobre o que uma implementacao correta deveria
    // produzir; uma queda abaixo disso e sinal de BUG (reducao/gate
    // errado), nao de instabilidade normal de busca.
    bool ok = true;
    if (illegalMoves > 0) { printf("FALHOU: lance ilegal encontrado\n"); ok = false; }
    if (agreementRate < 85.0) { printf("FALHOU: concordancia de score abaixo do piso (85%%)\n"); ok = false; }
    if (decisiveRate < 90.0) { printf("FALHOU: concordancia em posicoes decisivas abaixo do piso (90%%)\n"); ok = false; }
    if (nodeRatio > 0.95) {
        printf("AVISO: LMR+PVS reduziu nos menos do que o esperado (razao %.3fx, esperado bem < 1.0x) -- "
               "nao falha o teste, mas vale investigar se a reducao esta disparando com a frequencia certa\n", nodeRatio);
    }

    if (ok) {
        printf("OK -- LMR+PVS dentro dos pisos de regressao esperados\n");
        return 0;
    }
    return 1;
}
