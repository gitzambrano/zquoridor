// bench_mcab.cpp -- Fase 4 do plano plan-hybrid-mc-ab.md: custo isolado do
// híbrido MCαβ, e validação formal do requisito não-negociável da Seção 0.
//
// Duas medições, nesta ordem de importância:
//
// A) REGRESSÃO ZERO NO AB PURO (Seção 0/13). Este binário inclui
//    `src/mcab.hpp`; `benchmarks/bench_fixed_depth.cpp` não. A
//    primeira parte roda EXATAMENTE a mesma carga fixa daquele benchmark
//    (mesma seed 2026, mesmas 10 posições, mesma profundidade 5, mesma
//    `testFixedDepthFullWindow`) para que os dois resultados sejam
//    diretamente comparáveis:
//
//      bin\bench_fixed_depth.exe      <- baseline sem mcab.hpp
//      bin\bench_mcab.exe             <- mesma carga, binário com mcab.hpp
//
//    A CONTAGEM DE NÓS tem que bater exatamente (é determinística: mesma
//    carga, mesmo resultado de busca). Nós/s pode variar dentro do ruído de
//    CPU do ambiente -- rode as duas várias vezes e compare médias, como o
//    cabeçalho do bench_fixed_depth já recomenda.
//
// B) CUSTO DO HÍBRIDO. Reportado em "nós de AB equivalentes/s", não em
//    "nós MCTS/s": um nó MCTS custa 1 forward de política + uma busca AB
//    inteira de `leafDepth` plies, então nós-MCTS/s isolado seria uma
//    métrica enganosa (Seção 12, primeiro risco). O que se compara com o
//    AB puro é a soma de `SearchStats::nodes` de todas as folhas dividida
//    pelo tempo de parede.
//
// Uso: bin\bench_mcab.exe [nodeBudget] [leafDepth]
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;

using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                               RepetitionTable, SearchStats>;

namespace {

// Mesmo gerador de posições do bench_fixed_depth.cpp -- seed e parâmetros
// idênticos de propósito, para que as duas medições de AB puro sejam sobre
// o MESMO conjunto de posições.
std::vector<State> fixedPositions() {
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
    return positions;
}

}  // namespace

int main(int argc, char** argv) {
    int nodeBudget = (argc > 1) ? std::atoi(argv[1]) : 2000;
    int leafDepth = (argc > 2) ? std::atoi(argv[2]) : 4;

    std::vector<State> positions = fixedPositions();

    // ------------------------------------------------------------------
    // A) AB puro -- carga idêntica à do bench_fixed_depth.cpp
    // ------------------------------------------------------------------
    {
        uint64_t totalNodes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (auto& s : positions) {
            Negamax eng;
            SearchStats st;
            eng.testFixedDepthFullWindow(s, 5, st);
            totalNodes += st.nodes;
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        printf("=== A) AB PURO (Secao 0: regressao zero) ===\n");
        printf("carga identica a benchmarks/bench_fixed_depth.cpp, mas num binario\n"
               "que inclui src/mcab.hpp.\n");
        printf("nos totais=%llu tempo=%.3fs nos/seg=%.0f\n",
               (unsigned long long)totalNodes, secs, totalNodes / secs);
        printf("--> compare com `bin\\bench_fixed_depth.exe`: a CONTAGEM DE NOS tem\n"
               "    que ser identica; nos/seg dentro do ruido de CPU do ambiente.\n\n");
    }

    // ------------------------------------------------------------------
    // B) Híbrido MCαβ -- custo por chamada de chooseMoveMCAB
    // ------------------------------------------------------------------
    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        printf("[AVISO] nao consegui carregar data/nnue/nnue_weights_int8.bin -- rode a\n"
               "        partir da RAIZ do repo. Sem pesos, os priors de politica ficam\n"
               "        uniformes e o custo medido abaixo NAO representa o caso real.\n\n");
    }

    printf("=== B) HIBRIDO MCab (custo isolado) ===\n");
    printf("nodeBudget=%d leafDepth=%d posicoes=%zu\n\n", nodeBudget, leafDepth, positions.size());
    printf("%3s %9s %9s %9s %8s %12s %10s\n",
           "pos", "expand", "simul", "folhas", "prof.med", "nos-AB", "arvore-KB");

    long long grandNodes = 0, grandExpanded = 0, grandLeaves = 0;
    double grandSecs = 0.0;
    size_t peakTreeBytes = 0;

    for (size_t i = 0; i < positions.size(); i++) {
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::NNUE);

        Mcab mc;
        mc.params.enabled = true;
        mc.params.nodeBudget = nodeBudget;
        mc.params.leafDepth = leafDepth;

        SearchStats st;
        RepetitionTable hist;
        mcab::McabStats ms;

        auto t0 = std::chrono::steady_clock::now();
        mc.chooseMoveMCAB(eng, positions[i], /*maxDepthCap=*/40,
                           /*timeBudgetMs=*/0 /* sem teto de tempo: mede o custo do orcamento de nos */,
                           st, hist, &ms);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        size_t treeBytes = mc.approxTreeBytes();
        if (treeBytes > peakTreeBytes) peakTreeBytes = treeBytes;

        double avgLeafDepth = ms.leafSearches > 0 ? (double)ms.leafDepthSum / ms.leafSearches : 0.0;
        // Posições em que os dois lados já ficaram sem muros são delegadas
        // ao solver exato de corrida por chooseMoveMCAB (Seção 5, passo 1)
        // -- linha com tudo zerado é isso, não árvore vazia por bug.
        bool emptyHanded = (positions[i].wallsLeft[0] == 0 && positions[i].wallsLeft[1] == 0);
        printf("%3zu %9lld %9lld %9lld %8.2f %12llu %10.1f%s\n",
               i, ms.nodesExpanded, ms.simulations, ms.leafSearches, avgLeafDepth,
               (unsigned long long)st.nodes, treeBytes / 1024.0,
               emptyHanded ? "   <- maos vazias: delegado ao solver exato" : "");

        grandNodes += (long long)st.nodes;
        grandExpanded += ms.nodesExpanded;
        grandLeaves += ms.leafSearches;
        grandSecs += secs;
    }

    printf("\n--- Resumo do hibrido ---\n");
    printf("tempo total                 : %.3fs\n", grandSecs);
    printf("nos MCTS expandidos         : %lld  (%.0f/s)\n",
           grandExpanded, grandExpanded / grandSecs);
    printf("buscas de folha (searchLeaf): %lld\n", grandLeaves);
    printf("nos de AB equivalentes      : %lld  (%.0f nos-AB/s)  <-- metrica comparavel\n",
           grandNodes, grandNodes / grandSecs);
    printf("nos de AB por no MCTS       : %.1f\n",
           grandExpanded > 0 ? (double)grandNodes / grandExpanded : 0.0);
    printf("pico de memoria da arvore   : %.1f KB para nodeBudget=%d\n",
           peakTreeBytes / 1024.0, nodeBudget);
    printf("\nNota (Secao 12): o numero a confrontar com os nos/seg do bloco A e\n"
           "\"nos-AB/s\", nao \"nos MCTS/s\". A diferenca entre os dois e o overhead\n"
           "real do hibrido (forward de politica por expansao, PUCT, backup,\n"
           "acumulador incremental por passo de descida).\n");
    return 0;
}
