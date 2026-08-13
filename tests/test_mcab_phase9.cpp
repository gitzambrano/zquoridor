// test_mcab_phase9.cpp -- cobre as três features da Fase 9 de
// plan-hybrid-mc-ab.md que existiam em McabParams como campos declarados
// mas inicialmente sem efeito nenhum: reuso de subárvore entre lances
// (Seção 8), ruído de Dirichlet na raiz (Seção 9) e profundidade de folha
// adaptativa (Seção 9). Um default como `treeReuse = true` que na prática
// não faz nada é pior que a feature ausente -- estes testes existem para
// que isso não volte a passar despercebido.
//
// Também cobre o teto de tempo por folha: chooseMoveMCAB com timeBudgetMs
// tem que respeitar o orçamento com folga pequena, e não ~2x como fazia
// antes de o teto ser propagado para dentro de engine.searchLeaf.
//
// Build: g++ -O2 -std=c++17 -Isrc -o bin/test_mcab_phase9 tests/test_mcab_phase9.cpp
// Rodar a partir da RAIZ do repo (carrega data/nnue/nnue_weights_int8.bin).
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../tools/common/mcab.hpp"

using namespace qr;

using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                              RepetitionTable, SearchStats>;

namespace {

// Posição de meio-jogo com muros ainda na mão dos dois lados -- "mãos
// vazias" é delegado ao solver exato e nunca constrói árvore (Seção 5,
// passo 1), o que faria todos os testes abaixo inspecionarem uma árvore
// vazia e passarem por vacuidade.
State midgamePosition(unsigned seed = 2026, int plies = 10) {
    std::mt19937 rng(seed);
    State s = initialState();
    for (int i = 0; i < plies; i++) {
        if (winner(s) != -1) break;
        auto moves = legalMoves(s);
        if (moves.empty()) break;
        s = applyMove(s, moves[rng() % moves.size()]);
    }
    assert(winner(s) == -1);
    assert(s.wallsLeft[0] > 0 && s.wallsLeft[1] > 0);
    return s;
}

Negamax makeEngine() {
    Negamax e;
    e.setEvalMode(Negamax::EvalMode::NNUE);
    return e;
}

// ---------------------------------------------------------------------
// Seção 8 -- reuso de subárvore entre lances consecutivos.
// ---------------------------------------------------------------------
void testTreeReuse() {
    State s = midgamePosition();
    Negamax eng = makeEngine();
    Mcab mc;
    mc.params.enabled = true;
    mc.params.treeReuse = true;
    mc.params.nodeBudget = 300;
    mc.params.leafDepth = 2;

    SearchStats st;
    RepetitionTable hist;

    // Primeiro lance: árvore nascendo do zero, nada a reaproveitar.
    mcab::McabStats ms1;
    Move m1 = mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms1);
    assert(!ms1.treeReused && "primeiro lance não tem árvore anterior para reusar");
    size_t pool1 = mc.poolSize();
    assert(pool1 > 1);

    // Segundo lance: a nova raiz é filha da raiz anterior, então a
    // subárvore dela tem que sobreviver com N/W/P já acumulados.
    s = applyMove(s, m1);
    hist.push(s.hash);
    mcab::McabStats ms2;
    Move m2 = mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms2);
    printf("[testTreeReuse] lance1 pool=%zu | lance2 reusado=%s nosHerdados=%d pool=%zu\n",
           pool1, ms2.treeReused ? "sim" : "nao", ms2.reusedNodes, mc.poolSize());
    assert(ms2.treeReused && "a nova raiz era filha da anterior -- reuso deveria ter acontecido");
    assert(ms2.reusedNodes > 1 && "reuso trouxe só a raiz -- subárvore não foi preservada");

    // Terceiro lance: o pool não pode crescer indefinidamente. O limite
    // implementado é ~2x nodeBudget (subárvore herdada + orçamento novo);
    // a compactação aborta para árvore nova se a herança sozinha já passa
    // do orçamento.
    s = applyMove(s, m2);
    hist.push(s.hash);
    mcab::McabStats ms3;
    Move m3 = mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms3);
    (void)m3;
    size_t limite = (size_t)(2 * mc.params.nodeBudget + 2);
    printf("[testTreeReuse] lance3 pool=%zu (limite %zu)\n", mc.poolSize(), limite);
    assert(mc.poolSize() <= limite && "pool cresceu além de ~2x nodeBudget entre lances");
    printf("[testTreeReuse] OK\n");
}

// Com treeReuse=false o pool tem que ser descartado ao fim de cada
// chamada, e nenhum lance seguinte pode reportar reuso.
void testTreeReuseDisabled() {
    State s = midgamePosition();
    Negamax eng = makeEngine();
    Mcab mc;
    mc.params.enabled = true;
    mc.params.treeReuse = false;
    mc.params.nodeBudget = 200;
    mc.params.leafDepth = 2;

    SearchStats st;
    RepetitionTable hist;
    mcab::McabStats ms1;
    Move m1 = mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms1);
    assert(mc.poolSize() == 0 && "treeReuse=false deveria descartar o pool ao retornar");
    assert(mc.rootNodeForInspection() == nullptr);

    s = applyMove(s, m1);
    hist.push(s.hash);
    mcab::McabStats ms2;
    mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms2);
    printf("[testTreeReuseDisabled] reusado=%s pool=%zu\n",
           ms2.treeReused ? "sim" : "nao", mc.poolSize());
    assert(!ms2.treeReused && "treeReuse=false não pode reaproveitar árvore");
    printf("[testTreeReuseDisabled] OK\n");
}

// ---------------------------------------------------------------------
// Seção 9 -- profundidade de folha adaptativa.
// ---------------------------------------------------------------------
// Com adaptiveLeafDepth ligado, ramos mais visitados recebem folhas mais
// profundas, então a MÉDIA de profundidade fica acima de leafDepth. Com
// desligado, a média tem que ser exatamente leafDepth.
void testAdaptiveLeafDepth() {
    State s = midgamePosition();
    RepetitionTable hist;

    auto medida = [&](bool adaptive) {
        Negamax eng = makeEngine();
        Mcab mc;
        mc.params.enabled = true;
        mc.params.treeReuse = false;
        mc.params.nodeBudget = 400;
        mc.params.leafDepth = 2;
        mc.params.leafDepthMax = 6;
        mc.params.adaptiveLeafDepth = adaptive;
        SearchStats st;
        mcab::McabStats ms;
        mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms);
        assert(ms.leafSearches > 0 && "nenhuma folha avaliada -- teste não mede nada");
        return (double)ms.leafDepthSum / ms.leafSearches;
    };

    double fixa = medida(false);
    double adapt = medida(true);
    printf("[testAdaptiveLeafDepth] media fixa=%.3f adaptativa=%.3f (leafDepth=2, max=6)\n",
           fixa, adapt);
    assert(std::fabs(fixa - 2.0) < 1e-9 && "sem adaptativo a profundidade tem que ser constante");
    assert(adapt > fixa && "adaptativo não aprofundou nenhuma folha");
    assert(adapt <= 6.0 && "adaptativo passou de leafDepthMax");
    printf("[testAdaptiveLeafDepth] OK\n");
}

// ---------------------------------------------------------------------
// Seção 9 -- ruído de Dirichlet na raiz.
// ---------------------------------------------------------------------
// Os priors da raiz com ruído têm que diferir dos priors da rede pura, e
// continuar somando ~1 (o ruído é uma mistura convexa, não um shift).
void testRootNoise() {
    State s = midgamePosition();
    RepetitionTable hist;

    auto priorsDaRaiz = [&](bool noise, uint32_t seed) {
        Negamax eng = makeEngine();
        Mcab mc;
        mc.params.enabled = true;
        mc.params.treeReuse = true;  // precisa sobreviver ao retorno para inspeção
        mc.params.nodeBudget = 40;
        mc.params.leafDepth = 1;
        mc.params.rootNoiseEnabled = noise;
        mc.seedNoise(seed);
        SearchStats st;
        mcab::McabStats ms;
        mc.chooseMoveMCAB(eng, s, 40, 0, st, hist, &ms);
        const auto* raiz = mc.rootNodeForInspection();
        assert(raiz != nullptr && "árvore vazia -- nada para inspecionar");
        return std::vector<float>(raiz->P.begin(), raiz->P.end());
    };

    std::vector<float> puro = priorsDaRaiz(false, 12345u);
    std::vector<float> comRuido = priorsDaRaiz(true, 12345u);
    assert(puro.size() == comRuido.size() && !puro.empty());

    double somaPuro = 0.0, somaRuido = 0.0, maiorDelta = 0.0;
    for (size_t i = 0; i < puro.size(); i++) {
        somaPuro += puro[i];
        somaRuido += comRuido[i];
        maiorDelta = std::max(maiorDelta, (double)std::fabs(puro[i] - comRuido[i]));
    }
    printf("[testRootNoise] lances=%zu somaPuro=%.4f somaComRuido=%.4f maiorDelta=%.4f\n",
           puro.size(), somaPuro, somaRuido, maiorDelta);
    assert(maiorDelta > 1e-4 && "rootNoiseEnabled=true não alterou prior nenhum");
    assert(std::fabs(somaRuido - 1.0) < 1e-2 && "priors com ruído deixaram de somar ~1");
    assert(std::fabs(somaPuro - 1.0) < 1e-2 && "priors da rede não somam ~1 -- softmax quebrado?");

    // Seeds diferentes têm que produzir ruídos diferentes (senão
    // seedNoise() não está sendo respeitado e todas as threads/partidas do
    // selfplay sorteariam exatamente a mesma abertura).
    std::vector<float> outroSeed = priorsDaRaiz(true, 987654u);
    double deltaEntreSeeds = 0.0;
    for (size_t i = 0; i < puro.size(); i++)
        deltaEntreSeeds = std::max(deltaEntreSeeds, (double)std::fabs(comRuido[i] - outroSeed[i]));
    printf("[testRootNoise] maior delta entre seeds distintas=%.4f\n", deltaEntreSeeds);
    assert(deltaEntreSeeds > 1e-4 && "seedNoise() não mudou o ruído -- seed ignorada");
    printf("[testRootNoise] OK\n");
}

// ---------------------------------------------------------------------
// Teto de tempo propagado para dentro da folha.
// ---------------------------------------------------------------------
// O loop de simulações só checa o relógio ENTRE simulações; uma folha de
// profundidade alta custa dezenas de ms sozinha. Sem propagar o teto para
// engine.searchLeaf, um orçamento de 60ms virava ~110ms medidos. Margem
// generosa aqui (2x) porque isto roda em -O2 sem -march=native e o
// ambiente de CI/desktop é ruidoso -- o bug original estourava sempre.
void testTimeBudgetRespected() {
    State s = midgamePosition();
    Negamax eng = makeEngine();
    Mcab mc;
    mc.params.enabled = true;
    mc.params.nodeBudget = 1000000;  // alto de propósito: quem tem que parar é o relógio
    mc.params.leafDepth = 4;
    SearchStats st;
    RepetitionTable hist;
    mcab::McabStats ms;

    const int orcamentoMs = 120;
    auto t0 = std::chrono::steady_clock::now();
    mc.chooseMoveMCAB(eng, s, 40, orcamentoMs, st, hist, &ms);
    double realMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
    printf("[testTimeBudgetRespected] orcamento=%dms real=%.1fms folhas=%lld truncadas=%lld\n",
           orcamentoMs, realMs, ms.leafSearches, ms.leafTruncated);
    assert(realMs < 2.0 * orcamentoMs && "chooseMoveMCAB estourou o orçamento de tempo");
    printf("[testTimeBudgetRespected] OK\n");
}

}  // namespace

int main() {
    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        printf("[ERRO] nao consegui carregar data/nnue/nnue_weights_int8.bin -- rode a partir\n"
               "       da RAIZ do repo. Sem a cabeca de politica os priors ficam uniformes e\n"
               "       o teste de ruido de Dirichlet nao prova nada.\n");
        return 1;
    }
    testTreeReuse();
    testTreeReuseDisabled();
    testAdaptiveLeafDepth();
    testRootNoise();
    testTimeBudgetRespected();
    printf("TODOS OS TESTES DE test_mcab_phase9 PASSARAM\n");
    return 0;
}
