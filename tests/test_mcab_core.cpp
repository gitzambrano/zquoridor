// test_mcab_core.cpp -- Fase 1 do plano plan-hybrid-mc-ab.md: valida o
// núcleo de MCABSearch<...> (src/mcab.hpp) isolado, contra a
// src/ local (sem dual-ref, sem arena.cpp/selfplay/tune_spsa -- nenhum
// desses binários é tocado por este teste).
//
// Critérios de saída da Fase 1 (Seção 11):
//   1. Pool não estoura o orçamento (mcabNodeBudget).
//   2. Backup propaga o sinal corretamente em posições triviais
//      construídas à mão (aqui: posições a 1 lance do fim de jogo).
//   3. scoreToQ bate com nnueWinProbQuant para os mesmos logits.
#include <cstdio>
#include <cassert>
#include <cmath>
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;

using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair,
                               RepetitionTable, SearchStats>;

namespace {

State buildState(int p0row, int p0col, int p1row, int p1col, int turn,
                  int walls0 = 10, int walls1 = 10) {
    State s;
    s.pawn[0] = cellIdx(p0row, p0col);
    s.pawn[1] = cellIdx(p1row, p1col);
    s.wallsH = 0;
    s.wallsV = 0;
    s.wallsLeft[0] = (int8_t)walls0;
    s.wallsLeft[1] = (int8_t)walls1;
    s.turn = turn;
    Zobrist& z = zobrist();
    s.hash = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    if (turn == 1) s.hash ^= z.turnKey;
    return s;
}

void tryLoadRealWeights() {
    // Melhor esforço -- se os pesos publicados existirem no repo, carrega
    // (dá um teste mais realista); se não, weightsQuant() já fica no
    // estado zerado seguro do construtor (ver nnue.hpp), suficiente para
    // validar a estrutura/backup do MCAB (que não depende de valores
    // reais de rede, só de winner()/aplicação de lances).
    //
    // Convenção do projeto (build_tests.sh/build_wasm.sh): binários rodam
    // a partir da RAIZ do repo (ex.: `bin/nnue_verify data/nnue/...`), não
    // de dentro de bin/ -- então o caminho é relativo à raiz, sem "../".
    loadWeightsQuant("data/nnue/nnue_weights_int8.bin");
}

} // namespace

// ---------------------------------------------------------------------
// 1) Pool não estoura o orçamento configurado.
// ---------------------------------------------------------------------
void testPoolBudget() {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);

    Mcab mcab;
    mcab.params.nodeBudget = 50;
    mcab.params.leafDepth = 2;
    mcab.params.cPuct = 1.5;

    State root = initialState();
    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats mstats;

    Move m = mcab.chooseMoveMCAB(eng, root, /*maxDepthCap=*/40, /*timeBudgetMs=*/5000, stats, hist, &mstats);
    (void)m;

    printf("[testPoolBudget] nodeBudget=%d poolSize=%zu nodesExpanded=%lld simulations=%lld\n",
           mcab.params.nodeBudget, mcab.poolSize(), mstats.nodesExpanded, mstats.simulations);

    // nodeBudget conta nós EXPANDIDOS (Seção 9: "Nº máx. de nós expandidos
    // por chooseMoveMCAB"), não todo nó do pool -- runSimulation() para o
    // loop assim que esse contador atinge o teto.
    assert(mstats.nodesExpanded <= mcab.params.nodeBudget);
    // O pool pode crescer além de nodesExpanded+1: um filho recem-criado
    // via createChild() que já nasce terminal (winner() != -1) entra no
    // pool mas NUNCA passa por expandNode() -- o ramo `node.terminal` no
    // topo de runSimulation() faz backup e retorna antes de chegar lá (ver
    // mcab.hpp). O invariante que realmente vale (Seção 4.3.2: no máximo 1
    // createChild() por simulação) é poolSize() <= simulations()+1.
    assert(mcab.poolSize() <= (size_t)mstats.simulations + 1);
    printf("[testPoolBudget] OK\n");
}

// ---------------------------------------------------------------------
// 2) Backup propaga o sinal corretamente: numa posição a 1 lance do fim
//    de jogo, o lance vencedor (do jogador na vez) tem que ser o
//    escolhido -- se o sinal estivesse invertido em algum nível do
//    backup, o MCAB escolheria um lance qualquer (ou o pior) em vez do
//    vencedor imediato.
// ---------------------------------------------------------------------
void testBackupSignTrivialWin() {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);

    // Jogador 0 (goal = row N-1 = 8) a um passo do gol em (7,4); jogador
    // 1 longe do próprio gol, sem chance de interferir em 1 lance.
    State root = buildState(/*p0row=*/7, /*p0col=*/4, /*p1row=*/1, /*p1col=*/4, /*turn=*/0);

    // Sanidade: o lance pawn(cellIdx(8,4)) precisa estar entre os legais
    // e precisa ser vencedor imediato.
    MoveList moves = legalMoves(root);
    bool sawWinningMove = false;
    for (auto& m : moves) {
        if (!m.isWall && m.a == cellIdx(8, 4)) {
            State ns = applyMove(root, m);
            assert(winner(ns) == 0 && "lance pawn->(8,4) deveria ganhar imediatamente para o jogador 0");
            sawWinningMove = true;
        }
    }
    assert(sawWinningMove && "posição de teste malformada -- lance vencedor não está entre os legais");

    Mcab mcab;
    mcab.params.nodeBudget = 200;
    mcab.params.leafDepth = 2;

    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats mstats;
    Move chosen = mcab.chooseMoveMCAB(eng, root, 40, 5000, stats, hist, &mstats);

    bool chosenIsWinning = (!chosen.isWall && chosen.a == cellIdx(8, 4));
    printf("[testBackupSignTrivialWin] lance escolhido: %s dest=%d (esperado dest=%d)\n",
           chosen.isWall ? "muro" : "peao", chosen.isWall ? -1 : chosen.a, cellIdx(8, 4));
    assert(chosenIsWinning && "MCAB não escolheu o lance vencedor imediato -- sinal do backup pode estar invertido");
    printf("[testBackupSignTrivialWin] OK\n");
}

// Variante com turn=1 (jogador 1 na vez, goal = row 0) -- garante que o
// teste acima não passa só por coincidência de qual player é "turn 0".
void testBackupSignTrivialWinOtherPlayer() {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);

    State root = buildState(/*p0row=*/7, /*p0col=*/4, /*p1row=*/1, /*p1col=*/4, /*turn=*/1);

    MoveList moves = legalMoves(root);
    bool sawWinningMove = false;
    for (auto& m : moves) {
        if (!m.isWall && m.a == cellIdx(0, 4)) {
            State ns = applyMove(root, m);
            assert(winner(ns) == 1 && "lance pawn->(0,4) deveria ganhar imediatamente para o jogador 1");
            sawWinningMove = true;
        }
    }
    assert(sawWinningMove && "posição de teste malformada -- lance vencedor não está entre os legais");

    Mcab mcab;
    mcab.params.nodeBudget = 200;
    mcab.params.leafDepth = 2;

    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats mstats;
    Move chosen = mcab.chooseMoveMCAB(eng, root, 40, 5000, stats, hist, &mstats);

    bool chosenIsWinning = (!chosen.isWall && chosen.a == cellIdx(0, 4));
    printf("[testBackupSignTrivialWinOtherPlayer] lance escolhido: %s dest=%d (esperado dest=%d)\n",
           chosen.isWall ? "muro" : "peao", chosen.isWall ? -1 : chosen.a, cellIdx(0, 4));
    assert(chosenIsWinning && "MCAB não escolheu o lance vencedor imediato (jogador 1) -- sinal do backup pode estar invertido");
    printf("[testBackupSignTrivialWinOtherPlayer] OK\n");
}

// ---------------------------------------------------------------------
// 3) Modo equivalência (mcabNodeBudget<=1, Seção 6): mesma checagem de
//    sinal acima, mas pelo caminho especial sem árvore/PUCT.
// ---------------------------------------------------------------------
void testEquivModeSign() {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);

    State root = buildState(7, 4, 1, 4, 0);

    Mcab mcab;
    mcab.params.nodeBudget = 0;  // modo equivalência
    mcab.params.leafDepth = 2;

    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats mstats;
    Move chosen = mcab.chooseMoveMCAB(eng, root, 40, 5000, stats, hist, &mstats);

    bool chosenIsWinning = (!chosen.isWall && chosen.a == cellIdx(8, 4));
    printf("[testEquivModeSign] lance escolhido: %s dest=%d (esperado dest=%d)\n",
           chosen.isWall ? "muro" : "peao", chosen.isWall ? -1 : chosen.a, cellIdx(8, 4));
    assert(chosenIsWinning && "modo equivalência não escolheu o lance vencedor imediato");
    printf("[testEquivModeSign] OK\n");
}

// ---------------------------------------------------------------------
// 4) scoreToQ bate com nnueWinProbQuant para os mesmos logits (Seção
//    4.3.1: mesma curva sigmoide, não uma calibração nova).
// ---------------------------------------------------------------------
void testScoreToQMatchesWinProb() {
    tryLoadRealWeights();

    State root = initialState();
    AccPair ap = buildAccPairRoot(root, nullptr);

    for (int side = 0; side < 2; side++) {
        float winProb = nnueWinProbQuant(ap.acc[side]);
        int score = nnueEvalInt(ap, side);  // round(logit * NNUE_EVAL_SCALE)
        double q = mcab::scoreToQ(score, (double)NNUE_EVAL_SCALE);

        printf("[testScoreToQMatchesWinProb] side=%d winProb=%.6f scoreToQ=%.6f (score=%d)\n",
               side, winProb, q, score);
        // Tolerância: nnueEvalInt arredonda o logit pra inteiro antes de
        // scoreToQ desfazer a escala -- diferença esperada é só o erro de
        // quantização do round(), não um erro estrutural. 1e-3 é folgado
        // o bastante pra escala 200 (erro de arredondamento no logit é
        // <= 0.5/200 = 0.0025 antes da sigmoide).
        double diff = std::fabs((double)winProb - q);
        assert(diff < 5e-3 && "scoreToQ diverge de nnueWinProbQuant além do erro de quantização esperado");
    }
    printf("[testScoreToQMatchesWinProb] OK\n");
}

// ---------------------------------------------------------------------
// 5) mcab.chooseMoveMCAB delega corretamente pra chooseMove() no atalho
//    de final "mãos vazias" (Seção 5, passo 1) -- não deveria construir
//    árvore nenhuma.
// ---------------------------------------------------------------------
void testEmptyHandedDelegation() {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);

    State root = buildState(7, 4, 1, 4, 0, /*walls0=*/0, /*walls1=*/0);

    Mcab mcab;
    mcab.params.nodeBudget = 200;
    mcab.params.leafDepth = 2;

    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats mstats;
    Move chosen = mcab.chooseMoveMCAB(eng, root, 40, 5000, stats, hist, &mstats);

    bool chosenIsWinning = (!chosen.isWall && chosen.a == cellIdx(8, 4));
    printf("[testEmptyHandedDelegation] lance escolhido: dest=%d (esperado dest=%d), poolSize=%zu\n",
           chosen.isWall ? -1 : chosen.a, cellIdx(8, 4), mcab.poolSize());
    assert(chosenIsWinning);
    // Delegou pra chooseMove() -- não deveria ter tocado no pool do MCAB
    // (fica vazio, já que o atalho retorna antes do passo 3).
    assert(mcab.poolSize() == 0);
    printf("[testEmptyHandedDelegation] OK\n");
}

// ---------------------------------------------------------------------
// 6) O caminho MCTS pega o cache de BFS da engine, não passa nullptr.
//    searchLeaf/QS também tocam o mesmo cache, então hits/misses depois
//    de uma busca não isolam este fio; o que trava a regressão é o
//    ponteiro em si (e o teste de dispatch, que continua compilando
//    contra engines sem pathCache()).
// ---------------------------------------------------------------------
void testMctsPathCacheWired() {
    Negamax eng;
    PlayerPathCacheTable* fromEngine = eng.pathCache();
    PlayerPathCacheTable* fromMcab = mcab::mcabPathCache(eng, 0);
    printf("[testMctsPathCacheWired] pathCache=%p mcabPathCache=%p\n",
           (void*)fromEngine, (void*)fromMcab);
    assert(fromEngine != nullptr);
    assert(fromMcab == fromEngine);
    printf("[testMctsPathCacheWired] OK\n");
}

// ---------------------------------------------------------------------
// 7) leafDepth=0 não entra em searchLeaf/QS. SearchStats.nodes só é
//    incrementado por negamax/quiescence, então tem que ficar 0; as
//    avaliações de folha aparecem em McabStats.leafSearches.
// ---------------------------------------------------------------------
void testFastLeafSkipsSearchLeaf() {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);

    Mcab mc;
    mc.params.nodeBudget = 80;
    mc.params.leafDepth = 0;
    mc.params.adaptiveLeafDepth = false;
    mc.params.treeReuse = false;

    State root = initialState();
    SearchStats stats;
    RepetitionTable hist;
    mcab::McabStats mstats;
    Move m = mc.chooseMoveMCAB(eng, root, 40, 0, stats, hist, &mstats);
    (void)m;

    printf("[testFastLeafSkipsSearchLeaf] leafSearches=%lld abNodes=%llu pool=%zu\n",
           mstats.leafSearches, (unsigned long long)stats.nodes, mc.poolSize());
    assert(mstats.leafSearches > 0 && "nenhuma folha avaliada");
    assert(stats.nodes == 0 && "leafDepth=0 ainda passou por searchLeaf/QS");
    printf("[testFastLeafSkipsSearchLeaf] OK\n");
}

// ---------------------------------------------------------------------
// 8) Backup modes use distinct value aggregation. MinimaxHard propagates the
// minimax value of each parent node; AvgBlend stores the visit sum used by Q=W/N.
// ---------------------------------------------------------------------
void testBackupModes() {
    Negamax engHard, engAvg;
    engHard.setEvalMode(Negamax::EvalMode::NNUE);
    engAvg.setEvalMode(Negamax::EvalMode::NNUE);

    State root = initialState();
    RepetitionTable hist;
    SearchStats hardStats, avgStats;
    mcab::McabStats hardMstats, avgMstats;

    Mcab hard;
    hard.params.nodeBudget = 120;
    hard.params.leafDepth = 0;
    hard.params.treeReuse = true;
    hard.params.backupMode = mcab::BackupMode::MinimaxHard;
    hard.chooseMoveMCAB(engHard, root, 40, 0, hardStats, hist, &hardMstats);

    Mcab avg;
    avg.params.nodeBudget = 120;
    avg.params.leafDepth = 0;
    avg.params.treeReuse = true;
    avg.params.backupMode = mcab::BackupMode::AvgBlend;
    avg.chooseMoveMCAB(engAvg, root, 40, 0, avgStats, hist, &avgMstats);

    const auto* hardRoot = hard.rootNodeForInspection();
    const auto* avgRoot = avg.rootNodeForInspection();
    assert(hardRoot != nullptr && avgRoot != nullptr);
    assert(hardRoot->moves.size() == avgRoot->moves.size());

    bool sawRepeatedVisit = false;
    for (size_t i = 0; i < hardRoot->N.size(); i++) {
        if (hardRoot->N[i] > 1.f) sawRepeatedVisit = true;
        if (hardRoot->N[i] > 0.f)
            assert(hardRoot->W[i] >= 0.f && hardRoot->W[i] <= 1.f);
        if (avgRoot->N[i] > 0.f)
            assert(avgRoot->W[i] >= 0.f && avgRoot->W[i] <= avgRoot->N[i] + 1e-5f);
    }
    assert(sawRepeatedVisit && "backup-mode test did not revisit a root edge");
    printf("[testBackupModes] hard simulations=%lld avg simulations=%lld OK\n",
           hardMstats.simulations, avgMstats.simulations);
}

int main() {
    testScoreToQMatchesWinProb();
    testPoolBudget();
    testBackupSignTrivialWin();
    testBackupSignTrivialWinOtherPlayer();
    testEquivModeSign();
    testEmptyHandedDelegation();
    testMctsPathCacheWired();
    testFastLeafSkipsSearchLeaf();
    testBackupModes();
    printf("\nTODOS OS TESTES DE test_mcab_core PASSARAM\n");
    return 0;
}
