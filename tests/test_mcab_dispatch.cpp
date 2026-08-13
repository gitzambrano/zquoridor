// test_mcab_dispatch.cpp -- Fase 2 do plano plan-hybrid-mc-ab.md: valida a
// camada de compatibilidade SFINAE (Seção 4.4) -- hasMcabSupport<...> e
// chooseMoveAuto(...) -- com duas "engines falsas" minimalistas, SEM
// envolver git worktree/dual-ref real (é um teste de tipos em C++ puro,
// conforme Seção 11).
//
// Critério de saída da Fase 2 (Seção 11):
//   1. FakeEngineOld (sem searchLeaf/resetOrderingState, simula um ref
//      anterior a este plano): chooseMoveAuto cai no fallback
//      eng.chooseMove(...) sem erro de compilação.
//   2. FakeEngineNew (com searchLeaf/resetOrderingState, Seção 3): usa o
//      caminho MCAB de verdade -- não só compila, mas escolhe o lance
//      correto por backup/score (mesmo espírito de
//      testBackupSignTrivialWin da Fase 1), provando que a sobrecarga
//      certa foi selecionada em tempo de compilação.
#include <cstdio>
#include <cassert>
#include <vector>
#include <array>
#include "../tools/common/mcab.hpp"

// =========================================================================
// "Jogo" de brinquedo minimalista -- não usa qr::/src nenhum de propósito
// (prova que mcab.hpp não depende de rules.hpp/nnue.hpp por include direto,
// só via ADL nos pontos de instanciação, Seção 4.2). 2 lances por posição:
// id==0 sempre vence pra quem joga, id==1 sempre perde -- serve pra
// verificar que o backup/score do modo equivalência (Seção 6) escolhe o
// lance certo neste tipo completamente alheio a Quoridor, não só que
// compila.
// =========================================================================
namespace faketest {

struct FakeMove {
    int id = 0;
};
using FakeMoveList = std::vector<FakeMove>;

struct FakeAccumulator {
    int dummy = 0;
};
struct FakeAccPair {
    FakeAccumulator acc[2];
};

struct FakeState {
    int wallsLeft[2] = {1, 1};  // não-vazio de propósito -- evita o atalho
                                // da Seção 5 passo 1, exercitando de
                                // verdade o dispatch dentro de
                                // chooseMoveMCAB (não só o de
                                // chooseMoveAuto).
    int turn = 0;
    int winningSide = -1;  // -1 = não-terminal (só a raiz); senão, lado
                            // vencedor desta posição (ver applyMove).
};

struct FakeRepTbl {
    void markRoot() {}
};

struct FakeSearchStats {};

inline FakeMoveList legalMoves(const FakeState&) {
    return {FakeMove{0}, FakeMove{1}};
}

inline FakeState applyMove(const FakeState& s, const FakeMove& m) {
    FakeState child;
    child.wallsLeft[0] = s.wallsLeft[0];
    child.wallsLeft[1] = s.wallsLeft[1];
    child.turn = 1 - s.turn;
    // lance id==0: quem jogou (s.turn) vence; id==1: quem jogou perde.
    child.winningSide = (m.id == 0) ? s.turn : (1 - s.turn);
    return child;
}

inline int winner(const FakeState& s) { return s.winningSide; }

inline FakeAccPair buildAccPairRoot(const FakeState&, const void*) { return FakeAccPair{}; }

inline void makeChildAccPair(FakeAccPair&, FakeAccPair&, const FakeState&, const FakeMove&,
                              const void*) {}

inline void forwardPolicyQuant(const FakeAccumulator&, std::array<float, 2>& out) {
    out.fill(0.f);
}

inline float policyLogitForMove(const std::array<float, 2>&, const FakeMove&, int) { return 0.f; }

}  // namespace faketest

// =========================================================================
// FakeEngineOld -- simula um ref anterior à Fase 0 deste plano: tem
// chooseMove() (já existia antes), mas NÃO tem searchLeaf/resetOrderingState.
// =========================================================================
struct FakeEngineOld {
    int chooseMoveCalls = 0;

    faketest::FakeMove chooseMove(const faketest::FakeState& root, int /*maxDepthCap*/,
                                   int /*timeBudgetMs*/, faketest::FakeSearchStats&,
                                   const faketest::FakeRepTbl&) {
        chooseMoveCalls++;
        // AB "puro" de brinquedo: escolhe sempre o primeiro lance legal.
        return faketest::legalMoves(root)[0];
    }
};

// =========================================================================
// FakeEngineNew -- simula a engine pós-Fase 0: tem searchLeaf E
// resetOrderingState públicos (Seção 3), além de chooseMove/clearTT/
// getEvalMode (superfície mínima que chooseMoveMCAB precisa, Seção 5).
// =========================================================================
struct FakeEngineNew {
    enum class EvalMode { Heuristic, NNUE };

    int chooseMoveCalls = 0;
    int searchLeafCalls = 0;
    int resetOrderingStateCalls = 0;

    EvalMode getEvalMode() const { return EvalMode::NNUE; }
    void clearTT() {}
    void resetOrderingState() { resetOrderingStateCalls++; }
    // Parte da superfície detectada por hasMcabSupport: MCABSearch usa isto
    // para descartar folhas truncadas pelo teto de tempo. Esta engine falsa
    // nunca estoura tempo, então é sempre false.
    bool searchWasStopped() const { return false; }

    int searchLeaf(const faketest::FakeState&, int /*depth*/, faketest::FakeSearchStats&,
                    faketest::FakeRepTbl&, faketest::FakeAccPair* /*seedAcc*/ = nullptr,
                    int /*timeBudgetMs*/ = 0) {
        // Nunca deveria ser chamado nesta suíte -- as duas posições-filha
        // de teste já nascem terminais (winner() != -1), então
        // equivalenceMove() nunca cai no ramo que chama searchLeaf.
        // Mantido só pra satisfazer a assinatura exigida por
        // MCABSearch/hasMcabSupport (Seção 3.2).
        searchLeafCalls++;
        return 0;
    }

    faketest::FakeMove chooseMove(const faketest::FakeState& root, int /*maxDepthCap*/,
                                   int /*timeBudgetMs*/, faketest::FakeSearchStats&,
                                   const faketest::FakeRepTbl&) {
        chooseMoveCalls++;
        return faketest::legalMoves(root)[0];
    }
};

using namespace faketest;

// ---------------------------------------------------------------------
// 1) hasMcabSupport<...> -- teste de tipos puro (Seção 11, Fase 2).
// ---------------------------------------------------------------------
void testHasMcabSupportTrait() {
    constexpr bool oldSupports =
        mcab::hasMcabSupport<FakeEngineOld, FakeState, FakeSearchStats, FakeRepTbl>(0);
    constexpr bool newSupports =
        mcab::hasMcabSupport<FakeEngineNew, FakeState, FakeSearchStats, FakeRepTbl>(0);
    static_assert(!oldSupports, "FakeEngineOld não deveria satisfazer hasMcabSupport (não tem searchLeaf/resetOrderingState)");
    static_assert(newSupports, "FakeEngineNew deveria satisfazer hasMcabSupport (tem searchLeaf/resetOrderingState)");
    printf("[testHasMcabSupportTrait] oldSupports=%d newSupports=%d\n", oldSupports, newSupports);
    printf("[testHasMcabSupportTrait] OK\n");
}

// ---------------------------------------------------------------------
// 2) FakeEngineOld: chooseMoveAuto cai no fallback sem erro de compilação,
//    e de fato delega pra eng.chooseMove() em runtime.
// ---------------------------------------------------------------------
void testFallbackForOldEngine() {
    FakeEngineOld eng;
    mcab::McabParams p;
    p.enabled = true;  // mesmo com enabled=true, Eng sem suporte ignora `p` (Seção 4.4)
    p.nodeBudget = 1;

    FakeState root;
    FakeSearchStats stats;
    FakeRepTbl hist;

    FakeMove chosen = mcab::chooseMoveAuto<FakeEngineOld, FakeState, FakeMove, FakeMoveList,
                                            FakeAccPair, FakeRepTbl, FakeSearchStats, 2>(
        eng, p, root, 40, 1000, stats, hist);

    printf("[testFallbackForOldEngine] chooseMoveCalls=%d lance escolhido id=%d\n",
           eng.chooseMoveCalls, chosen.id);
    assert(eng.chooseMoveCalls == 1 && "fallback deveria ter chamado eng.chooseMove() exatamente 1x");
    printf("[testFallbackForOldEngine] OK\n");
}

// ---------------------------------------------------------------------
// 3) FakeEngineNew, p.enabled=false: mesma sobrecarga "com suporte" é
//    selecionada (prova de tipo), mas o próprio chooseMoveAuto respeita
//    `enabled=false` e delega pra eng.chooseMove() sem tocar em
//    MCABSearch -- mesmo padrão do atalho já testado na Fase 1
//    (mcabParams1.enabled == false não deveria ter overhead extra,
//    Seção 10).
// ---------------------------------------------------------------------
void testSupportedEngineDisabled() {
    FakeEngineNew eng;
    mcab::McabParams p;
    p.enabled = false;
    p.nodeBudget = 1;

    FakeState root;
    FakeSearchStats stats;
    FakeRepTbl hist;

    FakeMove chosen = mcab::chooseMoveAuto<FakeEngineNew, FakeState, FakeMove, FakeMoveList,
                                            FakeAccPair, FakeRepTbl, FakeSearchStats, 2>(
        eng, p, root, 40, 1000, stats, hist);

    printf("[testSupportedEngineDisabled] chooseMoveCalls=%d resetOrderingStateCalls=%d lance id=%d\n",
           eng.chooseMoveCalls, eng.resetOrderingStateCalls, chosen.id);
    assert(eng.chooseMoveCalls == 1);
    assert(eng.resetOrderingStateCalls == 0 && "p.enabled=false não deveria ter tocado em MCABSearch");
    printf("[testSupportedEngineDisabled] OK\n");
}

// ---------------------------------------------------------------------
// 4) FakeEngineNew, p.enabled=true, nodeBudget=1 (modo equivalência,
//    Seção 6): usa o caminho MCAB de verdade -- passa por
//    resetOrderingState() e MCABSearch::chooseMoveMCAB/equivalenceMove --
//    e escolhe o lance vencedor (id==0), provando backup/score corretos
//    neste "jogo" alheio a Quoridor.
// ---------------------------------------------------------------------
void testMcabPathUsedForNewEngine() {
    FakeEngineNew eng;
    mcab::McabParams p;
    p.enabled = true;
    p.nodeBudget = 1;  // modo equivalência -- ver Seção 6

    FakeState root;  // turn=0
    FakeSearchStats stats;
    FakeRepTbl hist;

    FakeMove chosen = mcab::chooseMoveAuto<FakeEngineNew, FakeState, FakeMove, FakeMoveList,
                                            FakeAccPair, FakeRepTbl, FakeSearchStats, 2>(
        eng, p, root, 40, 1000, stats, hist);

    printf("[testMcabPathUsedForNewEngine] chooseMoveCalls=%d resetOrderingStateCalls=%d "
           "searchLeafCalls=%d lance escolhido id=%d (esperado id=0)\n",
           eng.chooseMoveCalls, eng.resetOrderingStateCalls, eng.searchLeafCalls, chosen.id);

    // Prova de que passou pelo caminho MCAB (não pelo atalho de
    // chooseMoveAuto nem pelo de "mãos vazias" da Seção 5 passo 1, já que
    // root.wallsLeft={1,1}): resetOrderingState() foi chamado (Passo 2,
    // Seção 5) e eng.chooseMove() NUNCA foi chamado (equivalenceMove não
    // delega pra chooseMove()).
    assert(eng.resetOrderingStateCalls == 1 && "deveria ter passado pelo Passo 2 de chooseMoveMCAB");
    assert(eng.chooseMoveCalls == 0 && "modo equivalência não deveria chamar eng.chooseMove()");
    assert(eng.searchLeafCalls == 0 && "ambos os filhos já nascem terminais -- searchLeaf não deveria ser chamado");
    assert(chosen.id == 0 && "MCAB não escolheu o lance vencedor -- sinal do score/backup pode estar invertido");
    printf("[testMcabPathUsedForNewEngine] OK\n");
}

int main() {
    testHasMcabSupportTrait();
    testFallbackForOldEngine();
    testSupportedEngineDisabled();
    testMcabPathUsedForNewEngine();
    printf("\nTODOS OS TESTES DE test_mcab_dispatch PASSARAM\n");
    return 0;
}
