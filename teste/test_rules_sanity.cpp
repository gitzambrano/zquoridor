#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>
#include "rules.hpp"
using namespace qr;

// Implementação de referência de legalWallMoves: sempre roda hasPathToGoal
// (BFS completo) duas vezes por slot candidato, sem o pré-filtro da Fase
// 4.2.1. É exatamente a lógica antiga de rules.hpp antes da otimização.
// Usada só neste teste, como oráculo pra checar que o pré-filtro não
// mudou o conjunto de lances legais gerado (sem regressão).
static void legalWallMovesReference(const State& s, int player, std::vector<Move>& out) {
    if (s.wallsLeft[player] <= 0) return;
    for (int orientation = 0; orientation < 2; orientation++) {
        for (int r = 0; r < WS; r++) {
            for (int c = 0; c < WS; c++) {
                if (!wallSlotAvailable(s.wallsH, s.wallsV, orientation, r, c)) continue;
                uint64_t nh = s.wallsH, nv = s.wallsV;
                if (orientation == 0) nh |= (1ull << slotIdx(r, c));
                else nv |= (1ull << slotIdx(r, c));
                if (!hasPathToGoal(nh, nv, s.pawn[0], 0)) continue;
                if (!hasPathToGoal(nh, nv, s.pawn[1], 1)) continue;
                out.push_back(Move::wall(orientation, r, c));
            }
        }
    }
}

static bool sameMoveSet(std::vector<Move> a, std::vector<Move> b) {
    auto cmp = [](const Move& x, const Move& y) {
        if (x.isWall != y.isWall) return x.isWall < y.isWall;
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        return x.c < y.c;
    };
    std::sort(a.begin(), a.end(), cmp);
    std::sort(b.begin(), b.end(), cmp);
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
        [](const Move& x, const Move& y) { return x == y; });
}

// Regressão do pré-filtro + union-find (Fase 4.2.1 completa): gera muitas
// posições via partidas aleatórias (peão e muro, sem viés) e compara, em
// CADA posição e para os dois jogadores, legalWallMoves (otimizado, com
// pré-filtro + DSU de rollback sobre o grafo dual de muros -- ver
// dsu.hpp) contra legalWallMovesReference (sempre BFS duplo, sem
// otimização nenhuma). Precisam bater exatamente -- não é uma checagem
// estatística. Esta suíte foi o que pegou, na primeira versão do DSU, um
// caso de "bolso de canto" (peão preso contra duas bordas
// perpendiculares sem barreira ponta-a-ponta) que a checagem inicial
// (só esquerda-direita) deixava passar como falso-legal -- ver a prova
// completa e a correção em dsu.hpp.
static bool testWallPrefilterRegression() {
    std::mt19937 rng(12345);
    int statesChecked = 0;
    for (int game = 0; game < 300; game++) {
        State s = initialState();
        for (int ply = 0; ply < 60; ply++) {
            for (int player = 0; player < 2; player++) {
                MoveList got;
                std::vector<Move> ref;
                legalWallMoves(s, player, got);
                legalWallMovesReference(s, player, ref);
                statesChecked++;
                if (!sameMoveSet(got.toVector(), ref)) {
                    printf("FALHOU: divergencia no pre-filtro (jogo %d, ply %d, player %d): "
                           "otimizado=%d lances, referencia=%d lances\n",
                           game, ply, player, (int)got.size(), (int)ref.size());
                    return false;
                }
            }
            auto moves = legalMoves(s);
            if (winner(s) != -1 || moves.empty()) break;
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            s = applyMove(s, moves[dist(rng)]);
        }
    }
    printf("pre-filtro+DSU de muro: %d estados x 2 jogadores checados, 0 divergencias\n", statesChecked);
    return true;
}

// Segunda camada de regressão, com viés forte para lances de muro (85%) partidas mais longas (até 200 plies): posições assim ficam
// bem mais densas de muros (perto do limite de 10+10), o que aumenta
// muito a chance de ciclos e bolsos de canto no grafo dual -- exatamente
// o regime que mais estressa o DSU (o pré-filtro sozinho já cobria bem
// o caso esparso). Mesmo oráculo e mesma checagem exata de
// testWallPrefilterRegression, só a distribuição de lances muda.
static bool testWallDsuCornerPocketRegression() {
    std::mt19937 rng(20260718);
    long statesChecked = 0;
    for (int game = 0; game < 400; game++) {
        std::mt19937 gameRng(20260718u + (unsigned)game);
        State s = initialState();
        for (int ply = 0; ply < 200; ply++) {
            for (int player = 0; player < 2; player++) {
                MoveList got;
                std::vector<Move> ref;
                legalWallMoves(s, player, got);
                legalWallMovesReference(s, player, ref);
                statesChecked++;
                if (!sameMoveSet(got.toVector(), ref)) {
                    printf("FALHOU: divergencia no DSU com vies de muro (jogo %d, ply %d, player %d): "
                           "otimizado=%d lances, referencia=%d lances "
                           "(wallsH=%016llx wallsV=%016llx pawn0=%d pawn1=%d)\n",
                           game, ply, player, (int)got.size(), (int)ref.size(),
                           (unsigned long long)s.wallsH, (unsigned long long)s.wallsV, s.pawn[0], s.pawn[1]);
                    return false;
                }
            }
            auto moves = legalMoves(s);
            if (winner(s) != -1 || moves.empty()) break;
            std::vector<Move> wallMoves, pawnMoves;
            for (auto& m : moves) (m.isWall ? wallMoves : pawnMoves).push_back(m);
            std::uniform_real_distribution<double> coin(0.0, 1.0);
            Move chosen;
            if (!wallMoves.empty() && coin(gameRng) < 0.85) {
                std::uniform_int_distribution<size_t> dist(0, wallMoves.size() - 1);
                chosen = wallMoves[dist(gameRng)];
            } else {
                std::uniform_int_distribution<size_t> dist(0, pawnMoves.size() - 1);
                chosen = pawnMoves[dist(gameRng)];
            }
            s = applyMove(s, chosen);
        }
    }
    printf("DSU com vies de muro (posicoes densas): %ld estados x 2 jogadores checados, 0 divergencias\n", statesChecked);
    return true;
}

// Testes unitários diretos do RollbackDSU (dsu.hpp), isolados de
// rules.hpp/geometria de muro: cobrem union-by-size, no-op em união
// redundante (mesmo componente), e que rollbackTo() restaura find()/sz
// exatamente ao estado anterior -- a garantia da qual buildWallDSU e
// wallCandidateAmbiguous dependem a cada chamada (push+teste+rollback
// no mesmo dsu, milhares de vezes por partida).
static bool testDsuUnitCore() {
    RollbackDSU dsu;
    dsu.init(10);
    for (int i = 0; i < 10; i++) if (dsu.find(i) != i) { printf("FALHOU: dsu init\n"); return false; }

    int snap0 = dsu.snapshot();
    bool redundant = dsu.unite(0, 1);
    if (redundant) { printf("FALHOU: unite(0,1) nao deveria ser redundante\n"); return false; }
    if (dsu.find(0) != dsu.find(1)) { printf("FALHOU: 0 e 1 deveriam estar unidos\n"); return false; }

    redundant = dsu.unite(1, 0);
    if (!redundant) { printf("FALHOU: unite(1,0) deveria ser redundante (mesmo componente)\n"); return false; }

    dsu.unite(2, 3);
    dsu.unite(0, 2); // une {0,1} com {2,3}
    if (!(dsu.find(1) == dsu.find(3))) { printf("FALHOU: transitividade 1-3\n"); return false; }
    if (dsu.find(4) == dsu.find(1)) { printf("FALHOU: 4 nao deveria estar unido a 1\n"); return false; }

    dsu.rollbackTo(snap0);
    for (int i = 0; i < 10; i++) if (dsu.find(i) != i) {
        printf("FALHOU: rollbackTo nao restaurou estado inicial (find(%d)=%d)\n", i, dsu.find(i));
        return false;
    }
    printf("dsu unit core: unite/find/rollback OK\n");
    return true;
}

// Testes unitários de wallCandidateAmbiguous nos cantos/span (condições
// (b) e (c) da nota de corretude em dsu.hpp), verificando também que a
// chamada é idempotente (prova indireta de que rollbackTo() restaurou o
// dsu por completo -- rodar a mesma checagem duas vezes tem que dar o
// mesmo resultado).
static bool testDsuCornerUnitCases() {
    struct Case { const char* name; int orient; int r, c; std::vector<std::pair<int,int>> priorH; std::vector<std::pair<int,int>> priorV; bool expectAmbiguous; };
    std::vector<Case> cases = {
        {"top-left corner",  1, 0, 0, {{0,0}}, {}, true},
        {"top-right corner", 1, 0, WS-1, {{1, WS-1}}, {}, true},
        {"left-right span",  0, 0, 7, {{0,0},{0,1},{0,3},{0,5}}, {{0,1}}, true},
    };
    for (auto& tc : cases) {
        RollbackDSU dsu;
        uint64_t wallsH = 0, wallsV = 0;
        for (auto& p : tc.priorH) wallsH |= (1ull << slotIdx(p.first, p.second));
        for (auto& p : tc.priorV) wallsV |= (1ull << slotIdx(p.first, p.second));
        buildWallDSU(dsu, wallsH, wallsV);
        bool ambiguous = wallCandidateAmbiguous(dsu, wallsH, wallsV, tc.orient, tc.r, tc.c);
        bool ambiguous2 = wallCandidateAmbiguous(dsu, wallsH, wallsV, tc.orient, tc.r, tc.c);
        if (ambiguous != ambiguous2) {
            printf("FALHOU: %s -- wallCandidateAmbiguous nao idempotente (rollback incompleto?)\n", tc.name);
            return false;
        }
        if (ambiguous != tc.expectAmbiguous) {
            printf("FALHOU: %s -- esperado ambiguous=%d, obtido %d\n", tc.name, tc.expectAmbiguous, ambiguous);
            return false;
        }
    }
    printf("dsu casos de canto/span (top-left, top-right, left-right): %d/%d OK\n", (int)cases.size(), (int)cases.size());
    return true;
}

int main() {
    if (!testDsuUnitCore()) return 1;
    if (!testDsuCornerUnitCases()) return 1;
    if (!testWallPrefilterRegression()) return 1;
    if (!testWallDsuCornerPocketRegression()) return 1;

    State s = initialState();
    auto moves = legalMoves(s);
    int pawnCount = 0, wallCount = 0;
    for (auto& m : moves) (m.isWall ? wallCount : pawnCount)++;
    printf("lances iniciais: %d peao + %d muro = %d total\n", pawnCount, wallCount, (int)moves.size());
    if (pawnCount != 3 || wallCount != 128) { printf("FALHOU: esperado 3 peao + 128 muro\n"); return 1; }

    // salto reto
    State s2 = s;
    s2.pawn[0] = cellIdx(4, 4);
    s2.pawn[1] = cellIdx(3, 4);
    auto m2 = legalMoves(s2);
    bool hasJump = false, hasDiagL = false, hasDiagR = false;
    for (auto& m : m2) if (!m.isWall) {
        if (m.a == cellIdx(2, 4)) hasJump = true;
        if (m.a == cellIdx(3, 3)) hasDiagL = true;
        if (m.a == cellIdx(3, 5)) hasDiagR = true;
    }
    printf("salto reto presente: %d (esperado 1), diagonais presentes: %d,%d (esperado 0,0)\n", hasJump, hasDiagL, hasDiagR);
    if (!hasJump || hasDiagL || hasDiagR) { printf("FALHOU: salto reto\n"); return 1; }

    // com muro atras do oponente -> diagonais
    State s3 = s2;
    s3.wallsH |= (1ull << slotIdx(2, 4));
    auto m3 = legalMoves(s3);
    hasJump = hasDiagL = hasDiagR = false;
    for (auto& m : m3) if (!m.isWall) {
        if (m.a == cellIdx(2, 4)) hasJump = true;
        if (m.a == cellIdx(3, 3)) hasDiagL = true;
        if (m.a == cellIdx(3, 5)) hasDiagR = true;
    }
    printf("com muro atras: salto reto %d (esperado 0), diagonais %d,%d (esperado 1,1)\n", hasJump, hasDiagL, hasDiagR);
    if (hasJump || !hasDiagL || !hasDiagR) { printf("FALHOU: diagonal\n"); return 1; }

    // bolso de 2 celulas: 3o muro deve ser ilegal
    State s4;
    s4.pawn[0] = cellIdx(0, 1);
    s4.pawn[1] = cellIdx(8, 4);
    s4.wallsV |= (1ull << slotIdx(0, 0));
    s4.wallsH |= (1ull << slotIdx(0, 1));
    bool availBefore = wallSlotAvailable(s4.wallsH, s4.wallsV, 1, 0, 2);
    uint64_t nv = s4.wallsV | (1ull << slotIdx(0, 2));
    bool pathAfter = hasPathToGoal(s4.wallsH, nv, s4.pawn[0], 0);
    printf("bolso: slot disponivel=%d (esperado 1), caminho apos fechar=%d (esperado 0)\n", availBefore, pathAfter);
    if (!availBefore || pathAfter) { printf("FALHOU: bolso\n"); return 1; }

    printf("TODOS OS TESTES DE SANIDADE PASSARAM\n");
    return 0;
}
