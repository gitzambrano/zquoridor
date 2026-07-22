// test_move_ordering.cpp -- validação mínima do move ordering melhorado
// (Fase 4.2.4/4.2.5 do plano; Prioridade 1 do plano-additional.md: CAT
// substituiu o antigo WALL_TOUCH_BONUS binário).
//
// Testes, propositalmente pequenos (o plano pede "um teste pequeno que
// funcione", testes mais fortes ficam para depois):
//   1. Permutação: orderWallMoves/orderPawnMoves nunca perdem, duplicam
//      ou inventam um lance -- só reordenam o mesmo conjunto de entrada.
//      Isso sozinho já pegaria qualquer bug de índice/cópia no buf[].
//   2. Sinal do CAT em orderWallMoves: numa posição construída à mão, um
//      muro que toca o caminho mínimo atual do oponente deve ficar
//      ordenado antes de um muro que provadamente não toca -- isolado de
//      killer/history (TT fresca) e de wallByBFS (ply além de
//      WALL_BFS_ORDER_MAX_PLY, então só o calor CAT está em jogo).
//   3. computeCorridorHeat isolado (cat.hpp): calor decrescente com
//      delta, zero além de CAT_MAX_DELTA, e bônus de gargalo aplicado só
//      quando esperado.
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <algorithm>
#include <vector>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::printf("FALHOU: %s\n", msg);
        failures++;
    } else {
        std::printf("ok: %s\n", msg);
    }
}

// mesmo multiset de lances antes/depois de ordenar (ordem pode mudar,
// conteúdo não)
static bool sameMultiset(const MoveList& a, const MoveList& b) {
    if (a.size() != b.size()) return false;
    std::vector<Move> va(a.begin(), a.end()), vb(b.begin(), b.end());
    auto cmp = [](const Move& x, const Move& y) {
        if (x.isWall != y.isWall) return x.isWall < y.isWall;
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        return x.c < y.c;
    };
    std::sort(va.begin(), va.end(), cmp);
    std::sort(vb.begin(), vb.end(), cmp);
    return va == vb;
}

static void testPermutation() {
    State s = initialState();
    Negamax neg;

    MoveList pawn;
    pawnStepMoves(s, s.turn, pawn);
    MoveList pawnOrig = pawn;
    neg.testOrderPawnMoves(pawn, /*ply=*/10, s.turn);
    check(sameMultiset(pawn, pawnOrig), "orderPawnMoves preserva o multiset de lances (posicao inicial)");

    MoveList wall;
    legalWallMoves(s, s.turn, wall);
    MoveList wallOrig = wall;
    CorridorHeat oppHeat = computeCorridorHeat(s.wallsH, s.wallsV, s.pawn[1 - s.turn], 1 - s.turn);
    neg.testOrderWallMoves(wall, /*ply=*/10, s.turn, s, oppHeat);
    check(sameMultiset(wall, wallOrig), "orderWallMoves preserva o multiset de lances (posicao inicial, 128 muros)");
}

// Constrói uma posição em que dá pra provar, sem ambiguidade, que um
// muro H toca o caminho mínimo do oponente e outro não: oponente (jogador
// 1) parado perto do meio, com um único corredor estreito na frente dele
// (muros já colocados dos dois lados), e um muro H "longe" (outro canto
// do tabuleiro) que não pode tocar o caminho mínimo dele.
static void testCatRanking() {
    State s = initialState();
    // jogador 0 (própria vez) em algum lugar neutro; jogador 1 (oponente,
    // meta = linha 0) no centro do tabuleiro.
    s.pawn[0] = cellIdx(8, 4);
    s.pawn[1] = cellIdx(4, 4);
    s.turn = 0;

    // Muro candidato A: horizontal em (r=3,c=4) -- toca diretamente a
    // aresta vertical (4,4)-(3,4) do caminho mais curto do oponente rumo
    // à linha 0 (BFS reto, sem obstáculo, esse é exatamente o caminho
    // testemunha, delta==0 nas 4 casas tocadas por este muro).
    Move wallTouch = Move::wall(0, 3, 4);
    // Muro candidato B: horizontal no canto oposto (r=0,c=0) -- as 4 casas
    // que ele toca ficam todas com delta > CAT_MAX_DELTA (calor 0).
    Move wallFar = Move::wall(0, 0, 0);

    check(isWallMoveLegal(s, s.turn, wallTouch.a, wallTouch.b, wallTouch.c), "muro A (toca o caminho) e legal na posicao de teste");
    check(isWallMoveLegal(s, s.turn, wallFar.a, wallFar.b, wallFar.c), "muro B (longe) e legal na posicao de teste");

    MoveList moves;
    moves.push_back(wallFar);   // B primeiro na entrada, de propósito --
    moves.push_back(wallTouch); // se a ordenação não fizer nada, B continua na frente

    Negamax neg;
    // ply além de WALL_BFS_ORDER_MAX_PLY: isola o calor CAT (sem TT/
    // killer/history nesta Negamax recém-criada, sem cutoffs ainda
    // registrados -- e sem o delta exato de wallByBFS, que só roda em
    // ply <= WALL_BFS_ORDER_MAX_PLY).
    int ply = WALL_BFS_ORDER_MAX_PLY + 5;
    CorridorHeat oppHeat = computeCorridorHeat(s.wallsH, s.wallsV, s.pawn[1], 1);
    neg.testOrderWallMoves(moves, ply, s.turn, s, oppHeat);

    check(moves[0] == wallTouch, "muro que toca o caminho minimo do oponente vem primeiro (calor CAT)");
}

// Valida computeCorridorHeat isoladamente (cat.hpp), sem depender de
// orderWallMoves: tabuleiro totalmente aberto (sem muro nenhum), oponente
// no meio -- o caminho ótimo é a coluna reta até a meta.
static void testCorridorHeatShape() {
    uint64_t wallsH = 0, wallsV = 0;
    int player = 1;  // meta = linha 0
    int startCell = cellIdx(4, 4);
    CorridorHeat heat = computeCorridorHeat(wallsH, wallsV, startCell, player);

    check(heat.heat[cellIdx(4, 4)] > 0, "casa inicial (delta==0) tem calor > 0");
    check(heat.heat[cellIdx(3, 4)] > 0, "casa no caminho reto (delta==0) tem calor > 0");
    check(heat.heat[cellIdx(0, 0)] == 0, "casa longe (delta > CAT_MAX_DELTA) tem calor 0");

    // delta cresce com a distância lateral na mesma linha -- calor deve
    // decrescer (não necessariamente estrito em toda casa por causa do
    // bônus de gargalo, mas aqui o tabuleiro está totalmente aberto, sem
    // gargalo nenhum, então nenhuma casa recebe o bônus e a comparação
    // pode ser estrita).
    check(heat.heat[cellIdx(4, 4)] > heat.heat[cellIdx(4, 5)],
          "calor decresce ao se afastar lateralmente do caminho otimo (tabuleiro aberto, sem bonus de gargalo)");
    check(heat.heat[cellIdx(4, 5)] >= heat.heat[cellIdx(4, 6)],
          "calor nao cresce ao continuar se afastando lateralmente");
}

int main() {
    testPermutation();
    testCatRanking();
    testCorridorHeatShape();
    std::printf("\n%s\n", failures == 0 ? "TODOS OS TESTES PASSARAM" : "HA FALHAS");
    return failures == 0 ? 0 : 1;
}
