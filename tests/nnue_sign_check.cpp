#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include <cstdio>

using namespace qr;

static void checkPosition(const char* label, State s) {
    EvalWeights weights; // default weights, same as evalSimple() convenience wrapper
    AccPair acc;
    acc.acc[0] = buildAccumulatorQuant(s, 0);
    acc.acc[1] = buildAccumulatorQuant(s, 1);
    int heur0 = evalSimpleW(s, 0, weights, nullptr, nullptr, nullptr);
    int heur1 = evalSimpleW(s, 1, weights, nullptr, nullptr, nullptr);
    int nnue0 = nnueEvalInt(acc, 0);
    int nnue1 = nnueEvalInt(acc, 1);
    std::printf("%-45s | heur(persp0)=%6d heur(persp1)=%6d | nnue(persp0)=%6d nnue(persp1)=%6d\n",
                label, heur0, heur1, nnue0, nnue1);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "uso: %s <pesos_int8.bin>\n", argv[0]); return 1; }
    if (!loadWeightsQuant(argv[1])) { std::fprintf(stderr, "erro ao carregar %s\n", argv[1]); return 1; }

    // Posicao inicial (simetrica) -- ambas as perspectivas devem estar perto de 0.
    {
        State s = initialState();
        s.wallsLeft[0] = 10; s.wallsLeft[1] = 10;
        s.turn = 0;
        checkPosition("posicao inicial (simetrica)", s);
    }

    // Jogador 0 quase no gol (fileira N-2, a 1 passo de vencer), jogador 1
    // ainda na fileira inicial -- deveria favorecer fortemente a perspectiva 0.
    {
        State s = initialState();
        s.pawn[0] = cellIdx(N - 2, N / 2);  // 1 passo do gol (fileira N-1)
        s.pawn[1] = cellIdx(N - 1, N / 2);  // posicao inicial de fato
        s.wallsLeft[0] = 10; s.wallsLeft[1] = 10;
        s.turn = 0;
        checkPosition("jogador 0 a 1 passo do gol", s);
    }

    // Espelho: jogador 1 quase no gol (fileira 1), jogador 0 na inicial --
    // deveria favorecer fortemente a perspectiva 1 (e penalizar a 0).
    {
        State s = initialState();
        s.pawn[0] = cellIdx(0, N / 2);      // inicial
        s.pawn[1] = cellIdx(1, N / 2);      // 1 passo do gol (fileira 0)
        s.wallsLeft[0] = 10; s.wallsLeft[1] = 10;
        s.turn = 1;
        checkPosition("jogador 1 a 1 passo do gol", s);
    }

    // Jogador 0 com vantagem grande de muros (10 vs 0) mas mesma distancia
    // -- deveria favorecer moderadamente a perspectiva 0.
    {
        State s = initialState();
        s.wallsLeft[0] = 10; s.wallsLeft[1] = 0;
        s.turn = 0;
        checkPosition("jogador 0 com 10 muros vs 0 do oponente", s);
    }

    return 0;
}
