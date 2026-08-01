#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include <cstdio>
#include <random>

using namespace qr;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "uso: %s <pesos_int8.bin>\n", argv[0]); return 1; }
    if (!loadWeightsQuant(argv[1])) { std::fprintf(stderr, "erro ao carregar %s\n", argv[1]); return 1; }

    std::mt19937_64 rng(12345);
    int totalPlies = 0, totalMismatches = 0;
    int worstAbsDiff = 0;

    for (int game = 0; game < 30; game++) {
        State s = initialState();
        AccPair acc;
        acc.acc[0] = buildAccumulatorQuant(s, 0);
        acc.acc[1] = buildAccumulatorQuant(s, 1);

        for (int ply = 0; ply < 80; ply++) {
            int w = winner(s);
            if (w != -1) break;

            // Checa consistência ANTES de aplicar o próximo lance: acumulador
            // incremental (mantido desde o início via updateAccumulatorForMoveQuant)
            // vs rebuild do zero na posição atual.
            AccPair rebuilt;
            rebuilt.acc[0] = buildAccumulatorQuant(s, 0);
            rebuilt.acc[1] = buildAccumulatorQuant(s, 1);

            for (int side = 0; side < 2; side++) {
                int vInc = nnueEvalInt(acc, side);
                int vRebuilt = nnueEvalInt(rebuilt, side);
                totalPlies++;
                int diff = std::abs(vInc - vRebuilt);
                if (diff > worstAbsDiff) worstAbsDiff = diff;
                if (diff != 0) {
                    totalMismatches++;
                    if (totalMismatches <= 10) {
                        std::printf("DIVERGENCIA jogo=%d ply=%d side=%d incremental=%d rebuild=%d diff=%d\n",
                                    game, ply, side, vInc, vRebuilt, diff);
                    }
                }
            }

            MoveList moves = legalMoves(s);
            if (moves.size() == 0) break;
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            Move m = moves[dist(rng)];

            // Atualiza acumulador incrementalmente para AMBAS as perspectivas,
            // exatamente como search.hpp faz ao descer um ply (childAcc a
            // partir de curAcc).
            AccPair childAcc;
            childAcc.acc[0] = acc.acc[0];
            childAcc.acc[1] = acc.acc[1];
            updateAccumulatorForMoveQuant(childAcc.acc[0], /*viewerIsMover=*/(0 == s.turn), s, m);
            updateAccumulatorForMoveQuant(childAcc.acc[1], /*viewerIsMover=*/(1 == s.turn), s, m);

            s = applyMove(s, m);
            acc = childAcc;
        }
    }

    std::printf("\n=== RESULTADO ===\n");
    std::printf("posicoes checadas (jogo x ply x perspectiva) = %d\n", totalPlies);
    std::printf("divergencias incremental vs rebuild          = %d\n", totalMismatches);
    std::printf("maior |diff| observado (unidades NNUE_EVAL_SCALE) = %d\n", worstAbsDiff);
    if (totalMismatches == 0) {
        std::printf("OK: acumulador incremental bate com rebuild em 100%% das posicoes.\n");
        return 0;
    } else {
        std::printf("FALHA: ha divergencia real entre update incremental e rebuild.\n");
        return 1;
    }
}
