// lazy_acc_parity.cpp -- paridade do Item 3 (update preguiçoso por
// perspectiva, ver AccPair/makeChildAccPair/resolvePending em nnue.hpp):
// compara a cadeia de AccPair produzida por makeChildAccPair (o mesmo
// caminho que search.hpp usa pra construir childAcc a cada lance) contra
// um rebuild do zero (buildAccumulatorQuant) na mesma posição, em partidas
// aleatórias. Checa tanto o vetor bruto do acumulador (v[HIDDEN]) quanto os
// 4 caches de bucket (ownDistBucket/oppDistBucket/ownWallsLeftBucket/
// oppWallsLeftBucket) das DUAS perspectivas -- não só o eval final, que
// poderia mascarar uma divergência pequena.
//
// Duas coberturas por partida:
//   (a) resolve + compara a CADA lance (caminho "eager" de fato, cobre o
//       caso comum de a busca ler o eval logo em seguida);
//   (b) deixa vários lances se acumularem SEM resolver manualmente (só
//       makeChildAccPair encadeando, exatamente como aconteceria numa
//       subárvore que não sofre cutoff antes de precisar da perspectiva
//       adiada) e só resolve/compara de vez em quando -- cobre o caso que
//       pegou o bug na sessão anterior (mismatch após cadeia de updates
//       adiados, não só um único hop).
#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include <cstdio>
#include <random>
#include <cmath>

using namespace qr;

static bool accEqual(const AccumulatorQuant& a, const AccumulatorQuant& b) {
    for (int i = 0; i < HIDDEN; i++) if (a.v[i] != b.v[i]) return false;
    return a.ownDistBucket == b.ownDistBucket && a.oppDistBucket == b.oppDistBucket &&
           a.ownWallsLeftBucket == b.ownWallsLeftBucket && a.oppWallsLeftBucket == b.oppWallsLeftBucket;
}

struct Mismatch {
    int game, ply, side;
};

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "uso: %s <pesos_int8.bin>\n", argv[0]); return 1; }
    if (!loadWeightsQuant(argv[1])) { std::fprintf(stderr, "erro ao carregar %s\n", argv[1]); return 1; }

    std::mt19937_64 rng(20260808);
    int totalChecks = 0, totalMismatches = 0;
    Mismatch firstMismatch{-1, -1, -1};

    const int NUM_GAMES = 80;
    const int MAX_PLIES = 90;

    for (int game = 0; game < NUM_GAMES; game++) {
        bool resolveEveryPly = (game % 2 == 0);   // alterna cobertura (a)/(b) descrita acima
        int resolveEvery = resolveEveryPly ? 1 : 5;

        State s = initialState();
        AccPair chain = buildAccPairRoot(s);

        for (int ply = 0; ply < MAX_PLIES; ply++) {
            int w = winner(s);
            if (w != -1) break;
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            Move m = moves[rng() % moves.size()];

            State before = s;
            AccPair next;
            makeChildAccPair(chain, next, before, m);
            s = applyMove(before, m);
            chain = next;

            if ((ply % resolveEvery) != 0) continue;

            // Materializa as duas perspectivas e compara contra rebuild.
            resolvePending(chain, 0);
            resolvePending(chain, 1);
            for (int side = 0; side < 2; side++) {
                AccumulatorQuant fresh = buildAccumulatorQuant(s, side);
                totalChecks++;
                if (!accEqual(chain.acc[side], fresh)) {
                    totalMismatches++;
                    if (firstMismatch.game == -1) firstMismatch = {game, ply, side};
                }
            }
        }
    }

    std::printf("lazy_acc_parity: %d checagens (partidas=%d), %d mismatches\n",
                totalChecks, NUM_GAMES, totalMismatches);
    if (totalMismatches > 0) {
        std::printf("primeiro mismatch: partida %d, ply %d, perspectiva %d\n",
                     firstMismatch.game, firstMismatch.ply, firstMismatch.side);
        return 1;
    }
    std::printf("OK -- Item 3 (update preguicoso por perspectiva) bate com rebuild do zero.\n");
    return 0;
}
