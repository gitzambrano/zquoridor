// Verify that the bilateral NNUE accumulator stays bit-exact with a cold rebuild
// across long random legal sequences.
#include <cstdio>
#include <cstdint>
#include <random>
#include "../src/rules.hpp"
#include "../src/nnue.hpp"

using namespace qr;

static void seedDeterministicWeights() {
    auto& w = weightsQuant();
    w.QA = QA_DEFAULT;
    w.QB = QB_DEFAULT;
    w.w1.assign(NUM_FEATURES, {});
    for (int f = 0; f < NUM_FEATURES; ++f)
        for (int h = 0; h < HIDDEN; ++h)
            w.w1[f][h] = (int16_t)(((f * 17 + h * 31 + 7) % 23) - 11);
    for (int h = 0; h < HIDDEN; ++h)
        w.b1[h] = (int16_t)(((h * 13 + 3) % 19) - 9);

    for (int i = 0; i < 2 * HIDDEN; ++i)
        for (int j = 0; j < VALUE_HIDDEN; ++j)
            w.wv1_wl[i][j] = (int8_t)(((i * 11 + j * 7 + 5) % 17) - 8);
    for (int j = 0; j < VALUE_HIDDEN; ++j) {
        w.bv1_wl[j] = ((j * 101 + 17) % 2001) - 1000;
        w.wv2_wl[j] = (int8_t)(((j * 5 + 3) % 15) - 7);
    }
    w.bv2_wl = 1234;
}

static bool sameAcc(const AccumulatorQuant& a, const AccumulatorQuant& b) {
    if (a.ownDistBucket != b.ownDistBucket || a.oppDistBucket != b.oppDistBucket ||
        a.ownWallsLeftBucket != b.ownWallsLeftBucket || a.oppWallsLeftBucket != b.oppWallsLeftBucket)
        return false;
    for (int i = 0; i < HIDDEN; ++i)
        if (a.v[i] != b.v[i]) return false;
    return true;
}

int main() {
    seedDeterministicWeights();
    std::mt19937_64 rng(0xB17A7EULL);
    constexpr int GAMES = 80;
    constexpr int MAX_PLIES = 140;
    long long checked = 0;

    for (int g = 0; g < GAMES; ++g) {
        State s = initialState();
        AccPair inc = buildAccPairRoot(s, nullptr);
        for (int ply = 0; ply < MAX_PLIES && winner(s) == -1; ++ply) {
            MoveList lm = legalMoves(s);
            if (lm.empty()) break;
            Move m = lm[(size_t)(rng() % lm.size())];

            AccPair child;
            makeChildAccPair(inc, child, s, m, nullptr);
            State ns = applyMove(s, m);
            AccPair cold = buildAccPairRoot(ns, nullptr);

            for (int p = 0; p < 2; ++p) {
                if (child.pending[p]) {
                    std::fprintf(stderr, "pending accumulator at game=%d ply=%d persp=%d\n", g, ply, p);
                    return 2;
                }
                if (!sameAcc(child.acc[p], cold.acc[p])) {
                    std::fprintf(stderr, "accumulator mismatch at game=%d ply=%d persp=%d\n", g, ply, p);
                    return 3;
                }
            }
            if (nnueEvalInt(child, ns.turn) != nnueEvalInt(cold, ns.turn)) {
                std::fprintf(stderr, "bilateral eval mismatch at game=%d ply=%d\n", g, ply);
                return 4;
            }

            inc = child;
            s = ns;
            ++checked;
        }
    }

    std::printf("bilateral accumulator parity: PASS (%lld plies)\n", checked);
    return 0;
}
