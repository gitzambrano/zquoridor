// Microbenchmark for bilateral NNUE value inference and accumulator updates.
#include <chrono>
#include <cstdio>
#include <cstdint>
#include "../src/rules.hpp"
#include "../src/nnue.hpp"

using namespace qr;

static void seedWeights() {
    auto& w = weightsQuant();
    w.QA = QA_DEFAULT; w.QB = QB_DEFAULT;
    w.w1.assign(NUM_FEATURES, {});
    for (int f=0; f<NUM_FEATURES; ++f)
        for (int h=0; h<HIDDEN; ++h)
            w.w1[f][h] = (int16_t)(((f*7+h*3)%9)-4);
    for (int i=0; i<2*HIDDEN; ++i)
        for (int j=0; j<VALUE_HIDDEN; ++j)
            w.wv1_wl[i][j] = (int8_t)(((i*5+j*11)%13)-6);
    for (int j=0; j<VALUE_HIDDEN; ++j) {
        w.bv1_wl[j] = 0;
        w.wv2_wl[j] = (int8_t)((j%7)-3);
    }
    w.bv2_wl = 0;
}

int main() {
    seedWeights();
    State s = initialState();
    AccPair ap = buildAccPairRoot(s, nullptr);
    constexpr int NITER = 200000;
    volatile float sink = 0.0f;

    auto t0 = std::chrono::steady_clock::now();
    for (int i=0; i<NITER; ++i)
        sink += forwardValueWLQuant(ap.acc[i&1], ap.acc[1-(i&1)]);
    auto t1 = std::chrono::steady_clock::now();

    MoveList lm = legalMoves(s);
    Move pawn = lm[0];
    for (size_t i=0; i<lm.size(); ++i) if (!lm[i].isWall) { pawn=lm[i]; break; }
    constexpr int UITER = 50000;
    auto t2 = std::chrono::steady_clock::now();
    for (int i=0; i<UITER; ++i) {
        AccPair child;
        makeChildAccPair(ap, child, s, pawn, nullptr);
        sink += (float)child.acc[0].v[0] * 1e-9f;
    }
    auto t3 = std::chrono::steady_clock::now();

    double value_ns = std::chrono::duration<double,std::nano>(t1-t0).count()/NITER;
    double update_ns = std::chrono::duration<double,std::nano>(t3-t2).count()/UITER;
    std::printf("bilateral_value_ns %.2f\n", value_ns);
    std::printf("bilateral_pair_update_ns %.2f\n", update_ns);
    std::printf("sink %.6f\n", sink);
    return 0;
}
