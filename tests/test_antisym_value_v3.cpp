// Exact zero-sum symmetry check for antisymmetric bilateral value v3.
#include <cstdio>
#include <cmath>
#include <random>
#include "../src/rules.hpp"
#include "../src/nnue.hpp"

using namespace qr;

static void seedWeights() {
    auto& w = weightsQuant();
    w.QA = QA_DEFAULT; w.QB = QB_DEFAULT;
    w.w1.assign(NUM_FEATURES, {});
    for (int f=0; f<NUM_FEATURES; ++f)
        for (int h=0; h<HIDDEN; ++h)
            w.w1[f][h] = (int16_t)(((f*13+h*19+5)%21)-10);
    for (int i=0; i<VALUE_INPUT; ++i)
        for (int j=0; j<VALUE_HIDDEN; ++j)
            w.wv1_wl[i][j] = (int8_t)(((i*7+j*11+3)%17)-8);
    for (int j=0; j<VALUE_HIDDEN; ++j) {
        w.bv1_wl[j] = ((j*97+13)%1601)-800;
        w.wv2_wl[j] = (int8_t)(((j*5+2)%13)-6);
    }
    w.bv2_wl = 777; // must cancel exactly between ordered scorers
}

int main() {
    seedWeights();
    std::mt19937_64 rng(0xA5715EEDULL);
    constexpr int GAMES=40, MAX_PLIES=120;
    long long checked=0;
    for (int g=0; g<GAMES; ++g) {
        State s=initialState();
        for (int ply=0; ply<MAX_PLIES && winner(s)==-1; ++ply) {
            AccPair ap=buildAccPairRoot(s,nullptr);
            float ab=forwardValueWLQuant(ap.acc[0],ap.acc[1]);
            float ba=forwardValueWLQuant(ap.acc[1],ap.acc[0]);
            if (ab != -ba) {
                std::fprintf(stderr,"antisymmetry failure g=%d ply=%d ab=%.9g ba=%.9g sum=%.9g\n",g,ply,ab,ba,ab+ba);
                return 2;
            }
            ++checked;
            MoveList lm=legalMoves(s);
            if (lm.empty()) break;
            s=applyMove(s,lm[(size_t)(rng()%lm.size())]);
        }
    }
    std::printf("antisymmetric value: PASS (%lld positions)\n",checked);
    return 0;
}
