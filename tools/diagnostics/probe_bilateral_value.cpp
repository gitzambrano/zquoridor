// Emit controlled WL calibration curves for a trained bilateral NNUE.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../src/rules.hpp"
#include "../../src/nnue.hpp"

using namespace qr;

static State controlled(int ownDist, int oppDist, int ownWalls, int oppWalls) {
    State s = initialState();
    // Player 0 moves toward row 8, player 1 toward row 0.
    int r0 = N - 1 - ownDist;
    int r1 = oppDist;
    if (r0 < 0) r0 = 0; if (r0 >= N) r0 = N-1;
    if (r1 < 0) r1 = 0; if (r1 >= N) r1 = N-1;
    s.pawn[0] = cellIdx(r0, 3);
    s.pawn[1] = cellIdx(r1, 5);
    s.wallsH = s.wallsV = 0;
    s.wallsLeft[0] = (int8_t)ownWalls;
    s.wallsLeft[1] = (int8_t)oppWalls;
    s.turn = 0;
    return s;
}

static void emit(const char* family, int od, int pd, int ow, int pw) {
    State s = controlled(od,pd,ow,pw);
    AccPair ap = buildAccPairRoot(s, nullptr);
    float z = forwardValueWLQuant(ap.acc[0], ap.acc[1]);
    float p = 1.0f/(1.0f+std::exp(-z));
    float zopp = forwardValueWLQuant(ap.acc[1], ap.acc[0]);
    std::printf("%s,%d,%d,%d,%d,%.7f,%.7f,%.7f\n",
                family,od,pd,ow,pw,z,p,z+zopp);
}

int main(int argc, char** argv) {
    if (argc < 2 || !loadWeightsQuant(argv[1])) {
        std::fprintf(stderr,"usage: %s <nnue_weights_int8.bin>\n", argv[0]);
        return 2;
    }
    std::puts("family,own_dist,opp_dist,own_walls,opp_walls,logit,pwin,antisym_error");

    for (int d=8; d>=1; --d) emit("own_distance",d,8,5,5);
    for (int d=8; d>=1; --d) emit("opp_distance",8,d,5,5);
    for (int w=0; w<=10; ++w) emit("own_walls",5,8,w,5);
    for (int w=0; w<=10; ++w) emit("opp_walls",5,8,5,w);

    // Wall-poor winning-race family: exactly the regime that used to wander.
    for (int d=7; d>=1; --d) emit("wall_poor_race",d,8,0,5);
    for (int gap=0; gap<=8; ++gap) emit("wall_gap",5,8,1,1+gap);
    return 0;
}
