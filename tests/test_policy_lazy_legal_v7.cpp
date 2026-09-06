#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "../src/rules.hpp"
#include "../src/nnue.hpp"
using namespace qr;

static float denseRow(const AccumulatorQuant& acc, int o) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t,HIDDEN> a{};
    for(int i=0;i<HIDDEN;i++) a[i]=screluQuant(acc.v[i],W.QA);
    int32_t s=W.bp[o];
    for(int i=0;i<HIDDEN;i++) s += (int32_t)a[i]*(int32_t)W.wp[o][i];
    return (float)((double)s/(double)((int64_t)W.QA*(int64_t)W.QB));
}

static bool sameBits(float a,float b) { return std::memcmp(&a,&b,sizeof(float))==0; }

int main(int argc,char** argv) {
    if(argc<2 || !loadWeightsQuant(argv[1])) return 2;

    State full=initialState();
    auto af=buildAccumulatorQuant(full,full.turn);
    std::array<float,POLICY_OUT> of{};
    forwardPolicyQuant(af,of);
    for(int o=0;o<POLICY_OUT;o++) {
        float ref=denseRow(af,o);
        if(!sameBits(of[o],ref)) { std::printf("full mismatch o=%d %.9g %.9g\n",o,of[o],ref); return 1; }
    }

    State z=initialState();
    z.wallsLeft[z.turn]=0;
    auto az=buildAccumulatorQuant(z,z.turn);
    std::array<float,POLICY_OUT> oz{};
    forwardPolicyQuant(az,oz);
    MoveList lm=legalMoves(z);
    if(lm.empty()) return 1;
    for(size_t i=0;i<lm.size();i++) {
        if(lm[i].isWall) { std::printf("illegal invariant: wall generated with zero walls\n"); return 1; }
        Move canon=mirrorMoveForPerspective(lm[i],z.turn);
        int idx=moveToPolicyIndex(canon);
        float ref=denseRow(az,idx);
        if(!sameBits(oz[idx],ref)) { std::printf("zero-wall mismatch idx=%d %.9g %.9g\n",idx,oz[idx],ref); return 1; }
    }
    for(int o=N*N;o<POLICY_OUT;o++) if(oz[o]!=0.f) { std::printf("skipped wall output not zero o=%d\n",o); return 1; }

    std::printf("policy lazy legal v7 parity: PASS (%zu zero-wall legal moves)\n",lm.size());
    return 0;
}
