// Strict production-policy equivalence oracle for the legal-action sparse head.
// Test both side-to-move frames, mirror canonicalization, random legal topologies,
// no-walls policy branch, and exact normalized priors. Full reference is unchanged.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <array>
#include "nnue.hpp"

int main() {
  using namespace qr;
  if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) return 2;
  std::mt19937 rng(20261027);
  uint64_t logitsChecked=0, priorsChecked=0, noWallCases=0, side1Cases=0;
  for (int game=0; game<42; ++game) {
    State s=initialState();
    for (int ply=0; ply<30 && winner(s)<0; ++ply) {
      for (int side=0; side<2; ++side) {
        State candidate=s;
        if(candidate.turn!=side) {
          candidate.hash ^= zobrist().turnKey;
          candidate.turn=side;
        }
        // Zero-wall cases exercise the existing production 81-output shortcut.
        if ((game%6)==0) candidate.wallsLeft[side]=0;
        MoveList moves=legalMoves(candidate);
        if(moves.empty()) continue;
        AccumulatorQuant acc=buildAccumulatorQuant(candidate,side);
        std::array<float,POLICY_OUT> ref{},opt{};
        forwardPolicyQuant(acc,ref);
        forwardPolicyQuantSparse(acc,moves,side,opt);
        if(side==1)++side1Cases;
        if(candidate.wallsLeft[side]==0)++noWallCases;
        double referenceExpSum=0, optimizedExpSum=0;
        float maxRef=-1e30f, maxOpt=-1e30f;
        for(const Move& m:moves) {
          int k=moveToPolicyIndex(mirrorMoveForPerspective(m,side));
          if(std::memcmp(&ref[k],&opt[k],sizeof(float))!=0) {
            std::fprintf(stderr,"MISMATCH game=%d ply=%d side=%d action=%d baseline=%.9g opt=%.9g\n",
                         game,ply,side,k,ref[k],opt[k]); return 1;
          }
          if(ref[k]>maxRef) maxRef=ref[k];
          if(opt[k]>maxOpt) maxOpt=opt[k];
          ++logitsChecked;
        }
        for(const Move& m:moves) {
          int k=moveToPolicyIndex(mirrorMoveForPerspective(m,side));
          referenceExpSum+=std::exp((double)(ref[k]-maxRef));
          optimizedExpSum+=std::exp((double)(opt[k]-maxOpt));
        }
        if(referenceExpSum!=optimizedExpSum || maxRef!=maxOpt) return 3;
        for(const Move& m:moves) {
          int k=moveToPolicyIndex(mirrorMoveForPerspective(m,side));
          double p=std::exp((double)(ref[k]-maxRef))/referenceExpSum;
          double q=std::exp((double)(opt[k]-maxOpt))/optimizedExpSum;
          if(p!=q) return 4;
          ++priorsChecked;
        }
      }
      MoveList moves=legalMoves(s);
      if(moves.empty()) break;
      int k=(int)(rng()%moves.size());
      if(ply%3==0) {
        for(int j=0;j<(int)moves.size();++j) if(moves[j].isWall){ k=j; break;}
      }
      s=applyMove(s,moves[k]);
    }
  }
  if(logitsChecked<100000 || noWallCases<100 || side1Cases<100) {
    std::fprintf(stderr,"Insufficient policy coverage: %llu logits %llu no-wall %llu side1\n",
      (unsigned long long)logitsChecked,(unsigned long long)noWallCases,(unsigned long long)side1Cases);return 5;
  }
  std::printf("PASS sparse policy: %llu exact legal logits, %llu exact priors, %llu no-wall, %llu side1\n",
      (unsigned long long)logitsChecked,(unsigned long long)priorsChecked,
      (unsigned long long)noWallCases,(unsigned long long)side1Cases);
  return 0;
}
