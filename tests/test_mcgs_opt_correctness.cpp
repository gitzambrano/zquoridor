// Exact oracle: quantized NNUE policy for all legal actions, both perspectives.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include "nnue.hpp"
int main(){
 using namespace qr;
 if(!loadWeightsQuant("data/nnue/nnue_weights_int8.bin"))return 2;
 std::mt19937 rng(20261017);
 int checked=0;
 for(int game=0;game<20;++game){
   State s=initialState();
   for(int ply=0;ply<20 && winner(s)<0;++ply){
     MoveList moves=legalMoves(s);
     for(int side=0;side<2;++side) {
       // Only use matching legal-to-move list; both players tested by turn in games.
       if(side!=s.turn)continue;
       AccumulatorQuant acc=buildAccumulatorQuant(s,side);
       std::array<float,POLICY_OUT> full{},sparse{};
       forwardPolicyQuant(acc,full);
       forwardPolicyQuantSparse(acc,moves,side,sparse);
       for(const Move& m:moves){
         int a=moveToPolicyIndex(mirrorMoveForPerspective(m,side));
         if(std::memcmp(&full[a],&sparse[a],sizeof(float))!=0){
           std::fprintf(stderr,"sparse policy mismatch p=%d action=%d\n",side,a);
           return 1;
         }
         ++checked;
       }
     }
     if(moves.empty())break;
     int k=(int)(rng()%moves.size());
     if(ply%3==0)for(int i=0;i<(int)moves.size();++i)if(moves[i].isWall){k=i;break;}
     s=applyMove(s,moves[k]);
   }
 }
 std::printf("sparse policy bit-exact oracle PASS: %d logits\n",checked);
 return checked>1000?0:2;
}