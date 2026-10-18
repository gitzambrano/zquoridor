// Exact oracle: reverse multi-goal reachability vs regular path-to-goal check.
#include <cassert>
#include <cstdio>
#include <random>
#include "rules.hpp"
int main(){
#if !ZQ_EXP_TOPOLOGY_REACH_CACHE
#error topology reach macro required
#endif
 using namespace qr;
 std::mt19937 rng(20261017); int checked=0;
 for(int game=0;game<16;++game){
   State s=initialState();
   for(int ply=0;ply<20 && winner(s)<0;++ply){
     for(int p=0;p<2;++p)for(int cell=0;cell<N*N;++cell){
        bool a=hasPathToGoal(s.wallsH,s.wallsV,cell,p);
        bool b=topologyCanReachGoal(s.wallsH,s.wallsV,cell,p);
        if(a!=b){std::fprintf(stderr,"reach mismatch game=%d ply=%d cell=%d p=%d\n",game,ply,cell,p);return 1;}
        ++checked;
     }
     auto moves=legalMoves(s);
     if(moves.empty())break;
     int idx=(int)(rng()%moves.size());
     if(ply%2==0)for(int i=0;i<(int)moves.size();++i)if(moves[i].isWall){idx=i;break;}
     s=applyMove(s,moves[idx]);
   }
 }
 std::printf("topology reachability oracle PASS: %d comparisons\n",checked);
 return checked>10000?0:2;
}