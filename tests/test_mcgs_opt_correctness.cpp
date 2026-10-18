// Exact oracle: every cell vs reference shortest-path BFS, for legal midgames.
#include <cassert>
#include <cstdio>
#include <random>
#include "nnue.hpp"
int main(){
#if !ZQ_EXP_GOAL_DISTANCE_MAP
#error goal map macro required
#endif
 using namespace qr;
 std::mt19937 rng(20261017);
 int checks=0;
 for(int game=0;game<16;++game){
   State s=initialState();
   for(int ply=0;ply<18 && winner(s)<0;++ply){
     for(int p=0;p<2;++p) for(int cell=0;cell<N*N;++cell){
       const int baseline=shortestPathLen(s.wallsH,s.wallsV,cell,p);
       const int optimized=goalDistanceFromTopology(s.wallsH,s.wallsV,cell,p);
       if(baseline!=optimized){std::fprintf(stderr,"goal map mismatch %d %d %d %d\n",game,ply,p,cell);return 1;}
       ++checks;
     }
     MoveList m=legalMoves(s);
     if(m.empty())break;
     int k=(int)(rng()%m.size());
     if(ply%2==0)for(int i=0;i<(int)m.size();++i)if(m[i].isWall){k=i;break;}
     s=applyMove(s,m[k]);
   }
 }
 std::printf("goal-map oracle PASS: %d exact comparisons\n",checks);
 return checks>10000?0:2;
}