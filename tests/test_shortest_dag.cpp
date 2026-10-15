// Differential shortest-path DAG test: all optimal routes survive iff
// the distance remains unchanged after adding a legal wall.
#include <cstdio>
#include <cstdint>
#include <random>
#include "rules.hpp"
using namespace qr;
int main() {
    std::mt19937 rng(20261014u);
    long long states = 0, candidates = 0, preserves = 0, forcesLonger = 0;
    for(int game=0; game<26; ++game) {
        State s = initialState();
        for(int ply=0; ply<40 && winner(s)==-1; ++ply) {
            ++states;
            ShortestDag128 dag[2];
            int before[2];
            for(int p=0;p<2;++p) {
                before[p] = shortestPathLen(s.wallsH,s.wallsV,s.pawn[p],p);
                if(!dag[p].build(s.wallsH,s.wallsV,s.pawn[p],p) || dag[p].distance != before[p]) {
                    std::fprintf(stderr,"DAG baseline mismatch: game=%d ply=%d p=%d\n",game,ply,p);
                    return 1;
                }
            }
            MoveList moves=legalMoves(s);
            for (const Move &m:moves) {
                if(!m.isWall) continue;
                State ns=applyMove(s,m);
                for(int p=0;p<2;++p) {
                    bool same=dag[p].preservesAfterWall(m.a,slotIdx(m.b,m.c));
                    int after=shortestPathLen(ns.wallsH,ns.wallsV,ns.pawn[p],p);
                    if(same != (after == before[p])) {
                        std::fprintf(stderr,"DAG mismatch: game=%d ply=%d p=%d wall=%d/%d/%d before=%d after=%d same=%d\n",game,ply,p,m.a,m.b,m.c,before[p],after,(int)same);
                        return 2;
                    }
                    if(same) ++preserves; else ++forcesLonger;
                    ++candidates;
                }
            }
            if(moves.empty()) break;
            int idx=(int)(rng()%moves.size());
            if(ply%3==0) for(size_t i=0;i<moves.size();++i) if(moves[i].isWall){idx=(int)i;break;}
            s=applyMove(s,moves[(size_t)idx]);
        }
    }
    std::printf("Shortest DAG oracle PASS: states=%lld, candidates=%lld, preserves=%lld, lengthens=%lld\n",states,candidates,preserves,forcesLonger);
    return states>100 && candidates>1000 ? 0 : 3;
}
