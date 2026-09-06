// Parametric wall-poor race regression suite for a trained NNUE.
#include <array>
#include <cstdio>
#include <string>
#include <vector>
#include "../src/rules.hpp"
#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include "../src/mcab.hpp"

using namespace qr;
struct Scenario { int ownDist, oppDist, oppWalls; const char* name; };
static State makeRace(const Scenario& sc) {
    State s = initialState();
    s.pawn[0] = cellIdx(N - 1 - sc.ownDist, 3);
    s.pawn[1] = cellIdx(sc.oppDist, 5);
    s.wallsH = s.wallsV = 0;
    s.wallsLeft[0] = 0; s.wallsLeft[1] = (int8_t)sc.oppWalls; s.turn = 0;
    Zobrist& z = zobrist();
    s.hash = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    return s;
}
static Move greedyPawn(const State& s) {
    MoveList lm=legalMoves(s); Move best=lm[0]; int bestD=999;
    for(size_t i=0;i<lm.size();++i){ if(lm[i].isWall) continue; State ns=applyMove(s,lm[i]); int d=shortestPathLen(ns.wallsH,ns.wallsV,ns.pawn[s.turn],s.turn); if(d<bestD){bestD=d;best=lm[i];}}
    return best;
}
static bool runOne(const Scenario& sc) {
    State s=makeRace(sc); Negamax eng; eng.setEvalMode(Negamax::EvalMode::NNUE); RepetitionTable rep; rep.push(s.hash);
    using Runner=mcab::McabRunner<Negamax,State,Move,MoveList,AccPair,RepetitionTable,SearchStats>; Runner runner;
    std::array<int,16> ownCells{}; int ownTurns=0,noProgress=0,lastDist=sc.ownDist; bool twoCycle=false;
    for(int ply=0;ply<28 && winner(s)==-1;++ply){
        if(s.turn==0){ SearchStats st; Move m=runner.choose(eng,s,30,500,st,rep); int before=shortestPathLen(s.wallsH,s.wallsV,s.pawn[0],0); s=applyMove(s,m); rep.push(s.hash); int after=shortestPathLen(s.wallsH,s.wallsV,s.pawn[0],0); ownCells[ownTurns%ownCells.size()]=s.pawn[0]; ++ownTurns; if(after>=before) ++noProgress; else noProgress=0; if(ownTurns>=4){int a=ownCells[(ownTurns-1)%ownCells.size()],b=ownCells[(ownTurns-2)%ownCells.size()],c=ownCells[(ownTurns-3)%ownCells.size()],d=ownCells[(ownTurns-4)%ownCells.size()]; if(a==c&&b==d&&a!=b) twoCycle=true;} lastDist=after; if(twoCycle||noProgress>=4) break; }
        else { Move m=greedyPawn(s); s=applyMove(s,m); rep.push(s.hash); }
    }
    bool reached=winner(s)==0; bool pass=reached&&!twoCycle&&noProgress<4;
    std::printf("%-18s reached=%d cycle=%d noProgress=%d finalDist=%d => %s\n",sc.name,(int)reached,(int)twoCycle,noProgress,lastDist,pass?"PASS":"FAIL"); return pass;
}
int main(int argc,char** argv){ if(argc<2||!loadWeightsQuant(argv[1])){std::fprintf(stderr,"usage: %s <weights>\n",argv[0]);return 2;} const std::vector<Scenario> cases={{4,8,4,"race_4v8_w4"},{5,8,5,"race_5v8_w5"},{5,8,8,"race_5v8_w8"},{6,8,10,"race_6v8_w10"},{3,7,10,"race_3v7_w10"}}; int fails=0; for(const auto& sc:cases) if(!runOne(sc)) ++fails; std::printf("wandering race suite: %d/%zu failed\n",fails,cases.size()); return fails?1:0; }
