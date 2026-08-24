// sweep_wander -- runs the wandering segment of benchmarks/repro_wander.cpp
// under different MCAB settings and scores how far the engine advances.
//
// Perfect play scores progress +5 and wins in 5 moves. The engine side holds
// 0 walls, so the score isolates whether the search still finds the goal
// after the wall stock is spent.
//
// RESULT (2026-08-24, 300ms per move). No AvgBlend variant makes progress:
// cPuct, fpuReduction, the root-select rule and a 10x node budget all leave
// it at +0 to +2. Every MinimaxHard variant plays perfectly, even at a node
// budget of 2000. MinimaxHard is NOT the fix: it loses the arena at
// approximately -585 Elo, and it also loses the corpus of
// benchmarks/bench_mcab_endgame_progress.cpp. The endgame leaf rule
// (McabParams::endgameMoverWallThreshold) fixes the position and keeps AvgBlend.
//
// Build (performance profile):
//   g++ -O3 -std=c++17 -march=native [-mavx2 -mfma] -Isrc \
//       -o bin/sweep_wander benchmarks/sweep_wander.cpp
// Run from the repo ROOT:
//   bin/sweep_wander [timeMs]
#include <cstdio>
#include <cstdlib>
#include <string>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"
using namespace qr;
using McabRunnerT = mcab::McabRunner<Negamax,State,Move,MoveList,AccPair,RepetitionTable,SearchStats>;

static void addH(State& s,int r,int c){ s.wallsH|=1ull<<slotIdx(r,c); s.hash^=zobrist().wallHKey[slotIdx(r,c)]; }
static std::string cn(int cell){ char b[8]; snprintf(b,8,"%c%d",(char)('a'+cell%N),cell/N+1); return b; }
static State startPos(){
    State s; s.pawn[0]=cellIdx(4,4); s.pawn[1]=cellIdx(2,7);
    s.wallsH=0;s.wallsV=0;s.wallsLeft[0]=0;s.wallsLeft[1]=8;s.turn=0;
    Zobrist& z=zobrist(); s.hash=z.pawnKey[0][s.pawn[0]]^z.pawnKey[1][s.pawn[1]];
    addH(s,0,0);addH(s,0,2);addH(s,1,1);addH(s,1,3);addH(s,1,5);addH(s,1,7);
    addH(s,2,0);addH(s,2,2);addH(s,2,4);addH(s,2,6);addH(s,5,4);addH(s,5,6);
    return s;
}
static int d0(const State& s){ return shortestPathLen(s.wallsH,s.wallsV,s.pawn[0],0); }

static void run(const char* label, const mcab::McabParams& params, int timeMs, bool verbose){
    Negamax eng; eng.setEvalMode(Negamax::EvalMode::NNUE);
    McabRunnerT runner; runner.setParams(params); runner.resetTree();
    State s=startPos();
    RepetitionTable hist; hist.push(s.hash);
    const int humanCols[]={6,5,4,3,2,1,0};
    std::string line;
    int startD=d0(s), plies=0;
    for(int i=0;i<7;i++){
        SearchStats st; mcab::McabStats ms{};
        Move m=runner.choose(eng,s,40,timeMs,st,hist,&ms);
        s=applyMove(s,m); hist.push(s.hash); plies++;
        line += cn(s.pawn[0]) + "(" + std::to_string(d0(s)) + ") ";
        if(winner(s)!=-1) break;
        MoveList lm=legalMoves(s); bool ok=false;
        for(size_t j=0;j<lm.size();j++) if(!lm[j].isWall&&lm[j].a==cellIdx(2,humanCols[i])){ s=applyMove(s,lm[j]); ok=true; }
        if(!ok) break;
        hist.push(s.hash);
    }
    printf("%-46s progress=%+d over %d moves | %s\n", label, startD-d0(s), plies, line.c_str());
    (void)verbose;
}

int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    int timeMs=argc>1?atoi(argv[1]):300;
    if(!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")){printf("no weights\n");return 1;}
    printf("(perfect play = progress +5 and a win in 5 moves)\n\n");

    mcab::McabParams base;                         run("PRODUCTION (leafDepth=0, AvgBlend, MaxVisits)", base, timeMs, true);
    { auto p=base; p.enabled=false;                run("pure alpha-beta", p, timeMs, true); }
    { auto p=base; p.backupMode=mcab::BackupMode::MinimaxHard; run("MinimaxHard backup", p, timeMs, true); }
    { auto p=base; p.leafDepth=2;                  run("leafDepth=2", p, timeMs, true); }
    { auto p=base; p.leafDepth=4;                  run("leafDepth=4", p, timeMs, true); }
    { auto p=base; p.leafDepth=6;                  run("leafDepth=6", p, timeMs, true); }
    { auto p=base; p.leafDepth=4; p.backupMode=mcab::BackupMode::MinimaxHard;
                                                   run("leafDepth=4 + MinimaxHard", p, timeMs, true); }
    { auto p=base; p.rootSelectMode=mcab::RootSelectMode::MaxQ; run("rootSelect=MaxQ", p, timeMs, true); }
    { auto p=base; p.nodeBudget=200000;            run("nodeBudget=200k", p, timeMs, true); }
    { auto p=base; p.fpuReduction=0.2;             run("AvgBlend + fpuReduction=0.2", p, timeMs, true); }
    { auto p=base; p.fpuReduction=0.5;             run("AvgBlend + fpuReduction=0.5", p, timeMs, true); }
    { auto p=base; p.rootSelectMode=mcab::RootSelectMode::MaxVisitsThenQ;
                                                   run("AvgBlend + rootSelect=MaxVisitsThenQ", p, timeMs, true); }
    { auto p=base; p.cPuct=0.5;                    run("AvgBlend + cPuct=0.5", p, timeMs, true); }
    { auto p=base; p.cPuct=3.0;                    run("AvgBlend + cPuct=3.0", p, timeMs, true); }
    { auto p=base; p.backupMode=mcab::BackupMode::MinimaxHard; p.fpuReduction=0.2;
                                                   run("MinimaxHard + fpuReduction=0.2 (the rejected combo)", p, timeMs, true); }
    { auto p=base; p.backupMode=mcab::BackupMode::MinimaxHard; p.nodeBudget=2000;
                                                   run("MinimaxHard, nodeBudget=2000", p, timeMs, true); }
    return 0;
}
