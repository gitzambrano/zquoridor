// diag_wander -- dumps what the engine actually thinks in the wandering
// position of benchmarks/repro_wander.cpp.
//
// For every legal pawn move it prints the resulting shortest-path distance,
// the static NNUE value from the player-0 point of view, and an alpha-beta
// score at several depths. Then it prints the MCAB root table: prior,
// visits and Q per move.
//
// WHAT IT SHOWS. The policy head is right and gives the advancing move a
// prior of 0.73. The value head is wrong and reports approximately -300 for
// a won race. The Q values of all root moves land within 0.02 of each other,
// and their order is inverted against the truth, so MaxVisits picks a
// shuffle. See status.md, section "MCTS endgame wandering".
//
// Build (performance profile):
//   g++ -O3 -std=c++17 -march=native [-mavx2 -mfma] -Isrc \
//       -o bin/diag_wander benchmarks/diag_wander.cpp
// Run from the repo ROOT (arguments are p0row p0col p1row p1col):
//   bin/diag_wander 5 4 2 6
#include <cstdio>
#include <cstdlib>
#include <string>
#include "rules.hpp"
#include "search.hpp"
#include "nnue.hpp"
#include "../src/mcab.hpp"
using namespace qr;
using Mcab = mcab::MCABSearch<Negamax, State, Move, MoveList, AccPair, RepetitionTable, SearchStats>;

static void addH(State& s, int r, int c) { s.wallsH |= 1ull << slotIdx(r,c); s.hash ^= zobrist().wallHKey[slotIdx(r,c)]; }
static std::string cn(int cell){ char b[8]; snprintf(b,8,"%c%d",(char)('a'+cell%N),cell/N+1); return b; }
static std::string mn(const Move& m){ if(!m.isWall) return cn(m.a); char b[16]; snprintf(b,16,"%c%d%d",m.a?'V':'H',m.b,m.c); return b; }

static State pos(int p0r,int p0c,int p1r,int p1c,int turn){
    State s; s.pawn[0]=cellIdx(p0r,p0c); s.pawn[1]=cellIdx(p1r,p1c);
    s.wallsH=0;s.wallsV=0; s.wallsLeft[0]=0; s.wallsLeft[1]=8; s.turn=turn;
    Zobrist& z=zobrist();
    s.hash=z.pawnKey[0][s.pawn[0]]^z.pawnKey[1][s.pawn[1]];
    if(turn==1) s.hash^=z.turnKey;
    addH(s,0,0);addH(s,0,2);
    addH(s,1,1);addH(s,1,3);addH(s,1,5);addH(s,1,7);
    addH(s,2,0);addH(s,2,2);addH(s,2,4);addH(s,2,6);
    addH(s,5,4);addH(s,5,6);
    return s;
}

int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    if(!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")){printf("no weights\n");return 1;}
    // root of move 23: engine at e6 = (5,4), human at g3 = (2,6), engine to move
    int p0r = argc>1?atoi(argv[1]):5, p0c = argc>2?atoi(argv[2]):4;
    int p1r = argc>3?atoi(argv[3]):2, p1c = argc>4?atoi(argv[4]):6;
    State root = pos(p0r,p0c,p1r,p1c,0);
    printf("root: p0=%s d0=%d  p1=%s d1=%d  walls(%d,%d)\n\n", cn(root.pawn[0]).c_str(),
        shortestPathLen(root.wallsH,root.wallsV,root.pawn[0],0), cn(root.pawn[1]).c_str(),
        shortestPathLen(root.wallsH,root.wallsV,root.pawn[1],1),
        root.wallsLeft[0], root.wallsLeft[1]);

    Negamax eng; eng.setEvalMode(Negamax::EvalMode::NNUE);
    RepetitionTable hist;

    // --- static NNUE eval of every pawn child, plus AB scores by depth
    MoveList ms = legalMoves(root);
    printf("%-6s %-4s %-9s", "move", "d0'", "nnue(p0)");
    for(int d=2; d<=10; d+=2) printf(" ab@%-2d", d);
    printf("\n");
    for(size_t i=0;i<ms.size();i++){
        if(ms[i].isWall) continue;
        State ns=applyMove(root,ms[i]);
        AccPair acc=buildAccPairRoot(ns,nullptr);
        int e=nnueEvalInt(acc,0);   // player-0 perspective
        printf("%-6s %-4d %-9d", mn(ms[i]).c_str(),
               shortestPathLen(ns.wallsH,ns.wallsV,ns.pawn[0],0), e);
        for(int d=2; d<=10; d+=2){
            SearchStats st; RepetitionTable rt; rt.markRoot();
            int sc = -eng.searchLeaf(ns, d, st, rt, nullptr, 3000);
            printf(" %-6d", sc);
        }
        printf("\n");
    }

    // --- MCAB root statistics
    printf("\nMCAB root (nodeBudget=20000, leafDepth=0):\n");
    Mcab m; m.params.treeReuse=true;
    SearchStats st; mcab::McabStats mst{};
    Move best = m.chooseMoveMCAB(eng, root, 40, 0, st, hist, &mst);
    const auto* rn = m.rootNodeForInspection();
    printf("chosen=%s  sims=%lld\n", mn(best).c_str(), mst.simulations);
    printf("%-6s %-4s %-8s %-9s %-8s\n","move","d0'","P","N","Q");
    for(size_t i=0;i<rn->moves.size();i++){
        if(rn->moves[i].isWall && rn->N[i] < 1.f) continue;
        State ns=applyMove(root,rn->moves[i]);
        printf("%-6s %-4d %-8.4f %-9.0f %-8.4f\n", mn(rn->moves[i]).c_str(),
            rn->moves[i].isWall?-1:shortestPathLen(ns.wallsH,ns.wallsV,ns.pawn[0],0),
            rn->P[i], rn->N[i], rn->N[i]>0?rn->W[i]/rn->N[i]:-1.0);
    }
    return 0;
}
