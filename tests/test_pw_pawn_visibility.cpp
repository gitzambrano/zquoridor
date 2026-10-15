// Screening invariant: progressive widening must not hide legal pawn
// actions, including backward retreats, jumps or side-steps.
#include <cassert>
#include <cstdio>
#include "search.hpp"
#include "mcab.hpp"
using namespace qr;
using Search = mcab::MCABSearch<Negamax,State,Move,MoveList,AccPair,RepetitionTable,SearchStats>;

static void check(const State& s) {
    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    Search search;
    search.params.progressiveWidening = true;
    search.params.wideningInitialMoves = 8;
    search.params.wideningCoefficient = 1.0;
    search.params.wideningExponent = 0.5;
    search.params.nodeBudget = 8;
    search.params.adaptiveTime = false;
    search.params.leafDepth = 0;
    search.params.treeReuse = true;
    RepetitionTable history;
    SearchStats stats;
    mcab::McabStats ms;
    const Move chosen = search.chooseMoveMCAB(engine,s,40,0,stats,history,&ms);
    const auto* root = search.rootNodeForInspection();
    assert(root && root->expanded);
    MoveList legal = legalMoves(s);
    bool chosenLegal=false;
    for(const Move& m : legal) if(m==chosen) chosenLegal=true;
    assert(chosenLegal);
    int pawnCount = 0, activePawn = 0;
    for(const Move& m : legal) if(!m.isWall) {
        ++pawnCount;
        bool found=false;
        for(size_t i=0;i<root->moves.size();++i)
            if(root->moves[i]==m) found=true;
        assert(found && "a pawn move must not be suppressed by widening");
    }
    for(const Move& m : root->moves) if(!m.isWall) ++activePawn;
    assert(pawnCount == activePawn);
    assert((int)root->moves.size() <= (int)legal.size());
    std::printf("PW pawn invariant: legalPawn=%d, activePawn=%d, initialActive=%zu, totalLegal=%zu\n",
                pawnCount,activePawn,root->moves.size(),legal.size());
}
int main() {
    State s=initialState();
    check(s);
    // Position after two advancing pawn plies: central retreat exists.
    MoveList lm=legalMoves(s);
    for (const Move& m : lm) if(!m.isWall && m.a==cellIdx(1,4)) {s=applyMove(s,m);break;}
    lm=legalMoves(s);
    for (const Move& m : lm) if(!m.isWall && m.a==cellIdx(7,4)) {s=applyMove(s,m);break;}
    check(s);
    std::puts("PW pawn invariant PASS");
    return 0;
}
