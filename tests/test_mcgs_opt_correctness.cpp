// Lazy leaf first-visit oracle: must evaluate newly created nodes without
// materializing their actions; terminal checks and root result remain legal.
#include <cassert>
#include <cstdio>
#include "search.hpp"
#include "mcab.hpp"
int main(){
 using namespace qr;
 loadWeightsQuant("data/nnue/nnue_weights_int8.bin");
 Negamax engine; engine.setEvalMode(Negamax::EvalMode::NNUE);
 mcab::MCABSearch<Negamax,State,Move,MoveList,AccPair,RepetitionTable,SearchStats> srch;
 srch.params.nodeBudget=120; srch.params.leafDepth=0; srch.params.autoNodeBudget=false;
 State root=initialState(); RepetitionTable rep; SearchStats ss; mcab::McabStats stats;
 Move chosen=srch.chooseMoveMCAB(engine,root,40,500,ss,rep,&stats);
 MoveList legal=legalMoves(root); bool found=false;
 for(const auto& mv:legal) if(mv==chosen)found=true;
 assert(found);
 assert(stats.lazyExpansionsDeferred>0);
 assert(stats.lazyExpansionsDeferred<=stats.simulations);
 assert(stats.nodesExpanded>=1);
 std::printf("lazy expansion PASS: deferred=%lld sims=%lld expanded=%lld\n",
   stats.lazyExpansionsDeferred,stats.simulations,stats.nodesExpanded);
 return 0;
}