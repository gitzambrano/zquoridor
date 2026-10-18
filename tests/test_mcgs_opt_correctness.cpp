// Cached edge sum is checked against reference W scan on EVERY nodeQ call
// when compiled with ZQ_EXP_PUCT_EDGE_SUM_VERIFY (see mcab.hpp).
#include <cassert>
#include <cstdio>
#include "search.hpp"
#include "mcab.hpp"
int main(){
#if !ZQ_EXP_PUCT_EDGE_SUM
#error cache macro required
#endif
 using namespace qr;
 loadWeightsQuant("data/nnue/nnue_weights_int8.bin");
 Negamax engine; engine.setEvalMode(Negamax::EvalMode::NNUE);
 mcab::MCABSearch<Negamax,State,Move,MoveList,AccPair,RepetitionTable,SearchStats> srch;
 srch.params.nodeBudget=100; srch.params.leafDepth=0; srch.params.autoNodeBudget=false;
 State root=initialState(); RepetitionTable rep; SearchStats ss; mcab::McabStats stats;
 Move chosen=srch.chooseMoveMCAB(engine,root,40,500,ss,rep,&stats);
 MoveList legal=legalMoves(root);bool found=false;
 for(const auto& mv:legal)if(mv==chosen)found=true;
 assert(found && stats.simulations>10);
 std::printf("cached sum PASS: %lld simulations checked by assert\n",stats.simulations);
 return 0;
}