#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "search.hpp"
#include "mcab.hpp"

using Runner = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                qr::AccPair, qr::RepetitionTable, qr::SearchStats>;

static std::string moveToText(const qr::Move& m) {
    if (!m.isWall) { std::string s; s.push_back(char('a'+qr::colOf(m.a))); s.push_back(char('1'+qr::rowOf(m.a))); return s; }
    std::string s; s.push_back(char('a'+m.c)); s.push_back(char('1'+m.b)); s.push_back(m.a==0?'h':'v'); return s;
}
static bool parseLegalMove(const qr::State& s,const std::string& text,qr::Move& out){
    int c=text[0]-'a',r=text[1]-'1'; qr::Move x;
    if(text.size()==2)x=qr::Move::pawn(qr::cellIdx(r,c));
    else x=qr::Move::wall(text[2]=='h'?0:1,r,c);
    for(const auto&m:qr::legalMoves(s))if(m==x){out=m;return true;} return false;
}
int main(int argc,char**argv){
    std::string nnue; int nodes=7000; double cpuct=0.8; std::vector<std::string> hist;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--nnue") nnue=argv[++i];
        else if(a=="--nodes") nodes=std::atoi(argv[++i]);
        else if(a=="--cpuct") cpuct=std::atof(argv[++i]);
        else hist.push_back(a);
    }
    if(!qr::loadWeightsQuant(nnue)) return 3;
    qr::State s=qr::initialState(); qr::RepetitionTable history;
    for(auto&t:hist){ qr::Move m; if(!parseLegalMove(s,t,m)) return 4; history.push(s.hash,m.isWall); s=qr::applyMove(s,m); }

    qr::Negamax engine;
    engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true); engine.setPolicyOrderingMinDepth(3);
    engine.setQuiescenceEnabled(true); engine.setLmrPvsEnabled(true);
    Runner runner; mcab::McabParams p; p.autoNodeBudget=false; p.nodeBudget=nodes; p.cPuct=cpuct; p.rootNoiseEnabled=false; runner.setParams(p);
    qr::SearchStats stats; mcab::McabStats ms;
    qr::Move best=runner.choose(engine,s,40,5000,stats,history,&ms);
    const auto* root=runner.search.rootNodeForInspection();
    if(!root) return 5;
    struct R{std::string m;double P,N,Q;};
    std::vector<R> rows;
    size_t nm=std::min((size_t)root->activeMoves,root->moves.size());
    for(size_t i=0;i<nm;i++) rows.push_back({moveToText(root->moves[i]),root->P[i],root->N[i],root->N[i]>0?root->W[i]/root->N[i]:-1.0});
    std::sort(rows.begin(),rows.end(),[](const R&a,const R&b){return a.N>b.N;});
    std::cout<<std::setprecision(9)<<"{\"best\":\""<<moveToText(best)<<"\",\"nodes\":"<<nodes<<",\"simulations\":"<<ms.simulations<<",\"top\":[";
    for(size_t i=0;i<std::min<size_t>(12,rows.size());i++){ if(i)std::cout<<","; auto&r=rows[i];
      std::cout<<"{\"move\":\""<<r.m<<"\",\"P\":"<<r.P<<",\"N\":"<<r.N<<",\"Q\":"<<r.Q<<"}"; }
    std::cout<<"]}\n";
}
