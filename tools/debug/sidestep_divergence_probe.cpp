
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "search.hpp"
#include "mcab.hpp"

using Runner = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                qr::AccPair, qr::RepetitionTable, qr::SearchStats>;

static std::string moveToText(const qr::Move& m) {
    if (!m.isWall) {
        std::string s; s.push_back(char('a'+qr::colOf(m.a))); s.push_back(char('1'+qr::rowOf(m.a))); return s;
    }
    std::string s; s.push_back(char('a'+m.c)); s.push_back(char('1'+m.b)); s.push_back(m.a==0?'h':'v'); return s;
}
static bool parseLegalMove(const qr::State& s, const std::string& t, qr::Move& out) {
    if (t.size()!=2 && t.size()!=3) return false;
    int c=t[0]-'a', r=t[1]-'1'; if(c<0||c>=qr::N||r<0||r>=qr::N) return false;
    qr::Move cand;
    if(t.size()==2) cand=qr::Move::pawn(qr::cellIdx(r,c));
    else { if(c>=qr::WS||r>=qr::WS||(t[2]!='h'&&t[2]!='v')) return false; cand=qr::Move::wall(t[2]=='h'?0:1,r,c); }
    for(const auto& m: qr::legalMoves(s)) if(m==cand){ out=m; return true; }
    return false;
}
struct P { int id; std::vector<std::string> hist; std::vector<std::string> watch; };

static qr::State makeState(const P& p, qr::RepetitionTable& hist) {
    qr::State s=qr::initialState(); hist=qr::RepetitionTable{};
    for(const auto& t:p.hist){ qr::Move m; if(!parseLegalMove(s,t,m)){ std::cerr<<"illegal "<<p.id<<" "<<t<<"\n"; std::exit(4);} hist.push(s.hash,m.isWall); s=qr::applyMove(s,m); }
    return s;
}

struct R { std::string move; double prior=0, childV=0, N=0, Q=-1; };

static void run(const char* label,const P& p,int fixedNodes){
    qr::RepetitionTable hist; qr::State s=makeState(p,hist);
    auto ap=qr::buildAccPairRoot(s);
    std::array<float,qr::POLICY_OUT> pol{}; qr::forwardPolicyQuant(ap.acc[s.turn],pol);
    auto legal=qr::legalMoves(s);
    std::vector<R> rows; rows.reserve(legal.size());
    double maxz=-1e100;
    std::vector<double> z; z.reserve(legal.size());
    for(const auto& m:legal){ double v=qr::policyLogitForMove(pol,m,s.turn); z.push_back(v); maxz=std::max(maxz,v); }
    double sum=0; for(double v:z) sum+=std::exp(v-maxz);
    for(size_t i=0;i<legal.size();++i){
        R r; r.move=moveToText(legal[i]); r.prior=std::exp(z[i]-maxz)/sum;
        auto ch=qr::applyMove(s,legal[i]); auto cap=qr::buildAccPairRoot(ch);
        r.childV=-double(qr::forwardValueWLQuant(cap.acc[ch.turn]));
        rows.push_back(r);
    }

    qr::Negamax eng; eng.setEvalMode(qr::Negamax::EvalMode::NNUE); eng.setPolicyOrderingEnabled(true); eng.setPolicyOrderingMinDepth(3); eng.setQuiescenceEnabled(true); eng.setLmrPvsEnabled(true);
    Runner runner; mcab::McabParams pa; pa.enabled=true; pa.rootNoiseEnabled=false; pa.treeReuse=true; pa.adaptiveTime=false; pa.adaptiveOptimumMs=0;
    if(fixedNodes>0){ pa.autoNodeBudget=false; pa.nodeBudget=fixedNodes; }
    runner.setParams(pa);
    qr::SearchStats st; mcab::McabStats ms;
    auto best=runner.choose(eng,s,40,fixedNodes>0?0:200,st,hist,&ms);
    const auto* root=runner.search.rootNodeForInspection();
    for(auto& r:rows) for(size_t i=0;i<root->moves.size();++i) if(moveToText(root->moves[i])==r.move){ r.N=root->N[i]; r.Q=root->N[i]>0?double(root->W[i])/double(root->N[i]):-1; break; }

    auto watched=[&](const std::string& m){ return std::find(p.watch.begin(),p.watch.end(),m)!=p.watch.end(); };
    std::sort(rows.begin(),rows.end(),[](const R&a,const R&b){return a.N>b.N;});
    std::cout<<std::setprecision(9);
    std::cout<<"{\"type\":\"root\",\"label\":\""<<label<<"\",\"id\":"<<p.id<<",\"mode\":\""<<(fixedNodes?"nodes6000":"time200")
             <<"\",\"turn\":"<<s.turn<<",\"chosen\":\""<<moveToText(best)<<"\",\"nodes\":"<<ms.nodesExpanded<<"}\n";
    int rank=0;
    for(const auto& r:rows){
        bool keep=rank<12||watched(r.move);
        if(keep) std::cout<<"{\"type\":\"edge\",\"label\":\""<<label<<"\",\"id\":"<<p.id<<",\"mode\":\""<<(fixedNodes?"nodes6000":"time200")
            <<"\",\"move\":\""<<r.move<<"\",\"prior\":"<<r.prior<<",\"childV\":"<<r.childV<<",\"N\":"<<r.N<<",\"Q\":"<<r.Q
            <<",\"watch\":"<<(watched(r.move)?"true":"false")<<"}\n";
        ++rank;
    }
}
int main(int argc,char**argv){
    if(argc!=3||!qr::loadWeightsQuant(argv[2])) return 2;
    std::vector<P> ps={
      {280,{"e2","e8","e3","e7","e4","e6","f4","f6","f5h","e6","e4","e6v","e3h","e5","c3h","d5h","f4v","e8v","b4v","b6h"},{"d5","a2h"}},
      {290,{"e2","e8","e3","e7","e4","e6","d4","e5","d4v","c5h","e4h","e5h","g4h","a5h","g5v","d2v","g7v"},{"c1h","c3h","c2h"}},
      {310,{"e2","e8","e3","e7","e4","e6","d4","f6","d4v","c6h"},{"e4h","f2h"}},
      {320,{"e2","e8","e3","e7","e4","e6","f4","d6","e4v","f6h"},{"f5","d4h"}},
      {321,{"e2","e8","e3","e7","e4","e6","f4","d6","e4v","f5h","c5v","h5h","f3","e2v","c7v","f2h","g3","g3v","g4"},{"e8v","d5"}}
    };
    for(const auto&p:ps){run(argv[1],p,0);run(argv[1],p,6000);}
}
