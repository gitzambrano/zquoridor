#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "search.hpp"

static std::string moveToText(const qr::Move& m) {
    if (!m.isWall) {
        int r = qr::rowOf(m.a), c = qr::colOf(m.a);
        std::string s; s.push_back(char('a'+c)); s.push_back(char('1'+r)); return s;
    }
    std::string s; s.push_back(char('a'+m.c)); s.push_back(char('1'+m.b)); s.push_back(m.a==0?'h':'v'); return s;
}

static bool parseLegalMove(const qr::State& s, const std::string& text, qr::Move& out) {
    if (text.size()!=2 && text.size()!=3) return false;
    int c=text[0]-'a', r=text[1]-'1';
    if (c<0||c>=qr::N||r<0||r>=qr::N) return false;
    qr::Move candidate;
    if (text.size()==2) candidate=qr::Move::pawn(qr::cellIdx(r,c));
    else {
        if (c>=qr::WS||r>=qr::WS||(text[2]!='h'&&text[2]!='v')) return false;
        candidate=qr::Move::wall(text[2]=='h'?0:1,r,c);
    }
    auto legal=qr::legalMoves(s);
    for (const auto& m: legal) if (m==candidate) { out=m; return true; }
    return false;
}

static double sigmoid(double x) { return 1.0/(1.0+std::exp(-x)); }

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--nnue") return 2;
    if (!qr::loadWeightsQuant(argv[2])) return 3;

    qr::State s = qr::initialState();
    for (int i=3;i<argc;i++) {
        qr::Move m;
        if (!parseLegalMove(s, argv[i], m)) {
            std::cerr << "illegal " << argv[i] << "\n";
            return 4;
        }
        s = qr::applyMove(s,m);
    }

    auto acc = qr::buildAccumulatorQuant(s, s.turn);
    float rootLogit = qr::forwardValueWLQuant(acc);
    std::array<float, qr::POLICY_OUT> logits{};
    qr::forwardPolicyQuant(acc, logits);
    auto legal = qr::legalMoves(s);

    struct Row { std::string move; double logit, prior, q1; };
    std::vector<Row> rows;
    rows.reserve(legal.size());
    double maxLogit=-1e100;
    for (const auto& m: legal) maxLogit=std::max(maxLogit,(double)qr::policyLogitForMove(logits,m,s.turn));
    double den=0.0;
    for (const auto& m: legal) den += std::exp((double)qr::policyLogitForMove(logits,m,s.turn)-maxLogit);

    for (const auto& m: legal) {
        double l=qr::policyLogitForMove(logits,m,s.turn);
        double p=std::exp(l-maxLogit)/den;
        qr::State child=qr::applyMove(s,m);
        auto childAcc=qr::buildAccumulatorQuant(child, child.turn);
        double childLogit=qr::forwardValueWLQuant(childAcc);
        double q1=1.0-sigmoid(childLogit);
        rows.push_back({moveToText(m),l,p,q1});
    }

    std::cout << std::setprecision(10);
    std::cout << "{\"turn\":" << s.turn
              << ",\"root_logit\":" << rootLogit
              << ",\"root_winprob\":" << sigmoid(rootLogit)
              << ",\"moves\":[";
    for (size_t i=0;i<rows.size();++i) {
        if (i) std::cout << ",";
        const auto& r=rows[i];
        std::cout << "{\"move\":\"" << r.move << "\",\"policy_logit\":" << r.logit
                  << ",\"prior\":" << r.prior << ",\"q1\":" << r.q1 << "}";
    }
    std::cout << "]}\n";
    return 0;
}
