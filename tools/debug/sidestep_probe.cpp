// Temporary benchmark probe: compare NNUE/search root behavior on sidestep_flank.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "search.hpp"
#include "mcab.hpp"

using Runner = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                qr::AccPair, qr::RepetitionTable, qr::SearchStats>;

static std::string moveToText(const qr::Move& m) {
    if (!m.isWall) {
        std::string s;
        s.push_back(char('a' + qr::colOf(m.a)));
        s.push_back(char('1' + qr::rowOf(m.a)));
        return s;
    }
    std::string s;
    s.push_back(char('a' + m.c));
    s.push_back(char('1' + m.b));
    s.push_back(m.a == 0 ? 'h' : 'v');
    return s;
}

static bool parseLegalMove(const qr::State& s, const std::string& text, qr::Move& out) {
    if (text.size() != 2 && text.size() != 3) return false;
    int c = text[0] - 'a';
    int r = text[1] - '1';
    if (c < 0 || c >= qr::N || r < 0 || r >= qr::N) return false;

    qr::Move candidate;
    if (text.size() == 2) {
        candidate = qr::Move::pawn(qr::cellIdx(r, c));
    } else {
        if (c >= qr::WS || r >= qr::WS) return false;
        if (text[2] != 'h' && text[2] != 'v') return false;
        candidate = qr::Move::wall(text[2] == 'h' ? 0 : 1, r, c);
    }
    auto legal = qr::legalMoves(s);
    for (const auto& m : legal) {
        if (m == candidate) { out = m; return true; }
    }
    return false;
}

struct Position {
    int idx;
    std::vector<std::string> moves;
};

struct EdgeDump {
    std::string move;
    double rawLogit = 0.0;
    double rawPrior = 0.0;
    double childValueRoot = 0.0;
    double p = 0.0;
    double n = 0.0;
    double q = -1.0;
};

static qr::State buildState(const Position& p, qr::RepetitionTable& hist) {
    qr::State s = qr::initialState();
    hist = qr::RepetitionTable{};
    for (const auto& t : p.moves) {
        qr::Move m;
        if (!parseLegalMove(s, t, m)) {
            std::cerr << "illegal history move " << t << " for opening " << p.idx << "\n";
            std::exit(4);
        }
        hist.push(s.hash, m.isWall);
        s = qr::applyMove(s, m);
    }
    return s;
}

static std::vector<EdgeDump> rawNetworkDump(const qr::State& s) {
    auto ap = qr::buildAccPairRoot(s);
    std::array<float, qr::POLICY_OUT> out{};
    qr::forwardPolicyQuant(ap.acc[s.turn], out);
    auto legal = qr::legalMoves(s);

    std::vector<EdgeDump> rows;
    rows.reserve(legal.size());
    double maxLogit = -1e100;
    for (const auto& m : legal) {
        double z = qr::policyLogitForMove(out, m, s.turn);
        maxLogit = std::max(maxLogit, z);
        EdgeDump e;
        e.move = moveToText(m);
        e.rawLogit = z;

        qr::State child = qr::applyMove(s, m);
        auto cap = qr::buildAccPairRoot(child);
        // Value head is from side-to-move perspective in child; negate for root mover.
        e.childValueRoot = -double(qr::forwardValueWLQuant(cap.acc[child.turn]));
        rows.push_back(e);
    }
    double sum = 0.0;
    for (auto& e : rows) {
        e.rawPrior = std::exp(e.rawLogit - maxLogit);
        sum += e.rawPrior;
    }
    if (sum > 0.0) for (auto& e : rows) e.rawPrior /= sum;
    return rows;
}

static void runOne(const char* label, const Position& p, int fixedNodes) {
    qr::RepetitionTable hist;
    qr::State s = buildState(p, hist);

    auto raw = rawNetworkDump(s);
    auto rootAcc = qr::buildAccPairRoot(s);
    double rootValue = qr::forwardValueWLQuant(rootAcc.acc[s.turn]);

    qr::Negamax engine;
    engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);
    engine.setPolicyOrderingMinDepth(3);
    engine.setQuiescenceEnabled(true);
    engine.setLmrPvsEnabled(true);

    Runner runner;
    mcab::McabParams params;
    params.enabled = true;
    params.rootNoiseEnabled = false;
    params.treeReuse = true;
    params.adaptiveTime = false;
    params.adaptiveOptimumMs = 0;
    if (fixedNodes > 0) {
        params.autoNodeBudget = false;
        params.nodeBudget = fixedNodes;
    }
    runner.setParams(params);

    qr::SearchStats stats;
    mcab::McabStats ms;
    int budgetMs = fixedNodes > 0 ? 0 : 200;
    qr::Move best = runner.choose(engine, s, 40, budgetMs, stats, hist, &ms);

    const auto* root = runner.search.rootNodeForInspection();
    if (!root) {
        std::cerr << "no root for opening " << p.idx << "\n";
        std::exit(5);
    }

    for (auto& e : raw) {
        for (size_t i = 0; i < root->moves.size(); ++i) {
            if (moveToText(root->moves[i]) != e.move) continue;
            e.p = i < root->P.size() ? root->P[i] : 0.0;
            e.n = i < root->N.size() ? root->N[i] : 0.0;
            e.q = (i < root->N.size() && root->N[i] > 0.0f)
                ? double(root->W[i]) / double(root->N[i]) : -1.0;
            break;
        }
    }

    std::sort(raw.begin(), raw.end(), [](const EdgeDump& a, const EdgeDump& b) {
        if (a.n != b.n) return a.n > b.n;
        return a.rawPrior > b.rawPrior;
    });

    std::cout << std::setprecision(9);
    std::cout << "{\"type\":\"root\",\"label\":\"" << label
              << "\",\"opening\":" << p.idx
              << ",\"mode\":\"" << (fixedNodes > 0 ? "nodes6000" : "time200")
              << "\",\"turn\":" << s.turn
              << ",\"root_value\":" << rootValue
              << ",\"chosen\":\"" << moveToText(best)
              << "\",\"nodes_expanded\":" << ms.nodesExpanded
              << ",\"simulations\":" << ms.simulations
              << ",\"root_totalN\":" << root->totalN
              << "}\n";

    int emitted = 0;
    for (const auto& e : raw) {
        bool isChosen = e.move == moveToText(best);
        if (emitted >= 15 && !isChosen) continue;
        std::cout << "{\"type\":\"edge\",\"label\":\"" << label
                  << "\",\"opening\":" << p.idx
                  << ",\"mode\":\"" << (fixedNodes > 0 ? "nodes6000" : "time200")
                  << "\",\"move\":\"" << e.move
                  << "\",\"raw_logit\":" << e.rawLogit
                  << ",\"raw_prior\":" << e.rawPrior
                  << ",\"child_value_root\":" << e.childValueRoot
                  << ",\"P\":" << e.p
                  << ",\"N\":" << e.n
                  << ",\"Q\":" << e.q
                  << ",\"chosen\":" << (isChosen ? "true" : "false")
                  << "}\n";
        ++emitted;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: sidestep_probe LABEL WEIGHTS\n";
        return 2;
    }
    if (!qr::loadWeightsQuant(argv[2])) {
        std::cerr << "cannot load weights\n";
        return 3;
    }

    const std::vector<Position> positions = {
        {27, {"e2","e8","e3","e7","e4","e6","d4","d6","d5h"}},
        {28, {"e2","e8","e3","e7","e4","e6","f4","f6","f5h"}},
        {29, {"e2","e8","e3","e7","e4","e6","d4","e5","d4v"}},
        {30, {"e2","e8","e3","e7","e4","e6","f4","e5","e4v"}},
        {31, {"e2","e8","e3","e7","e4","e6","d4","f6","d4v"}},
        {32, {"e2","e8","e3","e7","e4","e6","f4","d6","e4v"}},
    };

    for (const auto& p : positions) {
        runOne(argv[1], p, 0);
        runOne(argv[1], p, 6000);
    }
    return 0;
}
