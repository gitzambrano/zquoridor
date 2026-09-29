// Diagnostic-only root probe for pre-2.10 vs 2.10 NNUE behavior.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "search.hpp"
#include "mcab.hpp"

using namespace qr;
using ProbeSearch = mcab::MCABSearch<Negamax, State, Move, MoveList,
                                     AccPair, RepetitionTable, SearchStats>;

static std::string moveToText(const Move& m) {
    if (!m.isWall) {
        int r = rowOf(m.a), c = colOf(m.a);
        std::string s;
        s.push_back(char('a' + c));
        s.push_back(char('1' + r));
        return s;
    }
    std::string s;
    s.push_back(char('a' + m.c));
    s.push_back(char('1' + m.b));
    s.push_back(m.a == 0 ? 'h' : 'v');
    return s;
}

static bool parseLegalMove(const State& s, const std::string& text, Move& out) {
    if (text.size() != 2 && text.size() != 3) return false;
    int c = text[0] - 'a';
    int r = text[1] - '1';
    if (c < 0 || c >= N || r < 0 || r >= N) return false;

    Move candidate;
    if (text.size() == 2) {
        candidate = Move::pawn(cellIdx(r, c));
    } else {
        if (c >= WS || r >= WS) return false;
        if (text[2] != 'h' && text[2] != 'v') return false;
        candidate = Move::wall(text[2] == 'h' ? 0 : 1, r, c);
    }

    MoveList legal = legalMoves(s);
    for (const auto& m : legal) {
        if (m == candidate) {
            out = m;
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: root_probe WEIGHTS [move ...]\n";
        return 2;
    }

    if (!loadWeightsQuant(argv[1])) {
        std::cerr << "failed to load weights: " << argv[1] << "\n";
        return 3;
    }

    State state = initialState();
    RepetitionTable history;
    for (int i = 2; i < argc; ++i) {
        Move m;
        if (!parseLegalMove(state, argv[i], m)) {
            std::cerr << "illegal move: " << argv[i] << "\n";
            return 4;
        }
        history.push(state.hash, m.isWall);
        state = applyMove(state, m);
    }

    if (winner(state) != -1) {
        std::cerr << "terminal input position\n";
        return 5;
    }

    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);

    AccPair rootAcc = buildAccPairRoot(state, engine.pathCache());
    const double rootValue = (double)nnueWinProbQuant(rootAcc.acc[state.turn]);

    ProbeSearch search;
    search.params.treeReuse = true;
    search.params.nodeBudget = 20000;
    search.params.autoNodeBudget = false;
    search.params.adaptiveTime = false;

    SearchStats stats;
    mcab::McabStats mstats;
    Move chosen = search.chooseMoveMCAB(engine, state, 40, 0, stats, history, &mstats);

    const auto* root = search.rootNodeForInspection();
    if (!root || !root->expanded) {
        std::cerr << "root unavailable after search\n";
        return 6;
    }

    std::cout << std::setprecision(9);
    std::cout << "ROOT\t" << moveToText(chosen)
              << "\t" << rootValue
              << "\t" << mstats.simulations
              << "\t" << mstats.nodesExpanded
              << "\n";

    for (size_t i = 0; i < root->moves.size(); ++i) {
        const Move& m = root->moves[i];
        State child = applyMove(state, m);
        AccPair childAcc = buildAccPairRoot(child, engine.pathCache());
        const double childSideValue =
            (winner(child) == -1)
                ? (double)nnueWinProbQuant(childAcc.acc[child.turn])
                : (winner(child) == child.turn ? 1.0 : 0.0);
        const double parentChildValue = 1.0 - childSideValue;
        const double q = root->N[i] > 0.0f
            ? (double)root->W[i] / (double)root->N[i]
            : -1.0;
        std::cout << "EDGE\t" << moveToText(m)
                  << "\t" << root->P[i]
                  << "\t" << parentChildValue
                  << "\t" << root->N[i]
                  << "\t" << q
                  << "\n";
    }
    return 0;
}
