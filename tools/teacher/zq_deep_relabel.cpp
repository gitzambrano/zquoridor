// Deep-ZQuoridor teacher bridge.
//
// Reads architecture-neutral position rows from stdin:
//   <sample-id>\t<space-separated move history>
// and emits one JSON record per row containing a full root visit policy,
// root value, and per-action Q for every visited legal root move.
//
// The teacher uses the same NNUE/MCAB implementation as production but with a
// caller-selected node budget. time_ms=0 makes node budget the deterministic
// stopping rule; positive time_ms is available for very large practical runs.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "search.hpp"
#include "mcab.hpp"

using Search = mcab::MCABSearch<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                qr::AccPair, qr::RepetitionTable, qr::SearchStats>;

namespace {

struct Options {
    std::string nnue;
    int nodes = 200000;
    int timeMs = 0;
    int leafDepth = 0;
    double cpuct = -1.0;
    double fpu = -1.0;
    double scoreScale = -1.0;
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (++i >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "--nnue") o.nnue = need("--nnue");
        else if (arg == "--nodes") o.nodes = std::atoi(need("--nodes"));
        else if (arg == "--time-ms") o.timeMs = std::atoi(need("--time-ms"));
        else if (arg == "--leaf-depth") o.leafDepth = std::atoi(need("--leaf-depth"));
        else if (arg == "--cpuct") o.cpuct = std::atof(need("--cpuct"));
        else if (arg == "--fpu") o.fpu = std::atof(need("--fpu"));
        else if (arg == "--score-scale") o.scoreScale = std::atof(need("--score-scale"));
        else {
            std::cerr << "unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    if (o.nnue.empty() || o.nodes <= 1 || o.timeMs < 0 || o.leafDepth < 0) {
        std::cerr << "usage: zq_deep_relabel --nnue PATH [--nodes N] [--time-ms MS] "
                     "[--leaf-depth D] [--cpuct C] [--fpu F] [--score-scale S]\n";
        std::exit(2);
    }
    return o;
}

bool parseLegalMove(const qr::State& state, const std::string& text, qr::Move& out) {
    if (text.size() != 2 && text.size() != 3) return false;
    int col = text[0] - 'a';
    int row = text[1] - '1';
    if (col < 0 || col >= qr::N || row < 0 || row >= qr::N) return false;
    qr::Move candidate;
    if (text.size() == 2) {
        candidate = qr::Move::pawn(qr::cellIdx(row, col));
    } else {
        if (row >= qr::WS || col >= qr::WS || (text[2] != 'h' && text[2] != 'v')) return false;
        candidate = qr::Move::wall(text[2] == 'h' ? 0 : 1, row, col);
    }
    qr::MoveList legal = qr::legalMoves(state);
    for (const auto& move : legal) {
        if (move == candidate) {
            out = move;
            return true;
        }
    }
    return false;
}

bool replay(const std::string& historyText, qr::State& state, qr::RepetitionTable& history,
            std::string& error) {
    state = qr::initialState();
    history = qr::RepetitionTable{};
    std::istringstream input(historyText);
    std::string token;
    int ply = 0;
    while (input >> token) {
        if (qr::winner(state) != -1) {
            error = "terminal before ply " + std::to_string(ply);
            return false;
        }
        qr::Move move;
        if (!parseLegalMove(state, token, move)) {
            error = "illegal move " + token + " at ply " + std::to_string(ply);
            return false;
        }
        history.push(state.hash, move.isWall);
        state = qr::applyMove(state, move);
        ++ply;
    }
    if (qr::winner(state) != -1) {
        error = "position is terminal";
        return false;
    }
    return true;
}

int canonicalPolicyIndex(const qr::Move& move, int mover) {
    qr::Move canonical = qr::mirrorMoveForPerspective(move, mover);
    return qr::moveToPolicyIndex(canonical);
}

void emitError(const std::string& id, const std::string& message) {
    std::cout << "{\"id\":\"" << id << "\",\"error\":\"";
    for (char c : message) {
        if (c == '\\' || c == '\"') std::cout << '\\';
        std::cout << c;
    }
    std::cout << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);
    if (!qr::loadWeightsQuant(opt.nnue)) {
        std::cerr << "failed to load NNUE: " << opt.nnue << "\n";
        return 3;
    }

    qr::Negamax engine;
    engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            emitError("", "expected id and tab");
            continue;
        }
        const std::string id = line.substr(0, tab);
        const std::string historyText = line.substr(tab + 1);
        qr::State state;
        qr::RepetitionTable history;
        std::string error;
        if (!replay(historyText, state, history, error)) {
            emitError(id, error);
            continue;
        }

        engine.clearTT();
        engine.resetOrderingState();
        Search search;
        search.params.nodeBudget = opt.nodes;
        search.params.leafDepth = opt.leafDepth;
        search.params.adaptiveLeafDepth = false;
        search.params.treeReuse = true;  // retain the root for inspection after chooseMoveMCAB
        search.params.clearTTPerMove = false;
        search.params.rootNoiseEnabled = false;
        search.params.backupMode = mcab::BackupMode::AvgBlend;
        search.params.rootSelectMode = mcab::RootSelectMode::MaxVisits;
        if (opt.cpuct > 0.0) search.params.cPuct = opt.cpuct;
        if (opt.fpu >= 0.0) search.params.fpuReduction = opt.fpu;
        if (opt.scoreScale > 0.0) search.params.scoreScale = opt.scoreScale;

        qr::SearchStats stats;
        mcab::McabStats mstats;
        qr::Move best = search.chooseMoveMCAB(
            engine, state, 64, opt.timeMs, stats, history, &mstats);
        const auto* root = search.rootNodeForInspection();
        if (root == nullptr || root->moves.empty()) {
            emitError(id, "teacher search left no inspectable root");
            continue;
        }

        double visitSum = 0.0;
        double valueNumerator = 0.0;
        for (size_t i = 0; i < root->moves.size(); ++i) {
            if (root->N[i] > 0.f) {
                visitSum += root->N[i];
                valueNumerator += root->W[i];
            }
        }
        const double rootValue = visitSum > 0.0 ? valueNumerator / visitSum : 0.5;
        const int bestIndex = canonicalPolicyIndex(best, state.turn);

        std::cout << "{\"id\":\"" << id
                  << "\",\"side_to_move\":" << state.turn
                  << ",\"nodes_budget\":" << opt.nodes
                  << ",\"time_ms\":" << opt.timeMs
                  << ",\"leaf_depth\":" << opt.leafDepth
                  << ",\"simulations\":" << mstats.simulations
                  << ",\"nodes_expanded\":" << mstats.nodesExpanded
                  << ",\"best_action\":" << bestIndex
                  << ",\"root_value_prob\":" << rootValue
                  << ",\"edges\":[";
        bool first = true;
        for (size_t i = 0; i < root->moves.size(); ++i) {
            const int action = canonicalPolicyIndex(root->moves[i], state.turn);
            const double n = root->N[i];
            const double q = n > 0.0 ? static_cast<double>(root->W[i]) / n : 0.5;
            if (!first) std::cout << ',';
            first = false;
            std::cout << '[' << action << ',' << n << ',' << q << ']';
        }
        std::cout << "]}\n" << std::flush;
    }
    return 0;
}
