// Minimal UCI-style adapter for external Quoridor engine benchmarks.
// It intentionally keeps the production Zquoridor search path intact:
// Negamax + NNUE + McabRunner, with the defaults from the checked-out ref.
#include <algorithm>
#include <chrono>
#include <cstdlib>
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
        int r = qr::rowOf(m.a), c = qr::colOf(m.a);
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

    qr::MoveList legal = qr::legalMoves(s);
    for (const auto& m : legal) {
        if (m == candidate) {
            out = m;
            return true;
        }
    }
    return false;
}

struct Options {
    std::string nnuePath;
    bool useMcab = true;
    double cpuct = -1.0;
    double scoreScale = -1.0;
    int nodeBudget = -1;
    int leafDepth = -1;
    int policyMinDepth = 3;
};

static Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--nnue") o.nnuePath = need("--nnue");
        else if (a == "--no-mcab") o.useMcab = false;
        else if (a == "--cpuct") o.cpuct = std::atof(need("--cpuct"));
        else if (a == "--score-scale") o.scoreScale = std::atof(need("--score-scale"));
        else if (a == "--nodes") o.nodeBudget = std::atoi(need("--nodes"));
        else if (a == "--leaf-depth") o.leafDepth = std::atoi(need("--leaf-depth"));
        else if (a == "--policy-min-depth") o.policyMinDepth = std::atoi(need("--policy-min-depth"));
        else {
            std::cerr << "unknown argument: " << a << "\n";
            std::exit(2);
        }
    }
    return o;
}

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);
    if (opt.nnuePath.empty()) {
        std::cerr << "--nnue PATH is required\n";
        return 2;
    }
    if (!qr::loadWeightsQuant(opt.nnuePath)) {
        std::cerr << "failed to load NNUE weights: " << opt.nnuePath << "\n";
        return 3;
    }

    qr::Negamax engine;
    engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);
    engine.setPolicyOrderingMinDepth(opt.policyMinDepth);

    Runner runner;
    mcab::McabParams params;
    params.enabled = opt.useMcab;
    if (opt.cpuct > 0.0) params.cPuct = opt.cpuct;
    if (opt.scoreScale > 0.0) params.scoreScale = opt.scoreScale;
    if (opt.nodeBudget >= 0) params.nodeBudget = opt.nodeBudget;
    if (opt.leafDepth >= 0) params.leafDepth = opt.leafDepth;
    params.rootNoiseEnabled = false;
    runner.setParams(params);

    qr::State state = qr::initialState();
    qr::RepetitionTable history;
    std::vector<std::string> currentMoves;

    auto rebuildPosition = [&](const std::vector<std::string>& moves) -> bool {
        bool isExtension = moves.size() >= currentMoves.size() &&
            std::equal(currentMoves.begin(), currentMoves.end(), moves.begin());
        if (!isExtension) {
            runner.resetTree();
            engine.clearTT();
        }
        state = qr::initialState();
        history = qr::RepetitionTable{};
        for (const std::string& text : moves) {
            if (qr::winner(state) != -1) return false;
            qr::Move m;
            if (!parseLegalMove(state, text, m)) return false;
            history.push(state.hash, m.isWall);
            state = qr::applyMove(state, m);
        }
        currentMoves = moves;
        return true;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "uci") {
            std::cout << "id name Zquoridor-external-bench\n";
            std::cout << "id author gitzambrano\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            state = qr::initialState();
            history = qr::RepetitionTable{};
            currentMoves.clear();
            engine.clearTT();
            runner.resetTree();
        } else if (cmd == "position") {
            std::string token;
            iss >> token;
            if (token != "startpos") {
                std::cout << "info string error only startpos is supported\n" << std::flush;
                continue;
            }
            std::vector<std::string> moves;
            if (iss >> token) {
                if (token != "moves") {
                    std::cout << "info string error expected 'moves'\n" << std::flush;
                    continue;
                }
                while (iss >> token) moves.push_back(token);
            }
            if (!rebuildPosition(moves)) {
                std::cout << "info string error illegal position history\n" << std::flush;
            }
        } else if (cmd == "go") {
            int movetime = 200;
            std::string token;
            while (iss >> token) {
                if (token == "movetime") iss >> movetime;
                else if (token == "depth") {
                    int ignored = 0;
                    iss >> ignored;
                }
            }
            if (qr::winner(state) != -1) {
                std::cout << "bestmove (none)\n" << std::flush;
                continue;
            }

            qr::SearchStats stats;
            mcab::McabStats mstats;
            auto t0 = std::chrono::steady_clock::now();
            qr::Move best = runner.choose(engine, state, 40, movetime, stats, history, &mstats);
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            uint64_t nodes = runner.activeForThisEngine() ? (uint64_t)mstats.nodesExpanded : stats.nodes;
            std::cout << "info depth " << stats.reachedDepth
                      << " nodes " << nodes
                      << " time " << elapsed
                      << " string cpuct=" << params.cPuct
                      << " scale=" << params.scoreScale
                      << " mcab=" << (params.enabled ? 1 : 0) << "\n";
            std::cout << "bestmove " << moveToText(best) << "\n" << std::flush;
        } else if (cmd == "quit") {
            break;
        }
    }
    return 0;
}
