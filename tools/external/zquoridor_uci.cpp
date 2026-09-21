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
#include "time_manager.hpp"

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
    bool policyOrdering = true;
    bool quiescence = true;
    bool lmrPvs = true;
    long long catScoreScale = -999;
    int wallBfsOrderMaxPly = -999;
    int qsMaxExtraPlies = -999;
    int qsCriticalBfsDelta = -999;
    int lmrMinDepth = -999;
    int lmrMinMoveIndex = -999;
    double lmrDivisor = -1.0;
    bool progressiveWidening = false;
    bool clearTTPerMove = false;
    bool disableTreeReuse = false;
    bool graphQCorrection = true;
    double graphLeafMix = 0.75;
    int wideningInitialMoves = -1;
    double wideningCoefficient = -1.0;
    double wideningExponent = -1.0;
    int endgameMoverWallThreshold = -999;
    int endgameLeafDepth = -1;
    int moveOverheadMs = 20;
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
        else if (a == "--no-policy-order") o.policyOrdering = false;
        else if (a == "--no-qsearch") o.quiescence = false;
        else if (a == "--no-lmr-pvs") o.lmrPvs = false;
        else if (a == "--cat-scale") o.catScoreScale = std::atoll(need("--cat-scale"));
        else if (a == "--wall-bfs-max-ply") o.wallBfsOrderMaxPly = std::atoi(need("--wall-bfs-max-ply"));
        else if (a == "--qs-max-extra") o.qsMaxExtraPlies = std::atoi(need("--qs-max-extra"));
        else if (a == "--qs-critical-delta") o.qsCriticalBfsDelta = std::atoi(need("--qs-critical-delta"));
        else if (a == "--lmr-min-depth") o.lmrMinDepth = std::atoi(need("--lmr-min-depth"));
        else if (a == "--lmr-min-move") o.lmrMinMoveIndex = std::atoi(need("--lmr-min-move"));
        else if (a == "--lmr-divisor") o.lmrDivisor = std::atof(need("--lmr-divisor"));
        else if (a == "--progressive-widening") o.progressiveWidening = true;
        else if (a == "--clear-tt-per-move") o.clearTTPerMove = true;
        else if (a == "--no-tree-reuse") o.disableTreeReuse = true;
        else if (a == "--graph-qcorr") o.graphQCorrection = true;
        else if (a == "--no-graph-qcorr") o.graphQCorrection = false;
        else if (a == "--graph-leaf-mix") o.graphLeafMix = std::atof(need("--graph-leaf-mix"));
        else if (a == "--widening-initial") o.wideningInitialMoves = std::atoi(need("--widening-initial"));
        else if (a == "--widening-coeff") o.wideningCoefficient = std::atof(need("--widening-coeff"));
        else if (a == "--widening-exp") o.wideningExponent = std::atof(need("--widening-exp"));
        else if (a == "--endgame-mover-walls") o.endgameMoverWallThreshold = std::atoi(need("--endgame-mover-walls"));
        else if (a == "--endgame-leaf-depth") o.endgameLeafDepth = std::atoi(need("--endgame-leaf-depth"));
        else if (a == "--move-overhead") o.moveOverheadMs = std::max(0, std::atoi(need("--move-overhead")));
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
    engine.setPolicyOrderingEnabled(opt.policyOrdering);
    engine.setPolicyOrderingMinDepth(opt.policyMinDepth);
    engine.setQuiescenceEnabled(opt.quiescence);
    engine.setLmrPvsEnabled(opt.lmrPvs);
    if (opt.catScoreScale != -999) engine.setCatScoreScale(opt.catScoreScale);
    if (opt.wallBfsOrderMaxPly != -999) engine.setWallBfsOrderMaxPly(opt.wallBfsOrderMaxPly);
    if (opt.qsMaxExtraPlies != -999) engine.setQsMaxExtraPlies(opt.qsMaxExtraPlies);
    if (opt.qsCriticalBfsDelta != -999) engine.setQsCriticalBfsDelta(opt.qsCriticalBfsDelta);
    if (opt.lmrMinDepth != -999) engine.setLmrMinDepth(opt.lmrMinDepth);
    if (opt.lmrMinMoveIndex != -999) engine.setLmrMinMoveIndex(opt.lmrMinMoveIndex);
    if (opt.lmrDivisor > 0.0) engine.setLmrDivisor(opt.lmrDivisor);

    Runner runner;
    mcab::McabParams params;
    params.enabled = opt.useMcab;
    if (opt.cpuct > 0.0) params.cPuct = opt.cpuct;
    if (opt.scoreScale > 0.0) params.scoreScale = opt.scoreScale;
    params.autoNodeBudget = opt.nodeBudget < 0;
    if (opt.nodeBudget >= 0) params.nodeBudget = opt.nodeBudget;
    if (opt.leafDepth >= 0) params.leafDepth = opt.leafDepth;
    if (opt.progressiveWidening) params.progressiveWidening = true;
    if (opt.clearTTPerMove) params.clearTTPerMove = true;
    if (opt.disableTreeReuse) params.treeReuse = false;
    params.graphQCorrection = opt.graphQCorrection;
    params.graphLeafMix = std::clamp(opt.graphLeafMix, 0.0, 1.0);
    if (opt.wideningInitialMoves >= 0) params.wideningInitialMoves = opt.wideningInitialMoves;
    if (opt.wideningCoefficient > 0.0) params.wideningCoefficient = opt.wideningCoefficient;
    if (opt.wideningExponent > 0.0) params.wideningExponent = opt.wideningExponent;
    if (opt.endgameMoverWallThreshold != -999) params.endgameMoverWallThreshold = opt.endgameMoverWallThreshold;
    if (opt.endgameLeafDepth >= 0) params.endgameLeafDepth = opt.endgameLeafDepth;
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
        } else if (cmd == "ponder") {
            int movetime = -1;
            std::string token;
            while (iss >> token) {
                if (token == "movetime") iss >> movetime;
            }
            if (movetime <= 0) {
                std::cout << "info string error ponder requires 'movetime MS'\n" << std::flush;
                continue;
            }
            if (qr::winner(state) != -1) {
                std::cout << "ponderok time=0 nodes=0 treeHit=0 reusedNodes=0\n" << std::flush;
                continue;
            }

            // Experimental opponent-root pondering. The caller must set the
            // position to the state BEFORE the opponent move, then grant only
            // time the opponent actually consumed. We search that root without
            // changing game state or emitting a move. McabRunner keeps the
            // resulting tree; when the real opponent move is later appended to
            // the position history, normal choose() reroots onto that child.
            //
            // Adaptive clock allocation is deliberately disabled here: ponder
            // time is free compute on the opponent clock, not our game clock.
            runner.params().adaptiveTime = false;
            runner.params().adaptiveOptimumMs = 0;

            qr::SearchStats stats;
            mcab::McabStats mstats;
            auto t0 = std::chrono::steady_clock::now();
            qr::Move suggested =
                runner.choose(engine, state, 40, movetime, stats, history, &mstats);
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            uint64_t nodes =
                runner.activeForThisEngine() ? (uint64_t)mstats.nodesExpanded : stats.nodes;
            std::cout << "ponderok time=" << elapsed
                      << " nodes=" << nodes
                      << " treeHit=" << (mstats.treeReused ? 1 : 0)
                      << " reusedNodes=" << mstats.reusedNodes
                      << " suggested=" << moveToText(suggested) << "\n" << std::flush;
        } else if (cmd == "go") {
            int movetime = -1;
            long long wtime = -1;
            long long btime = -1;
            long long winc = 0;
            long long binc = 0;
            int movesToGo = 0;
            std::string token;
            while (iss >> token) {
                if (token == "movetime") iss >> movetime;
                else if (token == "wtime") iss >> wtime;
                else if (token == "btime") iss >> btime;
                else if (token == "winc") iss >> winc;
                else if (token == "binc") iss >> binc;
                else if (token == "movestogo") iss >> movesToGo;
                else if (token == "depth") {
                    int ignored = 0;
                    iss >> ignored;
                }
            }
            if (qr::winner(state) != -1) {
                std::cout << "bestmove (none)\n" << std::flush;
                continue;
            }

            int budgetMs = movetime > 0 ? movetime : 200;
            zqtime::TimeBudget timeBudget{budgetMs, budgetMs};
            if (movetime <= 0) {
                const long long remaining = state.turn == 0 ? wtime : btime;
                const long long increment = state.turn == 0 ? winc : binc;
                if (remaining >= 0) {
                    timeBudget = zqtime::allocate(
                        zqtime::TimeControl{remaining, increment,
                                            static_cast<int>(currentMoves.size()),
                                            movesToGo, opt.moveOverheadMs});
                    budgetMs = timeBudget.optimumMs;
                }
            }

            // Adaptive time management is a clock-only feature. Fixed
            // movetime benchmarks remain bit-for-bit on the normal path.
            const bool adaptiveClock = movetime <= 0 &&
                (state.turn == 0 ? wtime : btime) >= 0;
            runner.params().adaptiveTime = adaptiveClock;
            runner.params().adaptiveOptimumMs =
                adaptiveClock ? timeBudget.optimumMs : 0;
            if (adaptiveClock) budgetMs = timeBudget.maximumMs;

            qr::SearchStats stats;
            mcab::McabStats mstats;
            auto t0 = std::chrono::steady_clock::now();
            qr::Move best = runner.choose(engine, state, 40, budgetMs, stats, history, &mstats);
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            uint64_t nodes = runner.activeForThisEngine() ? (uint64_t)mstats.nodesExpanded : stats.nodes;
            std::cout << "info depth " << stats.reachedDepth
                      << " nodes " << nodes
                      << " time " << elapsed
                      << " string budget=" << budgetMs
                      << " nodecap=" << mstats.effectiveNodeBudget
                      << " hard=" << timeBudget.maximumMs
                      << " cpuct=" << params.cPuct
                      << " scale=" << params.scoreScale
                      << " mcab=" << (params.enabled ? 1 : 0)
                      << " leaf=" << params.leafDepth
                      << " endWalls=" << params.endgameMoverWallThreshold
                      << " endLeaf=" << params.endgameLeafDepth
                      << " qsearch=" << (engine.isQuiescenceEnabled() ? 1 : 0)
                      << " lmrpvs=" << (engine.isLmrPvsEnabled() ? 1 : 0)
                      << " policy=" << (engine.isPolicyOrderingEnabled() ? 1 : 0)
                      << " policyMin=" << engine.getPolicyOrderingMinDepth()
                      << " catScale=" << engine.getCatScoreScale()
                      << " wallBfsPly=" << engine.getWallBfsOrderMaxPly()
                      << " qsExtra=" << engine.getQsMaxExtraPlies()
                      << " qsDelta=" << engine.getQsCriticalBfsDelta()
                      << " lmrDepth=" << engine.getLmrMinDepth()
                      << " lmrMove=" << engine.getLmrMinMoveIndex()
                      << " lmrDiv=" << engine.getLmrDivisor()
                      << " pw=" << (params.progressiveWidening ? 1 : 0)
                      << " clearTT=" << (params.clearTTPerMove ? 1 : 0)
                      << " reuse=" << (params.treeReuse ? 1 : 0)
                      << " treeHit=" << (mstats.treeReused ? 1 : 0)
                      << " reusedNodes=" << mstats.reusedNodes << "\n";
            std::cout << "bestmove " << moveToText(best) << "\n" << std::flush;
        } else if (cmd == "quit") {
            break;
        }
    }
    return 0;
}
