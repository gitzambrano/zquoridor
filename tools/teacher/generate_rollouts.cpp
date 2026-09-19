// generate_rollouts.cpp -- Dynamic branching rollouts from crisis positions.
//
// Plays selfplay rollouts from seed positions with top-K exploration in the
// first N plies, followed by deterministic MCAB play until termination.
// Assigns temporal-discounted value targets: V = sign * (gamma ^ remaining_plies).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "search.hpp"
#include "mcab.hpp"
#include "nnue.hpp"

using Search = mcab::MCABSearch<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                qr::AccPair, qr::RepetitionTable, qr::SearchStats>;

namespace {

struct Options {
    std::string positions;
    std::string nnue;
    std::string out;
    int rollouts = 8;
    int nodes = 128;
    int explorePlies = 4;
    int topK = 4;
    double gamma = 0.98;
    int maxPlies = 100;
    int threads = 8;
    uint64_t seed = 20260919;
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
        if (arg == "--positions") o.positions = need("--positions");
        else if (arg == "--nnue") o.nnue = need("--nnue");
        else if (arg == "--out") o.out = need("--out");
        else if (arg == "--rollouts") o.rollouts = std::atoi(need("--rollouts"));
        else if (arg == "--nodes") o.nodes = std::atoi(need("--nodes"));
        else if (arg == "--explore-plies") o.explorePlies = std::atoi(need("--explore-plies"));
        else if (arg == "--top-k") o.topK = std::atoi(need("--top-k"));
        else if (arg == "--gamma") o.gamma = std::atof(need("--gamma"));
        else if (arg == "--max-plies") o.maxPlies = std::atoi(need("--max-plies"));
        else if (arg == "--threads") o.threads = std::atoi(need("--threads"));
        else if (arg == "--seed") o.seed = std::stoull(need("--seed"));
        else {
            std::cerr << "unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    if (o.positions.empty() || o.nnue.empty() || o.out.empty()) {
        std::cerr << "usage: generate_rollouts --positions PATH --nnue PATH --out PATH "
                     "[--rollouts N] [--nodes N] [--explore-plies N] [--top-k K] "
                     "[--gamma G] [--max-plies N] [--threads N] [--seed S]\n";
        std::exit(2);
    }
    return o;
}

std::string moveToText(const qr::Move& m) {
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

void recomputeHash(qr::State& state) {
    qr::Zobrist& z = qr::zobrist();
    uint64_t hash = z.pawnKey[0][state.pawn[0]] ^ z.pawnKey[1][state.pawn[1]];
    for (int slot = 0; slot < qr::WS * qr::WS; ++slot) {
        if ((state.wallsH >> slot) & 1ull) hash ^= z.wallHKey[slot];
        if ((state.wallsV >> slot) & 1ull) hash ^= z.wallVKey[slot];
    }
    if (state.turn == 1) hash ^= z.turnKey;
    state.hash = hash;
}

bool parseCanonicalState(const std::string& text, qr::State& state, std::string& error) {
    std::istringstream input(text);
    std::string tag, extra;
    unsigned own = 0, opp = 0, ownWalls = 0, oppWalls = 0;
    unsigned long long wallsH = 0, wallsV = 0;
    if (!(input >> tag >> own >> opp >> wallsH >> wallsV >> ownWalls >> oppWalls)
        || tag != "@state" || (input >> extra)) {
        error = "state requires @state plus six integer fields";
        return false;
    }
    state = qr::State{};
    state.pawn[0] = static_cast<uint8_t>(own);
    state.pawn[1] = static_cast<uint8_t>(opp);
    state.wallsLeft[0] = static_cast<int8_t>(ownWalls);
    state.wallsLeft[1] = static_cast<int8_t>(oppWalls);
    state.turn = 0;
    for (int orientation = 0; orientation < 2; ++orientation) {
        const uint64_t bits = orientation == 0 ? wallsH : wallsV;
        for (int slot = 0; slot < qr::WS * qr::WS; ++slot) {
            if (((bits >> slot) & 1ull) == 0) continue;
            int row = slot / qr::WS, col = slot % qr::WS;
            if (!qr::wallSlotAvailable(state.wallsH, state.wallsV, orientation, row, col)) {
                error = "illegal wall topology";
                return false;
            }
            if (orientation == 0) state.wallsH |= 1ull << slot;
            else state.wallsV |= 1ull << slot;
        }
    }
    recomputeHash(state);
    return true;
}

bool replay(const std::string& historyText, qr::State& state, qr::RepetitionTable& history,
            std::vector<std::string>& movesList, std::string& error) {
    movesList.clear();
    if (historyText.rfind("@state", 0) == 0) {
        history = qr::RepetitionTable{};
        movesList.push_back(historyText);
        return parseCanonicalState(historyText, state, error);
    }
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
        movesList.push_back(token);
        ++ply;
    }
    return true;
}

int canonicalPolicyIndex(const qr::Move& move, int mover) {
    qr::Move canonical = qr::mirrorMoveForPerspective(move, mover);
    return qr::moveToPolicyIndex(canonical);
}

struct SeedPosition {
    std::string id;
    std::string historyText;
};

std::vector<SeedPosition> loadPositions(const std::string& path) {
    std::ifstream file(path);
    std::vector<SeedPosition> positions;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line.front() == '{') {
            size_t idPos = line.find("\"id\":");
            size_t histPos = line.find("\"history\":");
            if (idPos == std::string::npos || histPos == std::string::npos) continue;
            size_t idStart = line.find('"', idPos + 5) + 1;
            size_t idEnd = line.find('"', idStart);
            std::string id = line.substr(idStart, idEnd - idStart);
            size_t arrStart = line.find('[', histPos);
            size_t arrEnd = line.find(']', arrStart);
            std::string histArr = line.substr(arrStart + 1, arrEnd - arrStart - 1);
            std::istringstream ss(histArr);
            std::string tok, historyText;
            while (std::getline(ss, tok, ',')) {
                size_t s1 = tok.find('"');
                if (s1 == std::string::npos) continue;
                size_t s2 = tok.find('"', s1 + 1);
                std::string m = tok.substr(s1 + 1, s2 - s1 - 1);
                if (!historyText.empty()) historyText.push_back(' ');
                historyText.append(m);
            }
            positions.push_back({id, historyText});
        } else {
            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                positions.push_back({line.substr(0, tab), line.substr(tab + 1)});
            }
        }
    }
    return positions;
}

struct StepRecord {
    std::string id;
    int mover;
    std::vector<std::string> history;
    int chosenAction;
    double discountedValue;
    int pliesRemaining;
};

void runWorker(
    int threadId,
    const Options& opt,
    const std::vector<SeedPosition>& seeds,
    std::atomic<size_t>& nextSeedIndex,
    std::vector<StepRecord>& outputRecords,
    std::mutex& outputMutex
) {
    qr::Negamax engine;
    engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);

    std::mt19937_64 rng(opt.seed + threadId * 10007);
    std::vector<StepRecord> localRecords;

    while (true) {
        size_t idx = nextSeedIndex.fetch_add(1);
        if (idx >= seeds.size()) break;

        const auto& seed = seeds[idx];
        qr::State baseState;
        qr::RepetitionTable baseHistory;
        std::vector<std::string> baseMoves;
        std::string error;

        if (!replay(seed.historyText, baseState, baseHistory, baseMoves, error)) {
            continue;
        }
        if (qr::winner(baseState) != -1) continue;

        for (int r = 0; r < opt.rollouts; ++r) {
            qr::State state = baseState;
            qr::RepetitionTable history = baseHistory;
            std::vector<std::string> moves = baseMoves;

            struct Step {
                int mover;
                std::vector<std::string> historySnap;
                qr::Move move;
                int action;
            };
            std::vector<Step> steps;

            int terminalWinner = -1;
            for (int ply = 0; ply < opt.maxPlies; ++ply) {
                int w = qr::winner(state);
                if (w != -1) {
                    terminalWinner = w;
                    break;
                }
                if (history.count(state.hash) >= 2) {
                    // Repetition cutoff -> treat as draw
                    terminalWinner = -1;
                    break;
                }

                Search search;
                search.params.nodeBudget = opt.nodes;
                search.params.leafDepth = 0;
                search.params.treeReuse = true;
                search.params.rootNoiseEnabled = false;

                qr::SearchStats stats;
                qr::Move best = search.chooseMoveMCAB(engine, state, 64, 0, stats, history);
                const auto* root = search.rootNodeForInspection();

                qr::Move chosenMove = best;
                if (ply < opt.explorePlies && root != nullptr && root->moves.size() > 1) {
                    std::vector<std::pair<float, size_t>> ranked;
                    for (size_t m = 0; m < root->moves.size(); ++m) {
                        ranked.push_back({root->N[m], m});
                    }
                    std::sort(ranked.rbegin(), ranked.rend());
                    size_t k = std::min<size_t>(opt.topK, ranked.size());
                    std::uniform_int_distribution<size_t> dist(0, k - 1);
                    chosenMove = root->moves[ranked[dist(rng)].second];
                }

                int action = canonicalPolicyIndex(chosenMove, state.turn);
                steps.push_back({state.turn, moves, chosenMove, action});

                moves.push_back(moveToText(chosenMove));
                history.push(state.hash, chosenMove.isWall);
                state = qr::applyMove(state, chosenMove);
            }

            if (terminalWinner == -1) {
                int finalW = qr::winner(state);
                if (finalW != -1) terminalWinner = finalW;
            }

            if (terminalWinner != -1) {
                int totalSteps = static_cast<int>(steps.size());
                for (int s = 0; s < totalSteps; ++s) {
                    const auto& step = steps[s];
                    int remaining = totalSteps - s;
                    double sign = (terminalWinner == step.mover) ? 1.0 : -1.0;
                    double discounted = sign * std::pow(opt.gamma, remaining);

                    std::string stepId = seed.id + "_r" + std::to_string(r) + "_p" + std::to_string(s);
                    localRecords.push_back({
                        stepId,
                        step.mover,
                        step.historySnap,
                        step.action,
                        discounted,
                        remaining
                    });
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(outputMutex);
    outputRecords.insert(outputRecords.end(), localRecords.begin(), localRecords.end());
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);

    if (!qr::loadWeightsQuant(opt.nnue)) {
        std::cerr << "failed to load NNUE weights: " << opt.nnue << "\n";
        return 3;
    }

    std::vector<SeedPosition> seeds = loadPositions(opt.positions);
    std::cerr << "Loaded " << seeds.size() << " seed positions from " << opt.positions << "\n";

    std::atomic<size_t> nextSeedIndex{0};
    std::vector<StepRecord> outputRecords;
    std::mutex outputMutex;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < opt.threads; ++t) {
        workers.emplace_back(runWorker, t, std::cref(opt), std::cref(seeds),
                             std::ref(nextSeedIndex), std::ref(outputRecords),
                             std::ref(outputMutex));
    }
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cerr << "Generated " << outputRecords.size() << " rollout steps in " << sec << "s ("
              << (outputRecords.size() / (sec > 0 ? sec : 1.0)) << " steps/s)\n";

    std::ofstream out(opt.out);
    for (const auto& rec : outputRecords) {
        out << "{\"id\":\"" << rec.id << "\",\"schema\":\"zquoridor.position.v1\",\"side_to_move\":"
            << rec.mover << ",\"history\":[";
        for (size_t i = 0; i < rec.history.size(); ++i) {
            if (i > 0) out << ',';
            out << '"' << rec.history[i] << '"';
        }
        out << "],\"best_action\":" << rec.chosenAction
            << ",\"root_value\":" << rec.discountedValue
            << ",\"plies_remaining\":" << rec.pliesRemaining << "}\n";
    }

    return 0;
}
