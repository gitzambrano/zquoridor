// Generate high-diversity, strictly sound central openings (3 White + 3 Black moves)
// using level-by-level BFS expansion and the Zquoridor 2.10 NNUE evaluator.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "nnue.hpp"

namespace {

std::string moveToText(const qr::Move& m) {
    std::string out;
    if (!m.isWall) {
        out.push_back(char('a' + qr::colOf(m.a)));
        out.push_back(char('1' + qr::rowOf(m.a)));
        return out;
    }
    out.push_back(char('a' + m.c));
    out.push_back(char('1' + m.b));
    out.push_back(m.a == 0 ? 'h' : 'v');
    return out;
}

bool parseLegalMove(const qr::State& s, const std::string& text, qr::Move& out) {
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

bool isCentralCandidate(const qr::State& s, const qr::Move& m) {
    if (!m.isWall) {
        int r = qr::rowOf(m.a);
        int c = qr::colOf(m.a);
        // Avoid pure backward retreat in opening
        int srcR = qr::rowOf(s.pawn[s.turn]);
        if (s.turn == 0 && r < srcR) return false; // White pawn should not move down
        if (s.turn == 1 && r > srcR) return false; // Black pawn should not move up

        // Pawn in central columns c, d, e, f, g (columns 2..6)
        if (c >= 2 && c <= 6) return true;
        // Or jump / adjacent to opponent pawn
        int oppR = qr::rowOf(s.pawn[1 - s.turn]);
        int oppC = qr::colOf(s.pawn[1 - s.turn]);
        if (std::abs(r - oppR) <= 2 && std::abs(c - oppC) <= 2) return true;
        return false;
    }
    // Wall move: row 2..5 (ranks 3..6), col 2..5 (files c..f)
    int wr = m.b;
    int wc = m.c;
    if (wr >= 2 && wr <= 5 && wc >= 2 && wc <= 5) return true;

    // Or walls in rows 1 or 6 if directly adjacent to central pawns
    int ownR = qr::rowOf(s.pawn[s.turn]);
    int ownC = qr::colOf(s.pawn[s.turn]);
    int oppR = qr::rowOf(s.pawn[1 - s.turn]);
    int oppC = qr::colOf(s.pawn[1 - s.turn]);
    if (std::abs(wr - ownR) <= 1 && std::abs(wc - ownC) <= 1) return true;
    if (std::abs(wr - oppR) <= 1 && std::abs(wc - oppC) <= 1) return true;

    return false;
}

struct ScoredMove {
    qr::Move move;
    float policyLogit;
    float winProb;
    float score;
};

std::vector<qr::Move> getSoundCandidateMoves(const qr::State& state, int topK,
                                             float minWinProb, float maxWinProb) {
    qr::MoveList legal = qr::legalMoves(state);
    if (legal.empty()) return {};

    auto acc = qr::buildAccumulatorQuant(state, state.turn);
    std::array<float, qr::POLICY_OUT> policy{};
    qr::forwardPolicyQuant(acc, policy);

    std::vector<ScoredMove> candidates;
    for (const auto& m : legal) {
        if (!isCentralCandidate(state, m)) continue;

        qr::State child = qr::applyMove(state, m);
        if (qr::winner(child) != -1) continue;

        auto childAcc = qr::buildAccumulatorQuant(child, child.turn);
        float oppWinProb = qr::nnueWinProbQuant(childAcc);
        float moverWinProb = 1.0f - oppWinProb;

        if (moverWinProb < minWinProb || moverWinProb > maxWinProb) continue;

        float logit = qr::policyLogitForMove(policy, m, state.turn);
        float balancePenalty = std::abs(moverWinProb - 0.50f) * 2.0f;
        float score = logit - balancePenalty;

        candidates.push_back({m, logit, moverWinProb, score});
    }

    if (candidates.empty()) {
        for (const auto& m : legal) {
            if (!m.isWall) {
                int r = qr::rowOf(m.a);
                int srcR = qr::rowOf(state.pawn[state.turn]);
                if (state.turn == 0 && r < srcR) continue;
                if (state.turn == 1 && r > srcR) continue;

                qr::State child = qr::applyMove(state, m);
                auto childAcc = qr::buildAccumulatorQuant(child, child.turn);
                float moverWinProb = 1.0f - qr::nnueWinProbQuant(childAcc);
                candidates.push_back({m, 0.0f, moverWinProb, 0.0f});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });

    std::vector<qr::Move> result;
    int limit = std::min((int)candidates.size(), topK);
    for (int i = 0; i < limit; ++i) {
        result.push_back(candidates[i].move);
    }
    return result;
}

std::string formatMovesJson(const std::vector<std::string>& moves) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i) ss << ",";
        ss << "\"" << moves[i] << "\"";
    }
    ss << "]";
    return ss.str();
}

struct TreeNode {
    qr::State state;
    std::vector<std::string> history;
    std::vector<uint64_t> hashes;
    float evalProb = 0.5f;
    std::string category;
    size_t basePlies = 0;
};

} // namespace

int main(int argc, char** argv) {
    std::string weightsPath = "data/nnue/nnue_weights_int8.bin";
    std::string outPath = "tools/external/openings_center_rush_sound_5k.jsonl";
    int targetCount = 5000;
    float earlyRatio = 0.20f;
    float minWinProb = 0.38f;
    float maxWinProb = 0.62f;
    int topK = 5;
    std::uint64_t seed = 20260925;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weightsPath = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (arg == "--count" && i + 1 < argc) targetCount = std::atoi(argv[++i]);
        else if (arg == "--early-ratio" && i + 1 < argc) earlyRatio = std::atof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) topK = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    }

    if (!qr::loadWeightsQuant(weightsPath.c_str())) {
        std::cerr << "Error: failed to load quantized weights from " << weightsPath << "\n";
        return 1;
    }
    std::cout << "Loaded quantized weights from: " << weightsPath << "\n";

    std::mt19937_64 rng(seed);

    const std::vector<std::string> classicRushBase = {"e2", "e8", "e3", "e7", "e4", "e6"};

    std::vector<std::vector<std::string>> earlyRoots = {
        {"e2", "e8"},
        {"e2", "e8", "e3", "e7"},
        {"e2", "e8", "d3h", "e7"},
        {"e2", "e8", "e3", "d6h"},
        {"e2", "e8", "d4v", "e7"},
        {"e2", "e8", "e3", "e5v"},
        {"e2", "e8", "e3h", "e7h"},
        {"e2", "e8", "d3h", "d7h"},
        {"e2", "e8", "c3v", "f7v"},
        {"e2", "e8", "e4v", "d5v"},
    };

    auto applyHistory = [](qr::State& s, const std::vector<std::string>& hist, std::vector<uint64_t>& hashes) -> bool {
        s = qr::initialState();
        hashes.clear();
        hashes.push_back(s.hash);
        for (const auto& txt : hist) {
            qr::Move m;
            if (!parseLegalMove(s, txt, m)) return false;
            s = qr::applyMove(s, m);
            hashes.push_back(s.hash);
        }
        return true;
    };

    // BFS expand 6 branch plies from any root
    auto expandRootsBfs = [&](const std::vector<std::vector<std::string>>& roots,
                             const std::string& catPrefix) -> std::vector<TreeNode> {
        std::vector<TreeNode> currentLevel;
        for (size_t i = 0; i < roots.size(); ++i) {
            TreeNode rootNode;
            if (applyHistory(rootNode.state, roots[i], rootNode.hashes)) {
                rootNode.history = roots[i];
                rootNode.basePlies = roots[i].size();
                rootNode.category = catPrefix + (roots.size() > 1 ? ("_" + std::to_string(i)) : "");
                currentLevel.push_back(rootNode);
            }
        }

        // Expand exactly 6 plies (3 White + 3 Black moves)
        for (int step = 0; step < 6; ++step) {
            std::vector<TreeNode> nextLevel;
            for (const auto& node : currentLevel) {
                if (qr::winner(node.state) != -1) continue;
                auto candidates = getSoundCandidateMoves(node.state, topK, minWinProb, maxWinProb);
                for (const auto& m : candidates) {
                    qr::State nextState = qr::applyMove(node.state, m);
                    // Cycle avoidance
                    if (std::find(node.hashes.begin(), node.hashes.end(), nextState.hash) != node.hashes.end())
                        continue;

                    TreeNode child;
                    child.state = nextState;
                    child.history = node.history;
                    child.history.push_back(moveToText(m));
                    child.hashes = node.hashes;
                    child.hashes.push_back(nextState.hash);
                    child.basePlies = node.basePlies;
                    child.category = node.category;
                    nextLevel.push_back(child);
                }
            }
            std::cout << "  Ply " << step + 1 << "/6 expanded: " << nextLevel.size() << " sound nodes.\n";
            currentLevel = std::move(nextLevel);
        }

        // Calculate final evaluation probability
        for (auto& leaf : currentLevel) {
            auto acc = qr::buildAccumulatorQuant(leaf.state, leaf.state.turn);
            leaf.evalProb = qr::nnueWinProbQuant(acc);
        }
        return currentLevel;
    };

    std::cout << "1. Expanding Classic Rush (6 base plies + 6 branch plies)...\n";
    std::vector<TreeNode> classicLeaves = expandRootsBfs({classicRushBase}, "classic_rush_sound");

    std::cout << "2. Expanding Early Deviations...\n";
    std::vector<TreeNode> earlyLeaves = expandRootsBfs(earlyRoots, "early_dev");

    std::cout << "Classic leaves generated: " << classicLeaves.size()
              << " | Early leaves generated: " << earlyLeaves.size() << "\n";

    // Shuffle and sample according to quota
    int earlyTarget = (int)(targetCount * earlyRatio);
    int classicTarget = targetCount - earlyTarget;

    std::shuffle(classicLeaves.begin(), classicLeaves.end(), rng);
    std::shuffle(earlyLeaves.begin(), earlyLeaves.end(), rng);

    std::vector<TreeNode> finalSelection;
    int takeClassic = std::min((int)classicLeaves.size(), classicTarget);
    for (int i = 0; i < takeClassic; ++i) finalSelection.push_back(classicLeaves[i]);

    int takeEarly = std::min((int)earlyLeaves.size(), earlyTarget);
    for (int i = 0; i < takeEarly; ++i) finalSelection.push_back(earlyLeaves[i]);

    // If early had extra capacity or classic had deficit, top up
    if ((int)finalSelection.size() < targetCount && (int)classicLeaves.size() > takeClassic) {
        for (size_t i = takeClassic; i < classicLeaves.size() && (int)finalSelection.size() < targetCount; ++i) {
            finalSelection.push_back(classicLeaves[i]);
        }
    }
    if ((int)finalSelection.size() < targetCount && (int)earlyLeaves.size() > takeEarly) {
        for (size_t i = takeEarly; i < earlyLeaves.size() && (int)finalSelection.size() < targetCount; ++i) {
            finalSelection.push_back(earlyLeaves[i]);
        }
    }

    std::shuffle(finalSelection.begin(), finalSelection.end(), rng);

    // Write output JSONL
    std::ofstream out(outPath);
    if (!out.is_open()) {
        std::cerr << "Error: could not open output file " << outPath << "\n";
        return 1;
    }

    for (size_t i = 0; i < finalSelection.size(); ++i) {
        const auto& node = finalSelection[i];
        std::ostringstream idSs;
        idSs << "cr_sound_" << std::setw(5) << std::setfill('0') << i;

        out << "{\"schema\":\"zquoridor.position.v1\","
            << "\"id\":\"" << idSs.str() << "\","
            << "\"category\":\"" << node.category << "\","
            << "\"base_plies\":" << node.basePlies << ","
            << "\"branch_plies\":6,"
            << "\"total_plies\":" << node.history.size() << ","
            << "\"eval_prob\":" << std::fixed << std::setprecision(4) << node.evalProb << ","
            << "\"history\":" << formatMovesJson(node.history) << ","
            << "\"moves\":" << formatMovesJson(node.history) << "}\n";
    }
    out.close();

    std::cout << "Successfully saved " << finalSelection.size()
              << " sound opening positions to: " << outPath << "\n";
    return 0;
}
