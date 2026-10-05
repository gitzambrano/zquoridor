// Generate sound weakness variation openings across all plies
// with 2nd, 3rd, and 4th best moves to prevent engine overfitting and bias.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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

std::string mirrorMoveText(const std::string& text) {
    if (text.size() != 2 && text.size() != 3) return text;
    int col = text[0] - 'a';
    char row = text[1];
    if (text.size() == 2) {
        return std::string(1, char('a' + (8 - col))) + row;
    }
    return std::string(1, char('a' + (7 - col))) + row + text[2];
}

std::vector<std::string> mirrorHistory(const std::vector<std::string>& hist) {
    std::vector<std::string> out;
    out.reserve(hist.size());
    for (const auto& m : hist) {
        out.push_back(mirrorMoveText(m));
    }
    return out;
}

bool extractJsonString(const std::string& line, const std::string& key, std::string& out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
    if (pos >= line.size() || line[pos] != '"') return false;
    ++pos;
    size_t end = line.find('"', pos);
    if (end == std::string::npos) return false;
    out = line.substr(pos, end - pos);
    return true;
}

bool extractJsonStringArray(const std::string& line, const std::string& key,
                            std::vector<std::string>& out) {
    out.clear();
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find('[', pos);
    if (pos == std::string::npos) return false;
    size_t end = line.find(']', pos);
    if (end == std::string::npos) return false;

    size_t idx = pos + 1;
    while (idx < end) {
        idx = line.find('"', idx);
        if (idx == std::string::npos || idx >= end) break;
        size_t close = line.find('"', idx + 1);
        if (close == std::string::npos || close > end) break;
        out.push_back(line.substr(idx + 1, close - idx - 1));
        idx = close + 1;
    }
    return true;
}

struct ScoredMove {
    qr::Move move;
    std::string text;
    float logit;
    float winProb;
    float score;
};

std::vector<ScoredMove> rankLegalMoves(const qr::State& state) {
    qr::MoveList legal = qr::legalMoves(state);
    if (legal.empty()) return {};

    auto acc = qr::buildAccumulatorQuant(state, state.turn);
    std::array<float, qr::POLICY_OUT> policy{};
    qr::forwardPolicyQuant(acc, policy);

    std::vector<ScoredMove> candidates;
    candidates.reserve(legal.size());

    for (const auto& m : legal) {
        qr::State child = qr::applyMove(state, m);
        if (qr::winner(child) != -1) continue;

        auto childAcc = qr::buildAccumulatorQuant(child, child.turn);
        float oppWinProb = qr::nnueWinProbQuant(childAcc);
        float moverWinProb = 1.0f - oppWinProb;

        float logit = qr::policyLogitForMove(policy, m, state.turn);
        float balancePenalty = std::abs(moverWinProb - 0.50f) * 2.0f;
        float score = logit - balancePenalty;

        candidates.push_back({m, moveToText(m), logit, moverWinProb, score});
    }

    std::sort(candidates.begin(), candidates.end(), [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });

    return candidates;
}

struct WeakSeed {
    std::string id;
    std::string origin;
    std::string category;
    std::string book;
    std::vector<std::string> moves;
};

std::vector<WeakSeed> loadWeakSeeds(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open seeds file: " << path << "\n";
        return {};
    }

    std::vector<WeakSeed> seeds;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        WeakSeed s;
        if (!extractJsonString(line, "id", s.id)) continue;
        extractJsonString(line, "origin", s.origin);
        extractJsonString(line, "category", s.category);
        extractJsonString(line, "book", s.book);
        if (!extractJsonStringArray(line, "moves", s.moves)) continue;
        if (s.moves.empty()) continue;
        seeds.push_back(s);
    }
    return seeds;
}

struct OutputVariation {
    std::string id;
    std::string sourceId;
    std::string origin;
    std::string category;
    std::string book;
    int branchPly;
    int variationRank;
    std::string originalMove;
    std::string variationMove;
    bool mirrored;
    bool isBase;
    bool isBranchRoot;
    std::vector<std::string> history;
};

std::string formatMovesJson(const std::vector<std::string>& moves) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i) ss << ", ";
        ss << "\"" << moves[i] << "\"";
    }
    ss << "]";
    return ss.str();
}

std::string toJsonLine(const OutputVariation& v) {
    std::ostringstream ss;
    ss << "{\"schema\": \"zquoridor.position.v1\", "
       << "\"id\": \"" << v.id << "\", "
       << "\"source_id\": \"" << v.sourceId << "\", "
       << "\"origin\": \"" << v.origin << "\", "
       << "\"category\": \"" << v.category << "\", "
       << "\"book\": \"" << v.book << "\", "
       << "\"branch_ply\": " << (v.branchPly >= 0 ? std::to_string(v.branchPly) : "null") << ", "
       << "\"variation_rank\": " << v.variationRank << ", "
       << "\"original_move\": \"" << v.originalMove << "\", "
       << "\"variation_move\": \"" << v.variationMove << "\", "
       << "\"mirrored\": " << (v.mirrored ? "true" : "false") << ", "
       << "\"is_base\": " << (v.isBase ? "true" : "false") << ", "
       << "\"is_branch_root\": " << (v.isBranchRoot ? "true" : "false") << ", "
       << "\"plies\": " << v.history.size() << ", "
       << "\"side_to_move\": " << (v.history.size() & 1) << ", "
       << "\"history\": " << formatMovesJson(v.history) << ", "
       << "\"moves\": " << formatMovesJson(v.history) << "}";
    return ss.str();
}

bool validateHistory(const std::vector<std::string>& history) {
    qr::State state = qr::initialState();
    for (const auto& text : history) {
        if (qr::winner(state) != -1) return false;
        qr::Move m;
        if (!parseLegalMove(state, text, m)) return false;
        state = qr::applyMove(state, m);
    }
    return qr::winner(state) == -1;
}

} // namespace

int main(int argc, char** argv) {
    std::string weightsPath = "data/nnue/nnue_weights_int8.bin";
    std::string seedsPath = "tools/external/weak_openings_mined.jsonl";
    std::string outJsonPath = "tools/external/openings_weakness_variations.json";
    std::string outJsonlPath = "tools/external/openings_weakness_variations.jsonl";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weightsPath = argv[++i];
        else if (arg == "--seeds" && i + 1 < argc) seedsPath = argv[++i];
        else if (arg == "--out-json" && i + 1 < argc) outJsonPath = argv[++i];
        else if (arg == "--out-jsonl" && i + 1 < argc) outJsonlPath = argv[++i];
    }

    std::cout << "Loading NNUE weights from: " << weightsPath << "\n";
    if (!qr::loadWeightsQuant(weightsPath)) {
        std::cerr << "Error: failed to load quantized NNUE weights\n";
        return 1;
    }

    std::cout << "Loading mined weak seeds from: " << seedsPath << "\n";
    std::vector<WeakSeed> seeds = loadWeakSeeds(seedsPath);
    if (seeds.empty()) {
        std::cerr << "Error: no weak seeds loaded from " << seedsPath << "\n";
        return 1;
    }
    std::cout << "Loaded " << seeds.size() << " weak seed openings.\n";

    std::vector<OutputVariation> outputList;
    std::set<std::vector<std::string>> seenHistories;
    int nextId = 0;

    auto tryAdd = [&](const OutputVariation& var) {
        if (seenHistories.count(var.history)) return false;
        if (!validateHistory(var.history)) return false;
        seenHistories.insert(var.history);
        OutputVariation v = var;
        std::ostringstream ss;
        ss << "wv_" << std::setw(5) << std::setfill('0') << nextId++;
        v.id = ss.str();
        outputList.push_back(v);
        return true;
    };

    int baseAdmitted = 0;
    int variationsAdmitted = 0;
    int branchesAdmitted = 0;

    for (size_t sIdx = 0; sIdx < seeds.size(); ++sIdx) {
        const auto& ws = seeds[sIdx];
        const int targetLength = (int)ws.moves.size();

        // 1. Add base weak line and its horizontal mirror
        OutputVariation baseVar;
        baseVar.sourceId = ws.id;
        baseVar.origin = ws.origin;
        baseVar.category = ws.category;
        baseVar.book = ws.book;
        baseVar.branchPly = -1;
        baseVar.variationRank = 1;
        baseVar.originalMove = "";
        baseVar.variationMove = "";
        baseVar.mirrored = false;
        baseVar.isBase = true;
        baseVar.isBranchRoot = false;
        baseVar.history = ws.moves;

        if (tryAdd(baseVar)) ++baseAdmitted;

        baseVar.mirrored = true;
        baseVar.history = mirrorHistory(ws.moves);
        if (tryAdd(baseVar)) ++baseAdmitted;

        // 2. Step ply-by-ply through the opening
        qr::State state = qr::initialState();
        std::vector<std::string> currentPrefix;

        for (int ply = 0; ply < targetLength; ++ply) {
            std::string originalMoveText = ws.moves[ply];

            // Rank legal moves at current position
            std::vector<ScoredMove> ranked = rankLegalMoves(state);

            // Collect top alternative moves for 2nd, 3rd, and 4th best choices
            std::vector<std::pair<int, ScoredMove>> alternatives;
            int rankNum = 1;
            for (const auto& sm : ranked) {
                if (sm.text == originalMoveText) {
                    continue; // Skip the exact move already played in the weak line
                }
                ++rankNum;
                if (rankNum >= 2 && rankNum <= 4) {
                    alternatives.push_back({rankNum, sm});
                }
                if (rankNum > 4) break;
            }

            // Generate variations for each alternative
            for (const auto& alt : alternatives) {
                int rank = alt.first;
                const ScoredMove& sm = alt.second;

                // Branch prefix
                std::vector<std::string> branchPrefix = currentPrefix;
                branchPrefix.push_back(sm.text);

                // Option A: Direct branch root (depth ply + 1), admitted if depth >= 4
                if ((int)branchPrefix.size() >= 4) {
                    OutputVariation rootVar;
                    rootVar.sourceId = ws.id;
                    rootVar.origin = ws.origin;
                    rootVar.category = ws.category;
                    rootVar.book = ws.book;
                    rootVar.branchPly = ply;
                    rootVar.variationRank = rank;
                    rootVar.originalMove = originalMoveText;
                    rootVar.variationMove = sm.text;
                    rootVar.mirrored = false;
                    rootVar.isBase = false;
                    rootVar.isBranchRoot = true;
                    rootVar.history = branchPrefix;

                    if (tryAdd(rootVar)) ++branchesAdmitted;

                    rootVar.mirrored = true;
                    rootVar.history = mirrorHistory(branchPrefix);
                    if (tryAdd(rootVar)) ++branchesAdmitted;
                }

                // Option B: Complete line up to targetLength
                qr::State branchState = qr::applyMove(state, sm.move);
                std::vector<std::string> fullHistory = branchPrefix;

                for (int remPly = ply + 1; remPly < targetLength; ++remPly) {
                    std::string nextTargetMove = ws.moves[remPly];
                    qr::Move nextM;
                    if (parseLegalMove(branchState, nextTargetMove, nextM)) {
                        branchState = qr::applyMove(branchState, nextM);
                        fullHistory.push_back(nextTargetMove);
                    } else {
                        // Original move illegal; pick top legal move from NNUE
                        std::vector<ScoredMove> subRanked = rankLegalMoves(branchState);
                        if (subRanked.empty()) break;
                        branchState = qr::applyMove(branchState, subRanked[0].move);
                        fullHistory.push_back(subRanked[0].text);
                    }
                }

                if ((int)fullHistory.size() == targetLength) {
                    OutputVariation fullVar;
                    fullVar.sourceId = ws.id;
                    fullVar.origin = ws.origin;
                    fullVar.category = ws.category;
                    fullVar.book = ws.book;
                    fullVar.branchPly = ply;
                    fullVar.variationRank = rank;
                    fullVar.originalMove = originalMoveText;
                    fullVar.variationMove = sm.text;
                    fullVar.mirrored = false;
                    fullVar.isBase = false;
                    fullVar.isBranchRoot = false;
                    fullVar.history = fullHistory;

                    if (tryAdd(fullVar)) ++variationsAdmitted;

                    fullVar.mirrored = true;
                    fullVar.history = mirrorHistory(fullHistory);
                    if (tryAdd(fullVar)) ++variationsAdmitted;
                }
            }

            // Advance state with original move for next ply iteration
            qr::Move origMove;
            if (!parseLegalMove(state, originalMoveText, origMove)) {
                std::cerr << "Warning: illegal move " << originalMoveText << " in seed " << ws.id << "\n";
                break;
            }
            state = qr::applyMove(state, origMove);
            currentPrefix.push_back(originalMoveText);
        }
    }

    std::cout << "Generated variation statistics:\n"
              << "  - Base weak lines (and mirrors): " << baseAdmitted << "\n"
              << "  - Full multi-ply variations: " << variationsAdmitted << "\n"
              << "  - Strategic branch roots (depth >= 4): " << branchesAdmitted << "\n"
              << "  - Total unique admitted positions: " << outputList.size() << "\n";

    // Write JSONL
    std::ofstream outJsonl(outJsonlPath);
    if (!outJsonl.is_open()) {
        std::cerr << "Error writing to " << outJsonlPath << "\n";
        return 1;
    }
    for (const auto& v : outputList) {
        outJsonl << toJsonLine(v) << "\n";
    }
    outJsonl.close();
    std::cout << "Wrote JSONL bank to: " << outJsonlPath << "\n";

    // Write formatted JSON
    std::ofstream outJson(outJsonPath);
    if (!outJson.is_open()) {
        std::cerr << "Error writing to " << outJsonPath << "\n";
        return 1;
    }
    outJson << "[\n";
    for (size_t i = 0; i < outputList.size(); ++i) {
        outJson << "  " << toJsonLine(outputList[i]);
        if (i + 1 < outputList.size()) outJson << ",";
        outJson << "\n";
    }
    outJson << "]\n";
    outJson.close();
    std::cout << "Wrote JSON bank to: " << outJsonPath << "\n";

    return 0;
}
