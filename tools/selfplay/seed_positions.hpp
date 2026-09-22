#pragma once

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rules.hpp"

namespace qr {

struct SelfPlaySeed {
    std::string id;
    State state;
    RepetitionTable repetition;
};

inline bool parseSeedMove(const State& state, const std::string& text, Move& out) {
    if (text.size() != 2 && text.size() != 3) return false;
    const int col = std::tolower((unsigned char)text[0]) - 'a';
    const int row = text[1] - '1';
    if (col < 0 || col >= N || row < 0 || row >= N) return false;

    Move candidate;
    if (text.size() == 2) {
        candidate = Move::pawn(cellIdx(row, col));
    } else {
        const char orientation = (char)std::tolower((unsigned char)text[2]);
        if (row >= WS || col >= WS || (orientation != 'h' && orientation != 'v')) return false;
        candidate = Move::wall(orientation == 'h' ? 0 : 1, row, col);
    }

    const MoveList legal = legalMoves(state);
    for (const Move& move : legal) {
        if (move == candidate) {
            out = move;
            return true;
        }
    }
    return false;
}

inline bool extractJsonString(const std::string& line, const std::string& key,
                              std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t pos = line.find(marker);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + marker.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    const size_t end = line.find('"', pos + 1);
    if (end == std::string::npos) return false;
    value = line.substr(pos + 1, end - pos - 1);
    return true;
}

inline bool extractJsonStringArray(const std::string& line, const std::string& key,
                                   std::vector<std::string>& values) {
    values.clear();
    const std::string marker = "\"" + key + "\"";
    size_t pos = line.find(marker);
    if (pos == std::string::npos) return false;
    pos = line.find('[', pos + marker.size());
    const size_t end = line.find(']', pos);
    if (pos == std::string::npos || end == std::string::npos) return false;

    while (++pos < end) {
        if (line[pos] != '"') continue;
        const size_t close = line.find('"', pos + 1);
        if (close == std::string::npos || close > end) return false;
        values.push_back(line.substr(pos + 1, close - pos - 1));
        pos = close;
    }
    return true;
}

inline std::vector<SelfPlaySeed> loadSelfPlaySeeds(const std::string& path,
                                                   std::string& error) {
    error.clear();
    std::ifstream input(path);
    if (!input) {
        error = "cannot open seed positions: " + path;
        return {};
    }

    std::vector<SelfPlaySeed> result;
    std::string line;
    int lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) continue;

        SelfPlaySeed seed;
        std::vector<std::string> history;
        if (!extractJsonString(line, "id", seed.id)
            || !extractJsonStringArray(line, "history", history)) {
            error = "invalid seed JSONL at line " + std::to_string(lineNumber);
            return {};
        }

        seed.state = initialState();
        for (size_t ply = 0; ply < history.size(); ++ply) {
            if (winner(seed.state) != -1) {
                error = "terminal seed history at line " + std::to_string(lineNumber)
                    + ", ply " + std::to_string(ply);
                return {};
            }
            Move move;
            if (!parseSeedMove(seed.state, history[ply], move)) {
                error = "illegal seed move at line " + std::to_string(lineNumber)
                    + ", ply " + std::to_string(ply) + ": " + history[ply];
                return {};
            }
            seed.repetition.push(seed.state.hash, move.isWall);
            seed.state = applyMove(seed.state, move);
        }
        if (winner(seed.state) != -1) continue;
        seed.repetition.markRoot();
        result.push_back(seed);
    }

    if (result.empty()) error = "seed positions file contains no nonterminal positions: " + path;
    return result;
}

}  // namespace qr
