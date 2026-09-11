// Convert teacher move histories to the canonical compact NNUE state.
//
// Line protocol on stdin:
//   <teacher-move>\t<history move 1> <history move 2> ...
//
// One tab-separated row is emitted per input line. The process caches the
// previous history prefix, so sequential trajectory records are replayed
// incrementally instead of rebuilding the game from the initial state.
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nnue.hpp"

namespace {

bool parseLegalMove(const qr::State& state, const std::string& text, qr::Move& out) {
    if (text.size() != 2 && text.size() != 3) return false;
    int col = text[0] - 'a';
    int row = text[1] - '1';
    if (col < 0 || col >= qr::N || row < 0 || row >= qr::N) return false;

    qr::Move candidate;
    if (text.size() == 2) {
        candidate = qr::Move::pawn(qr::cellIdx(row, col));
    } else {
        if (col >= qr::WS || row >= qr::WS) return false;
        if (text[2] != 'h' && text[2] != 'v') return false;
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

class PositionCache {
public:
    PositionCache() : state_(qr::initialState()) {}

    bool setHistory(const std::vector<std::string>& moves, std::string& error) {
        bool extends = moves.size() >= current_.size() &&
            std::equal(current_.begin(), current_.end(), moves.begin());
        size_t start = current_.size();
        if (!extends) {
            state_ = qr::initialState();
            current_.clear();
            start = 0;
        }

        for (size_t index = start; index < moves.size(); ++index) {
            if (qr::winner(state_) != -1) {
                error = "history continues after a terminal position";
                return false;
            }
            qr::Move move;
            if (!parseLegalMove(state_, moves[index], move)) {
                error = "illegal history move at ply " + std::to_string(index) + ": " + moves[index];
                return false;
            }
            state_ = qr::applyMove(state_, move);
            current_.push_back(moves[index]);
        }
        return true;
    }

    const qr::State& state() const { return state_; }

private:
    qr::State state_;
    std::vector<std::string> current_;
};

std::vector<std::string> splitMoves(const std::string& text) {
    std::istringstream input(text);
    std::vector<std::string> moves;
    std::string move;
    while (input >> move) moves.push_back(move);
    return moves;
}

}  // namespace

int main() {
    PositionCache cache;
    std::string line;
    size_t lineNumber = 0;
    while (std::getline(std::cin, line)) {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            std::cout << "error\tline " << lineNumber << ": expected teacher move and tab\n";
            continue;
        }

        std::string teacherText = line.substr(0, tab);
        std::vector<std::string> history = splitMoves(line.substr(tab + 1));
        std::string error;
        if (!cache.setHistory(history, error)) {
            std::cout << "error\tline " << lineNumber << ": " << error << "\n";
            continue;
        }

        const qr::State& state = cache.state();
        if (qr::winner(state) != -1) {
            std::cout << "error\tline " << lineNumber << ": teacher position is terminal\n";
            continue;
        }
        qr::Move teacherMove;
        if (!parseLegalMove(state, teacherText, teacherMove)) {
            std::cout << "error\tline " << lineNumber << ": illegal teacher move "
                      << teacherText << "\n";
            continue;
        }

        int mover = state.turn;
        int opponent = 1 - mover;
        int ownPawn = qr::mirroredPawnCell(state.pawn[mover], mover);
        int oppPawn = qr::mirroredPawnCell(state.pawn[opponent], mover);
        uint64_t wallsH = qr::mirrorWallBitboard(state.wallsH, mover);
        uint64_t wallsV = qr::mirrorWallBitboard(state.wallsV, mover);
        int ownDist = qr::shortestPathLen(
            state.wallsH, state.wallsV, state.pawn[mover], mover);
        int oppDist = qr::shortestPathLen(
            state.wallsH, state.wallsV, state.pawn[opponent], opponent);
        qr::Move canonicalMove = qr::mirrorMoveForPerspective(teacherMove, mover);
        int policyIndex = qr::moveToPolicyIndex(canonicalMove);

        std::cout
            << "ok\t" << ownPawn
            << '\t' << oppPawn
            << '\t' << wallsH
            << '\t' << wallsV
            << '\t' << state.wallsLeft[mover]
            << '\t' << state.wallsLeft[opponent]
            << '\t' << ownDist
            << '\t' << oppDist
            << '\t' << mover
            << '\t' << policyIndex
            << '\n';
    }
    return 0;
}
