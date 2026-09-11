// Convert teacher move histories to the canonical compact NNUE state.
//
// Line protocol on stdin:
//   <teacher-move>\t<history move 1> <history move 2> ...
//
// Protocol rows on stdout are prefixed with "ZQTEACH" so callers can
// distinguish them from any incidental diagnostic output produced by shared
// engine code. Exactly one ZQTEACH row is emitted per input line.
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nnue.hpp"

namespace {

constexpr const char* PROTO = "ZQTEACH";

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

void emitError(size_t lineNumber, const std::string& message) {
    std::cout << PROTO << "\terror\tline " << lineNumber << ": " << message << '\n';
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
            emitError(lineNumber, "expected teacher move and tab");
            continue;
        }

        std::string teacherText = line.substr(0, tab);
        std::vector<std::string> history = splitMoves(line.substr(tab + 1));
        std::string error;
        if (!cache.setHistory(history, error)) {
            emitError(lineNumber, error);
            continue;
        }

        const qr::State& state = cache.state();
        if (qr::winner(state) != -1) {
            emitError(lineNumber, "teacher position is terminal");
            continue;
        }
        qr::Move teacherMove;
        if (!parseLegalMove(state, teacherText, teacherMove)) {
            emitError(lineNumber, "illegal teacher move " + teacherText);
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

        // wallsLeft is byte-sized in State. Streaming it directly treats it
        // as a character: 9 becomes a TAB and 10 becomes a newline, which
        // corrupts this line protocol. Cast every byte-valued scalar before
        // serializing it as text.
        std::cout
            << PROTO
            << "\tok\t" << ownPawn
            << '\t' << oppPawn
            << '\t' << wallsH
            << '\t' << wallsV
            << '\t' << static_cast<int>(state.wallsLeft[mover])
            << '\t' << static_cast<int>(state.wallsLeft[opponent])
            << '\t' << ownDist
            << '\t' << oppDist
            << '\t' << mover
            << '\t' << policyIndex
            << '\n';
    }
    return 0;
}
