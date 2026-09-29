// Generate one-ply neighborhoods around the sidestep-flank positions that
// account for the measured 2.10 regression, plus opening #27 as a holdout.
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "rules.hpp"

using namespace qr;

static std::string moveText(const Move& m) {
    std::string s;
    if (!m.isWall) {
        s.push_back(char('a' + colOf(m.a)));
        s.push_back(char('1' + rowOf(m.a)));
    } else {
        s.push_back(char('a' + m.c));
        s.push_back(char('1' + m.b));
        s.push_back(m.a == 0 ? 'h' : 'v');
    }
    return s;
}

static bool parseMove(const State& s, const std::string& t, Move& out) {
    if (t.size() != 2 && t.size() != 3) return false;
    int c = t[0] - 'a', r = t[1] - '1';
    if (c < 0 || c >= N || r < 0 || r >= N) return false;
    Move x;
    if (t.size() == 2) x = Move::pawn(cellIdx(r, c));
    else {
        if (c >= WS || r >= WS || (t[2] != 'h' && t[2] != 'v')) return false;
        x = Move::wall(t[2] == 'h' ? 0 : 1, r, c);
    }
    MoveList legal = legalMoves(s);
    for (const Move& m : legal) if (m == x) { out = m; return true; }
    return false;
}

static std::vector<std::string> split(const std::string& line) {
    std::istringstream in(line);
    std::vector<std::string> out;
    std::string x;
    while (in >> x) out.push_back(x);
    return out;
}

static State replay(const std::vector<std::string>& hist) {
    State s = initialState();
    for (const auto& t : hist) {
        Move m;
        if (!parseMove(s, t, m)) {
            std::cerr << "illegal seed move " << t << "\n";
            std::exit(2);
        }
        s = applyMove(s, m);
    }
    return s;
}

static void emit(const std::string& id, int opening, const std::vector<std::string>& hist,
                 const char* splitName, const char* role) {
    State s = replay(hist);
    std::cout << "{\"schema\":\"zquoridor.position.v1\",\"id\":\"" << id
              << "\",\"side_to_move\":" << s.turn
              << ",\"opening_index\":" << opening
              << ",\"split\":\"" << splitName
              << "\",\"metadata\":{\"role\":\"" << role
              << "\"},\"history\":[";
    for (size_t i = 0; i < hist.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << "\"" << hist[i] << "\"";
    }
    std::cout << "]}\n";
}

int main() {
    struct Seed { int opening; const char* history; bool holdout; };
    const Seed seeds[] = {
        {28, "e2 e8 e3 e7 e4 e6 f4 f6 f5h e6 e4 e6v e3h e5 c3h d5h f4v e8v b4v b6h", false},
        {29, "e2 e8 e3 e7 e4 e6 d4 e5 d4v c5h e4h e5h g4h a5h g5v d2v g7v", false},
        {32, "e2 e8 e3 e7 e4 e6 f4 d6 e4v f6h", false},
        {27, "e2 e8 e3 e7 e4 e6 d4 d6 d5h", true},
    };

    for (const Seed& seed : seeds) {
        auto rootHist = split(seed.history);
        State root = replay(rootHist);
        std::string rootId = "sf" + std::to_string(seed.opening) + "_root";
        emit(rootId, seed.opening, rootHist, seed.holdout ? "val" : "train", "root");

        MoveList moves = legalMoves(root);
        int child = 0;
        for (const Move& m : moves) {
            State next = applyMove(root, m);
            if (winner(next) != -1) continue;
            auto hist = rootHist;
            hist.push_back(moveText(m));
            // Keep opening #27 entirely out of training.  For the three
            // regression roots, hold out every fifth legal continuation.
            const char* splitName = seed.holdout ? "val" : ((child % 5 == 0) ? "val" : "train");
            std::string id = "sf" + std::to_string(seed.opening) + "_c" + std::to_string(child);
            emit(id, seed.opening, hist, splitName, "one_ply");
            ++child;
        }
    }
    return 0;
}
