// Generate reproducible legal random opening seeds for teacher-data collection.
// Output: one JSON object per line: {"moves":["e2",...]}
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rules.hpp"

static std::string moveToText(const qr::Move& m) {
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

static std::string key(const std::vector<std::string>& moves) {
    std::ostringstream ss;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i) ss << ' ';
        ss << moves[i];
    }
    return ss.str();
}

int main(int argc, char** argv) {
    int count = 64;
    int plies = 4;
    std::uint64_t seed = 20260911;
    if (argc > 1) count = std::atoi(argv[1]);
    if (argc > 2) plies = std::atoi(argv[2]);
    if (argc > 3) seed = std::strtoull(argv[3], nullptr, 10);
    if (count <= 0 || plies < 0 || plies > 24) {
        std::cerr << "usage: generate_openings [count>0] [plies 0..24] [seed]\n";
        return 2;
    }

    std::mt19937_64 rng(seed);
    std::set<std::string> seen;
    int attempts = 0;
    while ((int)seen.size() < count && attempts < count * 1000) {
        ++attempts;
        qr::State state = qr::initialState();
        std::vector<std::string> moves;
        bool ok = true;
        for (int ply = 0; ply < plies; ++ply) {
            if (qr::winner(state) != -1) { ok = false; break; }
            qr::MoveList legal = qr::legalMoves(state);
            if (legal.empty()) { ok = false; break; }
            std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
            const qr::Move m = legal[pick(rng)];
            moves.push_back(moveToText(m));
            state = qr::applyMove(state, m);
        }
        if (!ok) continue;
        const std::string k = key(moves);
        if (!seen.insert(k).second) continue;
        std::cout << "{\"moves\":[";
        for (size_t i = 0; i < moves.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '\"' << moves[i] << '\"';
        }
        std::cout << "]}\n";
    }
    if ((int)seen.size() != count) {
        std::cerr << "generated only " << seen.size() << " unique openings\n";
        return 3;
    }
    return 0;
}
