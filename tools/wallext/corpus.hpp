#pragma once
// corpus.hpp -- deterministic corpus of low-wall positions for the wall
// quiescence extension experiments (inv/qsendgame-ext, 2026-08).
//
// The corpus feeds every tool of this experiment: the correctness test,
// the nodes-to-depth benchmark, and the pairwise match runner. All tools
// include this header and regenerate the same positions in-process from
// fixed seeds, so there is no data file to keep in sync.
//
// Selection rule: play random legal games from the initial position. Keep
// every visited state that satisfies all of:
//   1. total walls left (both players) between MIN_TOTAL and MAX_TOTAL,
//   2. side to move has at least one wall (a horizon node for this player
//      can then extend a critical wall),
//   3. both pawns have a path to their goal,
//   4. not already a finished game.
// A candidate needs one more condition: either both players hold at least
// one wall, or the position shows tension (some pawn has path robustness
// <= TENSION_MAX, which means few cheap detours remain). Per total-walls
// bucket the PER_BUCKET candidates with the lowest tension score win; the
// rest are discarded. The tension score is the minimum path robustness
// over both players.
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include "rules.hpp"

namespace wallext {

constexpr int CORPUS_MIN_TOTAL = 2;
constexpr int CORPUS_MAX_TOTAL = 10;   // covers thresholds up to 6 plus margin
constexpr int CORPUS_PER_BUCKET = 40;  // 40 x 9 buckets = up to 360 positions
constexpr int CORPUS_TENSION_MAX = 3;

struct CorpusEntry {
    qr::State s;
    int seed;        // game that produced the position
    int ply;         // ply inside that game
    int tension;     // min pathRobustness over both players
};

// One random legal game; calls visit(state, ply) for every state reached.
template <typename Visit>
void randomGame(int seed, int maxPly, Visit&& visit) {
    std::mt19937 rng((uint32_t)seed);
    qr::State s = qr::initialState();
    for (int ply = 0; ply < maxPly; ply++) {
        if (qr::winner(s) != -1) return;
        visit(s, ply);
        qr::MoveList moves = qr::legalMoves(s);
        if (moves.empty()) return;
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        s = qr::applyMove(s, moves[pick(rng)]);
    }
}

inline int corpusTension(const qr::State& s) {
    int r0 = qr::pathRobustness(s.wallsH, s.wallsV, s.pawn[0], 0);
    int r1 = qr::pathRobustness(s.wallsH, s.wallsV, s.pawn[1], 1);
    return std::min(r0, r1);
}

inline bool samePosition(const qr::State& a, const qr::State& b) {
    return a.pawn[0] == b.pawn[0] && a.pawn[1] == b.pawn[1]
        && a.wallsH == b.wallsH && a.wallsV == b.wallsV
        && a.wallsLeft[0] == b.wallsLeft[0] && a.wallsLeft[1] == b.wallsLeft[1]
        && a.turn == b.turn;
}

inline bool corpusEligible(const qr::State& s) {
    if (qr::winner(s) != -1) return false;
    int total = s.wallsLeft[0] + s.wallsLeft[1];
    if (total < CORPUS_MIN_TOTAL || total > CORPUS_MAX_TOTAL) return false;
    if (s.wallsLeft[s.turn] < 1) return false;
    if (qr::shortestPathLen(s.wallsH, s.wallsV, s.pawn[0], 0) < 0) return false;
    if (qr::shortestPathLen(s.wallsH, s.wallsV, s.pawn[1], 1) < 0) return false;
    // Either balanced wall reserves, or visible tactical tension.
    bool balanced = s.wallsLeft[0] >= 1 && s.wallsLeft[1] >= 1;
    if (!balanced && corpusTension(s) > CORPUS_TENSION_MAX) return false;
    return true;
}

inline std::vector<CorpusEntry> buildCorpus(int numGames = 4000, int maxPly = 120) {
    std::vector<std::vector<CorpusEntry>> buckets(CORPUS_MAX_TOTAL + 1);
    for (int seed = 0; seed < numGames; seed++) {
        randomGame(seed, maxPly, [&](const qr::State& s, int ply) {
            if (!corpusEligible(s)) return;
            int total = s.wallsLeft[0] + s.wallsLeft[1];
            buckets[total].push_back({s, seed, ply, corpusTension(s)});
        });
    }
    std::vector<CorpusEntry> out;
    for (int total = CORPUS_MIN_TOTAL; total <= CORPUS_MAX_TOTAL; total++) {
        auto& b = buckets[total];
        // Lowest tension first; stable order keeps the result deterministic.
        std::stable_sort(b.begin(), b.end(),
                         [](const CorpusEntry& x, const CorpusEntry& y) { return x.tension < y.tension; });
        int kept = 0;
        for (const CorpusEntry& e : b) {
            if (kept >= CORPUS_PER_BUCKET) break;
            bool dup = false;
            for (const CorpusEntry& k : out) {
                if (samePosition(k.s, e.s)) { dup = true; break; }
            }
            if (dup) continue;
            out.push_back(e);
            kept++;
        }
    }
    return out;
}

}  // namespace wallext
