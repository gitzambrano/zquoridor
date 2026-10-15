// Quiet-wall exploration signals for root priors. Experimental, opt-in.
// A: reduction in near-shortest-path corridor redundancy.
// B: increased strongest legal follow-up wall threat (a setup proxy, NOT a
// forced two-ply tactic: an opponent reply may neutralize the threat).
#pragma once

#include "rules.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace qr {
struct QuietPathMetric {
    int distance = -1;
    int corridor = 0;
    double shortestPaths = 0.0;
};

// Forward-only corridor edges avoid counting backtracking detours. Both the
// shortest-path count and the near-shortest corridor are computed from the
// actual wall graph. The pawn collision rule is deliberately excluded, as in
// shortestPathLen(), since this is a topology-only prior.
inline QuietPathMetric quietPathMetric(uint64_t h, uint64_t v, int start, int side) {
    constexpr int cells = N * N;
    int ds[cells], dg[cells], q[cells];
    double ways[cells] = {};
    std::fill(ds, ds + cells, -1);
    std::fill(dg, dg + cells, -1);
    int head = 0, tail = 0;
    ds[start] = 0;
    ways[start] = 1.0;
    q[tail++] = start;
    while (head < tail) {
        int u = q[head++];
        for (int dir = 0; dir < 4; ++dir) {
            int n = ORTH_NEIGHBOR[u][dir];
            if (n < 0 || edgeBlockedDir(h, v, u, dir)) continue;
            if (ds[n] == -1) {
                ds[n] = ds[u] + 1;
                q[tail++] = n;
            }
            if (ds[n] == ds[u] + 1) ways[n] += ways[u];
        }
    }
    head = tail = 0;
    for (int col = 0; col < N; ++col) {
        int g = cellIdx(GOAL_ROW[side], col);
        dg[g] = 0;
        q[tail++] = g;
    }
    while (head < tail) {
        int u = q[head++];
        for (int dir = 0; dir < 4; ++dir) {
            int n = ORTH_NEIGHBOR[u][dir];
            if (n < 0 || dg[n] != -1 || edgeBlockedDir(h, v, u, dir)) continue;
            dg[n] = dg[u] + 1;
            q[tail++] = n;
        }
    }
    QuietPathMetric out;
    out.distance = dg[start];
    if (out.distance < 0) return out;
    for (int col = 0; col < N; ++col) {
        int g = cellIdx(GOAL_ROW[side], col);
        if (ds[g] == out.distance) out.shortestPaths += ways[g];
    }
    for (int u = 0; u < cells; ++u) {
        if (ds[u] < 0) continue;
        for (int dir = 0; dir < 4; ++dir) {
            int n = ORTH_NEIGHBOR[u][dir];
            if (n < 0 || dg[n] < 0 || ds[n] != ds[u] + 1 ||
                edgeBlockedDir(h, v, u, dir)) continue;
            int slack = ds[u] + 1 + dg[n] - out.distance;
            if (slack == 0) out.corridor += 4;
            else if (slack <= 2) out.corridor += 1;
        }
    }
    return out;
}

inline int quietBestSingleWallGain(const State& s, int side, int baselineDistance) {
    if (s.wallsLeft[side] <= 0) return 0;
    int opponent = 1 - side, best = 0;
    for (int orient = 0; orient < 2; ++orient) {
        for (int r = 0; r < WS; ++r) {
            for (int c = 0; c < WS; ++c) {
                if (!isWallMoveLegal(s, side, orient, r, c)) continue;
                uint64_t h = s.wallsH, v = s.wallsV;
                const uint64_t bit = 1ull << slotIdx(r, c);
                if (orient == 0) h |= bit;
                else v |= bit;
                int d = shortestPathLen(h, v, s.pawn[opponent], opponent);
                best = std::max(best, d - baselineDistance);
                if (best >= 4) return 4; // bounded proxy, not a proof of best play
            }
        }
    }
    return best;
}

inline std::vector<float> quietWallScores(const State& state,
                                          const MoveList& legal,
                                          const std::vector<float>& priors,
                                          int mode) {
    const size_t count = legal.size();
    std::vector<float> signal(count, 0.f);
    if (mode < 1 || mode > 3 || state.wallsLeft[state.turn] < 1 ||
        priors.size() != count) return signal;

    const int me = state.turn, opp = 1 - me;
    const auto beforeOpp = quietPathMetric(state.wallsH, state.wallsV, state.pawn[opp], opp);
    const auto beforeOwn = quietPathMetric(state.wallsH, state.wallsV, state.pawn[me], me);
    if (beforeOpp.distance < 0 || beforeOwn.distance < 0) return signal;

    struct Candidate { size_t index; float flexibility; State next; };
    std::vector<Candidate> quiet;
    quiet.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const Move& m = legal[i];
        if (!m.isWall) continue;
        State next = applyMove(state, m);
        const auto afterOpp = quietPathMetric(next.wallsH, next.wallsV, next.pawn[opp], opp);
        if (afterOpp.distance != beforeOpp.distance) continue;
        const auto afterOwn = quietPathMetric(next.wallsH, next.wallsV, next.pawn[me], me);
        if (afterOwn.distance != beforeOwn.distance) continue;

        const double oppEdges = (double)(beforeOpp.corridor - afterOpp.corridor) /
                                std::max(1, beforeOpp.corridor);
        const double ownEdges = (double)(beforeOwn.corridor - afterOwn.corridor) /
                                std::max(1, beforeOwn.corridor);
        const double oppPaths = std::log((1.0 + beforeOpp.shortestPaths) /
                                          (1.0 + afterOpp.shortestPaths));
        const double ownPaths = std::log((1.0 + beforeOwn.shortestPaths) /
                                          (1.0 + afterOwn.shortestPaths));
        float redundancy = (float)std::clamp(
            0.85 * oppEdges + 0.15 * oppPaths -
            0.50 * ownEdges - 0.10 * ownPaths, 0.0, 1.0);
        quiet.push_back({i, redundancy, next});
        if (mode & 1) signal[i] = redundancy;
    }

    if ((mode & 2) && state.wallsLeft[me] >= 2 && !quiet.empty()) {
        // Evaluate only eight promising quiet first walls, sorted by corridor
        // change and NNUE prior. The second placement is legal but assumes the
        // opponent passes; search remains responsible for verifying replies.
        std::stable_sort(quiet.begin(), quiet.end(),
            [&](const Candidate& a, const Candidate& b) {
                return a.flexibility + 0.3f * priors[a.index] >
                       b.flexibility + 0.3f * priors[b.index];
            });
        const int directGain = quietBestSingleWallGain(state, me, beforeOpp.distance);
        for (size_t k = 0; k < std::min<size_t>(8, quiet.size()); ++k) {
            const auto& cand = quiet[k];
            int followGain = quietBestSingleWallGain(cand.next, me, beforeOpp.distance);
            float setup = (float)std::clamp((followGain - directGain) / 3.0, 0.0, 1.0);
            signal[cand.index] += setup;
        }
    }
    return signal;
}

inline void quietWallReweight(const State& s, const MoveList& moves,
                              std::vector<float>& priors, int mode, double strength) {
    if (mode < 1 || mode > 3 || strength <= 0.0 || priors.size() != moves.size()) return;
    const auto score = quietWallScores(s, moves, priors, mode);
    double total = 0.0;
    for (float x : score) total += x;
    if (total <= 1e-9) return;
    const double blend = std::clamp(strength, 0.0, 0.20);
    for (size_t i = 0; i < priors.size(); ++i)
        priors[i] = (float)((1.0 - blend) * priors[i] + blend * score[i] / total);
}
} // namespace qr
