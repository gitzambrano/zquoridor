// Deterministic correctness checks for experimental quiet-wall root priors.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;

int main() {
    std::mt19937 rng(20261015);
    int evaluated = 0;
    for (int game = 0; game < 18; ++game) {
        State s = initialState();
        for (int ply = 0; ply < 10; ++ply) {
            if (winner(s) != -1) break;
            MoveList legal = legalMoves(s);
            assert(!legal.empty());
            // Independent shortest path metric must match production BFS.
            for (int side = 0; side < 2; ++side) {
                QuietPathMetric q = quietPathMetric(s.wallsH, s.wallsV, s.pawn[side], side);
                assert(q.distance == shortestPathLen(s.wallsH, s.wallsV, s.pawn[side], side));
                assert(q.shortestPaths >= 1.0 && q.corridor >= 0);
            }
            std::vector<float> prior(legal.size(), 1.0f / (float)legal.size());
            const auto a = quietWallScores(s, legal, prior, 1);
            assert(a.size() == legal.size());
            for (size_t i = 0; i < a.size(); ++i) {
                assert(std::isfinite(a[i]) && a[i] >= 0 && a[i] <= 1.0f);
                if (a[i] == 0) continue;
                assert(legal[i].isWall);
                State child = applyMove(s, legal[i]);
                const int me = s.turn, opp = 1 - me;
                assert(shortestPathLen(child.wallsH, child.wallsV, child.pawn[opp], opp)
                       == shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp));
                assert(shortestPathLen(child.wallsH, child.wallsV, child.pawn[me], me)
                       == shortestPathLen(s.wallsH, s.wallsV, s.pawn[me], me));
            }
            // All variants preserve prior normalization and nonnegative mass.
            for (int mode = 0; mode <= 3; ++mode) {
                auto result = prior;
                quietWallReweight(s, legal, result, mode, 0.08);
                double sum = 0;
                for (float p : result) {
                    assert(std::isfinite(p) && p >= 0.0f && p <= 1.0f);
                    sum += p;
                }
                assert(std::abs(sum - 1.0) < 1e-4);
                if (mode == 0) assert(result == prior);
            }
            ++evaluated;
            const size_t chosen = (size_t)(rng() % legal.size());
            s = applyMove(s, legal[chosen]);
        }
    }

    // The second-wall setup signal must be disabled with fewer than two walls.
    State constrained = initialState();
    constrained.wallsLeft[0] = 1;
    MoveList legal = legalMoves(constrained);
    std::vector<float> p(legal.size(), 1.0f / (float)legal.size());
    const auto b = quietWallScores(constrained, legal, p, 2);
    for (float value : b) assert(value == 0.0f);
    std::printf("quiet-wall tests passed: %d states; all modes and invariants checked\n", evaluated);
    return 0;
}
