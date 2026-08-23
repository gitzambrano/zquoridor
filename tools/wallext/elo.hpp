// elo.hpp -- Elo mean and confidence margin for pairwise matches.
// Same formula family as calculate_elo_and_ci in tools/arena/run_arena.py:
// the score maps to Elo by log-odds, and the 95% margin comes from the
// binomial variance of the per-game results.
#pragma once
#include <cmath>

inline double wallextEloFromScore(double score) {
    if (score <= 0.0) return -800.0;
    if (score >= 1.0) return 800.0;
    return -400.0 * std::log10(1.0 / score - 1.0);
}

inline void wallextEloWithMargin(int winsB, int winsA, int draws,
                                 double& eloDiff, double& margin) {
    int total = winsA + winsB + draws;
    if (total == 0) { eloDiff = 0.0; margin = 0.0; return; }
    double score = (winsB + 0.5 * draws) / total;
    eloDiff = wallextEloFromScore(score);
    double pW = (double)winsB / total;
    double pL = (double)winsA / total;
    double pD = (double)draws / total;
    double variance = pW * (1.0 - score) * (1.0 - score)
                    + pL * (0.0 - score) * (0.0 - score)
                    + pD * (0.5 - score) * (0.5 - score);
    double stdError = total > 1 ? std::sqrt(variance / total) : 0.0;
    if (score > 0.0 && score < 1.0) {
        double factor = 400.0 / (std::log(10.0) * score * (1.0 - score));
        margin = 1.96 * stdError * factor;
    } else {
        margin = 0.0;
    }
}
