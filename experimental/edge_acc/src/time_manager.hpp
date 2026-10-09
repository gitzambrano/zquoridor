#pragma once

#include <algorithm>
#include <cstdint>

namespace zqtime {

struct TimeControl {
    std::int64_t remainingMs = 0;
    std::int64_t incrementMs = 0;
    int ply = 0;
    int movesToGo = 0;
    std::int64_t moveOverheadMs = 20;
};

struct TimeBudget {
    int optimumMs = 1;
    int maximumMs = 1;
};

inline int estimateMovesToGo(int ply) {
    int estimate = 30 - std::max(0, ply) / 4;
    return std::clamp(estimate, 10, 30);
}

inline TimeBudget allocate(const TimeControl& tc) {
    const std::int64_t overhead = std::max<std::int64_t>(0, tc.moveOverheadMs);
    const std::int64_t reserve = std::max<std::int64_t>(50, 3 * overhead);
    const std::int64_t safeRemaining = std::max<std::int64_t>(1, tc.remainingMs - reserve);
    const int movesToGo = tc.movesToGo > 0 ? std::max(1, tc.movesToGo)
                                            : estimateMovesToGo(tc.ply);

    const std::int64_t futureIncrement =
        std::max<std::int64_t>(0, tc.incrementMs) * std::max(0, movesToGo - 1);
    std::int64_t optimum =
        (safeRemaining + futureIncrement) / movesToGo - overhead;
    optimum = std::clamp<std::int64_t>(optimum, 1, safeRemaining);

    // Keep a larger bound available for a later adaptive search policy.
    // Version 1 uses optimumMs as the effective search limit.
    std::int64_t maximum =
        std::max<std::int64_t>(optimum, optimum * 3);
    maximum = std::min<std::int64_t>(maximum, safeRemaining);

    // When the clock is low, protect enough time for at least one more move.
    const std::int64_t lowClockThreshold =
        std::max<std::int64_t>(1000, 2 * std::max<std::int64_t>(0, tc.incrementMs) + reserve);
    if (tc.remainingMs <= lowClockThreshold) {
        const std::int64_t emergencyCap =
            std::max<std::int64_t>(1, safeRemaining / 3);
        optimum = std::min(optimum, emergencyCap);
        maximum = std::min(maximum, safeRemaining);
    }

    return {
        static_cast<int>(std::min<std::int64_t>(optimum, 0x7fffffffLL)),
        static_cast<int>(std::min<std::int64_t>(maximum, 0x7fffffffLL))
    };
}

}  // namespace zqtime
