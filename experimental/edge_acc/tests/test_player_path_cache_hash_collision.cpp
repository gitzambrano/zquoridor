#include "../src/rules.hpp"
#include <cstdio>
#include <cstring>

using namespace qr;

int main(int argc, char** argv) {
    const bool expectKnownAlias = argc == 2 &&
        std::strcmp(argv[1], "--expect-known-alias") == 0;
    if (argc > 1 && !expectKnownAlias) {
        std::fprintf(stderr, "usage: %s [--expect-known-alias]\n", argv[0]);
        return 1;
    }

    constexpr uint64_t corner = 1ull << 63;
    constexpr int start = cellIdx(7, 7);
    constexpr int player = 0;
    constexpr uint64_t empty = 0;

    const uint64_t keyH = playerPathCacheKey(corner, empty, start, player);
    const uint64_t keyV = playerPathCacheKey(empty, corner, start, player);
    if (keyH != keyV) {
        std::fprintf(stderr, "known alias no longer occurs: %016llx != %016llx\n",
                     (unsigned long long)keyH, (unsigned long long)keyV);
        return 2;
    }

    PlayerPathCache expectedH, expectedV;
    computeDistFull(corner, empty, start, player, expectedH);
    computeDistFull(empty, corner, start, player, expectedV);
    if (expectedH.distToGoal == expectedV.distToGoal) {
        std::fprintf(stderr, "test setup has equal BFS results: %d\n",
                     expectedH.distToGoal);
        return 2;
    }

    PlayerPathCacheTable table;
    PlayerPathCache cachedH, cachedV;
    computeDistCached(corner, empty, start, player, &table, cachedH);
    computeDistCached(empty, corner, start, player, &table, cachedV);
    if (cachedV.distToGoal != expectedV.distToGoal) {
        std::fprintf(stderr,
            "KNOWN_BASELINE_DEFECT path-cache-key-alias key=%016llx "
            "expected=%d got=%d hits=%llu\n",
            (unsigned long long)keyV, expectedV.distToGoal, cachedV.distToGoal,
            (unsigned long long)table.hits());
        return expectKnownAlias ? 0 : 3;
    }

    std::printf("PLAYER_PATH_CACHE_ALIAS_FIXED distance=%d hits=%llu\n",
                cachedV.distToGoal, (unsigned long long)table.hits());
    return expectKnownAlias ? 4 : 0;
}
