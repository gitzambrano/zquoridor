// Exact oracle for the optional wall-order shortest-path witness shortcut.
#include <cstdio>
#include <cstdint>
#include <random>
#include "rules.hpp"

int main() {
    using namespace qr;
    std::mt19937 rng(20261014u);
    uint64_t positions = 0, checked = 0, shortcut = 0;
    for (int game = 0; game < 80; ++game) {
        State state = initialState();
        for (int ply = 0; ply < 64 && winner(state) == -1; ++ply) {
            ++positions;
            for (int player = 0; player < 2; ++player) {
                PlayerPathCache cache;
                computeDistFull(state.wallsH, state.wallsV,
                                state.pawn[player], player, cache);
                const int before = cache.distToGoal;
                if (before < 0) {
                    std::fprintf(stderr, "Invalid legal state at game %d ply %d\n", game, ply);
                    return 1;
                }
                uint64_t h = 0, v = 0;
                cachedTouchSlots(cache, h, v);
                MoveList legalWalls;
                legalWallMoves(state, player, legalWalls);
                for (const Move& wall : legalWalls) {
                    ++checked;
                    const bool intersectsWitness =
                        (((wall.a == 0 ? h : v) >> slotIdx(wall.b, wall.c)) & 1ull) != 0;
                    if (intersectsWitness) continue;
                    ++shortcut;
                    const State after = applyMove(state, wall);
                    const int actual = shortestPathLen(after.wallsH, after.wallsV,
                                                       after.pawn[player], player);
                    if (actual != before) {
                        std::fprintf(stderr,
                                     "Witness shortcut mismatch: game=%d ply=%d player=%d before=%d actual=%d\n",
                                     game, ply, player, before, actual);
                        return 1;
                    }
                }
            }
            const MoveList moves = legalMoves(state);
            if (moves.empty()) break;
            // Exercise sparse and dense wall configurations without invalid moves.
            int index = std::uniform_int_distribution<int>(0, (int)moves.size() - 1)(rng);
            if (ply % 3 == 0) {
                for (size_t i = 0; i < moves.size(); ++i) {
                    if (moves[i].isWall) { index = (int)i; break; }
                }
            }
            state = applyMove(state, moves[(size_t)index]);
        }
    }
    std::printf("Wall-order shortcut oracle: positions=%llu, legal candidates=%llu, "
                "safe shortcuts=%llu, mismatches=0\n",
                (unsigned long long)positions,
                (unsigned long long)checked,
                (unsigned long long)shortcut);
    if (positions < 100 || shortcut < 100) return 2;
    return 0;
}
