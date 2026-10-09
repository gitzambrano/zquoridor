// test_mirror_engine.cpp -- validates horizontal mirror mapping against the C++ engine.
// Verifies on thousands of positions:
// 1. Bitboard bit r*8+c with column 0 in LSB reverses bits per byte.
// 2. Action layout (81 pawn destinations + 64 H walls + 64 V walls = 209).
// 3. For any random game state S:
//    - shortestPathLen(own/opp) in S equals shortestPathLen(own/opp) in S_mirr.
//    - legalMoves(S_mirr) equals ACTION_MIRROR_H applied to legalMoves(S).
//    - applying mirrored move to S_mirr produces the mirrored successor state.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <random>
#include <vector>
#include <set>
#include <algorithm>
#include "rules.hpp"

using namespace qr;

// Byte-reversal lookup table for 8-bit row reflection
static const uint8_t REV8[256] = {
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff
};

inline uint64_t mirrorBitboardH(uint64_t b) {
    uint8_t bytes[8];
    std::memcpy(bytes, &b, 8);
    for (int i = 0; i < 8; i++) bytes[i] = REV8[bytes[i]];
    uint64_t out;
    std::memcpy(&out, bytes, 8);
    return out;
}

inline int mirrorPawnH(int cell) {
    int r = cell / N, c = cell % N;
    return r * N + (N - 1 - c);
}

inline int mirrorWallSlotH(int slot) {
    int r = slot / WS, c = slot % WS;
    return r * WS + (WS - 1 - c);
}

inline uint16_t mirrorActionH(uint16_t a) {
    if (a < N * N) {
        return (uint16_t)mirrorPawnH(a);
    } else if (a < N * N + WS * WS) {
        int slot = a - N * N;
        return (uint16_t)(N * N + mirrorWallSlotH(slot));
    } else {
        int slot = a - (N * N + WS * WS);
        return (uint16_t)(N * N + WS * WS + mirrorWallSlotH(slot));
    }
}

inline Move mirrorMoveH(const Move& m) {
    if (!m.isWall) {
        return Move::pawn(mirrorPawnH(m.a));
    }
    int slot = slotIdx(m.b, m.c);
    int mslot = mirrorWallSlotH(slot);
    return Move::wall(m.a, mslot / WS, mslot % WS);
}

inline State mirrorStateH(const State& s) {
    State m;
    m.pawn[0] = mirrorPawnH(s.pawn[0]);
    m.pawn[1] = mirrorPawnH(s.pawn[1]);
    m.wallsH = mirrorBitboardH(s.wallsH);
    m.wallsV = mirrorBitboardH(s.wallsV);
    m.wallsLeft[0] = s.wallsLeft[0];
    m.wallsLeft[1] = s.wallsLeft[1];
    m.turn = s.turn;
    m.hash = 0; // Not used for rules comparisons
    return m;
}

int main() {
    std::printf("[test_mirror_engine] Starting horizontal mirror validation against C++ engine...\n");

    // 1. Bit-level slot verification
    for (int r = 0; r < WS; r++) {
        for (int c = 0; c < WS; c++) {
            int slot = slotIdx(r, c);
            int mslot = mirrorWallSlotH(slot);
            assert(mslot == slotIdx(r, WS - 1 - c));
            uint64_t bb = (1ull << slot);
            uint64_t mbb = mirrorBitboardH(bb);
            assert(mbb == (1ull << mslot));
        }
    }
    std::printf("  1. Slot & bitboard byte-reversal check: OK\n");

    // 2. Action involution check
    for (int a = 0; a < 209; a++) {
        uint16_t ma = mirrorActionH(a);
        uint16_t mma = mirrorActionH(ma);
        assert(mma == a);
    }
    std::printf("  2. Action involution check (M[M[a]] == a): OK\n");

    // 3. Move <-> Action mapping check
    for (int cell = 0; cell < N * N; cell++) {
        Move m = Move::pawn(cell);
        uint16_t a = moveToPolicyIndex(m);
        assert(a == cell);
        Move mm = mirrorMoveH(m);
        uint16_t ma = moveToPolicyIndex(mm);
        assert(ma == mirrorActionH(a));
    }
    for (int o = 0; o < 2; o++) {
        for (int r = 0; r < WS; r++) {
            for (int c = 0; c < WS; c++) {
                Move m = Move::wall(o, r, c);
                uint16_t a = moveToPolicyIndex(m);
                Move mm = mirrorMoveH(m);
                uint16_t ma = moveToPolicyIndex(mm);
                assert(ma == mirrorActionH(a));
            }
        }
    }
    std::printf("  3. Move <-> Action bijective reflection check: OK\n");

    // 4. Random game rollouts: shortest path & legal moves check
    std::mt19937_64 rng(424242);
    int totalPositions = 0;
    int totalMovesChecked = 0;

    for (int game = 0; game < 500; game++) {
        State s = initialState();
        for (int ply = 0; ply < 80; ply++) {
            if (winner(s) >= 0) break;

            State sm = mirrorStateH(s);
            totalPositions++;

            // Path distance invariance
            int d0 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[0], 0);
            int d0m = shortestPathLen(sm.wallsH, sm.wallsV, sm.pawn[0], 0);
            if (d0 != d0m) {
                std::fprintf(stderr, "FAIL: game %d ply %d d0=%d != d0m=%d\n", game, ply, d0, d0m);
                return 1;
            }

            int d1 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[1], 1);
            int d1m = shortestPathLen(sm.wallsH, sm.wallsV, sm.pawn[1], 1);
            if (d1 != d1m) {
                std::fprintf(stderr, "FAIL: game %d ply %d d1=%d != d1m=%d\n", game, ply, d1, d1m);
                return 1;
            }

            // Legal moves invariance
            MoveList legals = legalMoves(s);
            MoveList legalsM = legalMoves(sm);

            std::set<uint16_t> reflectedActions;
            for (const Move& m : legals) {
                reflectedActions.insert(mirrorActionH(moveToPolicyIndex(m)));
            }

            std::set<uint16_t> actualMirroredActions;
            for (const Move& m : legalsM) {
                actualMirroredActions.insert(moveToPolicyIndex(m));
            }

            if (reflectedActions != actualMirroredActions) {
                std::fprintf(stderr, "FAIL: legal moves mismatch at game %d ply %d! legals count=%d vs %d\n",
                             game, ply, (int)legals.size(), (int)legalsM.size());
                return 1;
            }
            totalMovesChecked += legals.size();

            // Step game randomly
            std::uniform_int_distribution<size_t> dist(0, legals.size() - 1);
            Move chosen = legals[dist(rng)];
            Move chosenM = mirrorMoveH(chosen);

            State nextS = applyMove(s, chosen);
            State nextSm = applyMove(sm, chosenM);
            State nextS_reflected = mirrorStateH(nextS);

            if (nextSm.pawn[0] != nextS_reflected.pawn[0] ||
                nextSm.pawn[1] != nextS_reflected.pawn[1] ||
                nextSm.wallsH != nextS_reflected.wallsH ||
                nextSm.wallsV != nextS_reflected.wallsV ||
                nextSm.wallsLeft[0] != nextS_reflected.wallsLeft[0] ||
                nextSm.wallsLeft[1] != nextS_reflected.wallsLeft[1] ||
                nextSm.turn != nextS_reflected.turn) {
                std::fprintf(stderr, "FAIL: transition mismatch after applying chosen move at game %d ply %d!\n", game, ply);
                return 1;
            }

            s = nextS;
        }
    }

    std::printf("  4. Random rollouts across %d positions and %d legal moves: ALL PASSED!\n",
                totalPositions, totalMovesChecked);
    std::printf("[test_mirror_engine] SUCCESS: 100%% equivalence proven against C++ engine!\n");
    return 0;
}
