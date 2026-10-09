#include "../src/search.hpp"
#include "../src/mcab.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace qr;

namespace {

bool sameState(const State& a, const State& b) {
    return a.pawn[0] == b.pawn[0] && a.pawn[1] == b.pawn[1] &&
           a.wallsH == b.wallsH && a.wallsV == b.wallsV &&
           a.wallsLeft[0] == b.wallsLeft[0] &&
           a.wallsLeft[1] == b.wallsLeft[1] &&
           a.turn == b.turn && a.hash == b.hash;
}

bool containsMove(const MoveList& moves, const Move& target) {
    for (const Move& m : moves) if (m == target) return true;
    return false;
}

void resolveAll(AccPair& ap) {
    resolvePending(ap, 0, nullptr);
    resolvePending(ap, 1, nullptr);
}

[[noreturn]] void mismatch(const char* label, int side, int index,
                           long long got, long long expected) {
    std::fprintf(stderr,
        "EDGE_ACC_PARITY_FAIL label=%s side=%d index=%d got=%lld expected=%lld\n",
        label, side, index, got, expected);
    std::exit(2);
}

void assertCanonicalEqual(AccPair a, AccPair b, const char* label) {
    resolveAll(a);
    resolveAll(b);
    for (int side = 0; side < 2; ++side) {
        if (a.pending[side] || b.pending[side])
            mismatch(label, side, -10, a.pending[side], b.pending[side]);
        const auto& x = a.acc[side];
        const auto& y = b.acc[side];
        if (x.ownDistBucket != y.ownDistBucket)
            mismatch(label, side, -1, x.ownDistBucket, y.ownDistBucket);
        if (x.oppDistBucket != y.oppDistBucket)
            mismatch(label, side, -2, x.oppDistBucket, y.oppDistBucket);
        if (x.ownWallsLeftBucket != y.ownWallsLeftBucket)
            mismatch(label, side, -3, x.ownWallsLeftBucket, y.ownWallsLeftBucket);
        if (x.oppWallsLeftBucket != y.oppWallsLeftBucket)
            mismatch(label, side, -4, x.oppWallsLeftBucket, y.oppWallsLeftBucket);
        for (int i = 0; i < HIDDEN; ++i) {
            if (x.v[(size_t)i] != y.v[(size_t)i])
                mismatch(label, side, i, x.v[(size_t)i], y.v[(size_t)i]);
        }
        const int vx = nnueEvalInt(a, side);
        const int vy = nnueEvalInt(b, side);
        if (vx != vy) mismatch(label, side, -5, vx, vy);

        std::array<float, POLICY_OUT> px{}, py{};
        forwardPolicyQuant(x, px);
        forwardPolicyQuant(y, py);
        for (int i = 0; i < POLICY_OUT; ++i) {
            if (px[(size_t)i] != py[(size_t)i]) {
                std::fprintf(stderr,
                    "EDGE_ACC_POLICY_FAIL label=%s side=%d index=%d got=%.9g expected=%.9g\n",
                    label, side, i, px[(size_t)i], py[(size_t)i]);
                std::exit(3);
            }
        }
    }
}

AccPair childPair(AccPair parent, const State& before, const Move& m) {
    AccPair child;
    makeChildAccPair(parent, child, before, m, nullptr);
    return child;
}

bool findCommutingWalls(const State& root, std::mt19937_64& rng,
                        Move& a, Move& b) {
    MoveList legal = legalMoves(root);
    std::vector<Move> walls;
    for (const Move& m : legal) if (m.isWall) walls.push_back(m);
    if (walls.size() < 2) return false;

    for (int attempt = 0; attempt < 200; ++attempt) {
        size_t ia = (size_t)(rng() % walls.size());
        size_t ib = (size_t)(rng() % walls.size());
        if (ia == ib) continue;
        Move ma = walls[ia], mb = walls[ib];

        State sa = applyMove(root, ma);
        if (!containsMove(legalMoves(sa), mb)) continue;
        State sb = applyMove(root, mb);
        if (!containsMove(legalMoves(sb), ma)) continue;

        State fa = applyMove(sa, mb);
        State fb = applyMove(sb, ma);
        if (!sameState(fa, fb)) continue;

        a = ma; b = mb;
        return true;
    }
    return false;
}

void verifyOneTransposition(const State& root, const Move& a, const Move& b,
                            int& continuationChecks) {
    AccPair rootA = buildAccPairRoot(root, nullptr);
    AccPair a1 = childPair(rootA, root, a);
    State sa = applyMove(root, a);
    AccPair a2 = childPair(a1, sa, b);
    State fa = applyMove(sa, b);

    AccPair rootB = buildAccPairRoot(root, nullptr);
    AccPair b1 = childPair(rootB, root, b);
    State sb = applyMove(root, b);
    AccPair b2 = childPair(b1, sb, a);
    State fb = applyMove(sb, a);

    if (!sameState(fa, fb)) {
        std::fprintf(stderr, "constructed paths are not a transposition\n");
        std::exit(4);
    }
    const uint64_t ka = mcab::mcabEvalStateKey(fa, 0);
    const uint64_t kb = mcab::mcabEvalStateKey(fb, 0);
    if (ka == 0 || ka != kb) {
        std::fprintf(stderr, "eval state key mismatch for equal states\n");
        std::exit(5);
    }

    // The lazy metadata is expected to be path-dependent here: the last move differs.
    // Correctness requires only that resolving it yields the same canonical state.
    AccPair cold = buildAccPairRoot(fa, nullptr);
    assertCanonicalEqual(a2, cold, "pathA-vs-cold");
    assertCanonicalEqual(b2, cold, "pathB-vs-cold");
    assertCanonicalEqual(a2, b2, "pathA-vs-pathB");

    // Emulate an EdgeAcc hit populated from path A while the search arrives through B.
    // Then continue the search one ply. If the cached lazy metadata is incompatible,
    // the next makeChildAccPair() is where it will normally become visible.
    MoveList next = legalMoves(fa);
    int checked = 0;
    for (const Move& c : next) {
        if (checked >= 8) break;
        AccPair cachedParent = a2;       // what EdgeAcc would return
        AccPair currentParent = b2;      // what the current path would have built
        AccPair fromCache = childPair(cachedParent, fa, c);
        AccPair fromCurrent = childPair(currentParent, fa, c);
        State child = applyMove(fa, c);
        AccPair coldChild = buildAccPairRoot(child, nullptr);

        assertCanonicalEqual(fromCache, coldChild, "cache-continuation-vs-cold");
        assertCanonicalEqual(fromCurrent, coldChild, "current-continuation-vs-cold");
        assertCanonicalEqual(fromCache, fromCurrent, "cache-vs-current-continuation");
        ++checked;
        ++continuationChecks;
    }
}

State randomReachableState(std::mt19937_64& rng, int maxPlies) {
    State s = initialState();
    int plies = (int)(rng() % (uint64_t)(maxPlies + 1));
    for (int ply = 0; ply < plies && winner(s) == -1; ++ply) {
        MoveList moves = legalMoves(s);
        if (moves.empty()) break;
        // Bias toward pawn moves sometimes so wall stock survives into deeper roots.
        std::vector<size_t> pawn, wall;
        for (size_t i = 0; i < moves.size(); ++i)
            (moves[i].isWall ? wall : pawn).push_back(i);
        size_t idx = 0;
        if (!pawn.empty() && ((rng() & 3ull) != 0ull))
            idx = pawn[(size_t)(rng() % pawn.size())];
        else
            idx = (size_t)(rng() % moves.size());
        s = applyMove(s, moves[idx]);
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights_int8.bin>\n", argv[0]);
        return 1;
    }
    if (!loadWeightsQuant(argv[1])) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    std::mt19937_64 rng(0xE6AACC2026ull);
    int transpositions = 0;
    int continuationChecks = 0;

    // Deterministic initial-state transposition first.
    {
        State root = initialState();
        Move a, b;
        if (!findCommutingWalls(root, rng, a, b)) {
            std::fprintf(stderr, "could not construct initial commuting-wall transposition\n");
            return 6;
        }
        verifyOneTransposition(root, a, b, continuationChecks);
        ++transpositions;
    }

    // Fuzz reachable positions. Every accepted case has two genuinely different
    // parent/move histories reaching exactly the same State.
    for (int rootNo = 0; rootNo < 1200 && transpositions < 5000; ++rootNo) {
        State root = randomReachableState(rng, 36);
        if (winner(root) != -1 || root.wallsLeft[0] <= 0 || root.wallsLeft[1] <= 0)
            continue;

        for (int k = 0; k < 6 && transpositions < 5000; ++k) {
            Move a, b;
            if (!findCommutingWalls(root, rng, a, b)) break;
            verifyOneTransposition(root, a, b, continuationChecks);
            ++transpositions;
        }
    }

    if (transpositions < 1000) {
        std::fprintf(stderr, "insufficient transposition coverage: %d\n", transpositions);
        return 7;
    }

    std::printf("EDGE_ACC_PARITY_OK transpositions=%d continuations=%d hidden=%d policy=%d\n",
                transpositions, continuationChecks, HIDDEN, POLICY_OUT);
    return 0;
}
