#include "../src/nnue.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace qr;

namespace {

[[noreturn]] void fail(const char* label, int side, int index,
                       long long got, long long expected) {
    std::fprintf(stderr,
        "EDGE_FEATURE_DELTA_FAIL label=%s side=%d index=%d got=%lld expected=%lld\n",
        label, side, index, got, expected);
    std::exit(2);
}

void sameAccumulator(const AccumulatorQuant& a, const AccumulatorQuant& b,
                     const char* label, int side) {
    if (a.ownDistBucket != b.ownDistBucket)
        fail(label, side, -1, a.ownDistBucket, b.ownDistBucket);
    if (a.oppDistBucket != b.oppDistBucket)
        fail(label, side, -2, a.oppDistBucket, b.oppDistBucket);
    if (a.ownWallsLeftBucket != b.ownWallsLeftBucket)
        fail(label, side, -3, a.ownWallsLeftBucket, b.ownWallsLeftBucket);
    if (a.oppWallsLeftBucket != b.oppWallsLeftBucket)
        fail(label, side, -4, a.oppWallsLeftBucket, b.oppWallsLeftBucket);
    for (int i = 0; i < HIDDEN; ++i)
        if (a.v[(size_t)i] != b.v[(size_t)i])
            fail(label, side, i, a.v[(size_t)i], b.v[(size_t)i]);
}

bool sameState(const State& a, const State& b) {
    return a.wallsH == b.wallsH && a.wallsV == b.wallsV &&
           a.pawn[0] == b.pawn[0] && a.pawn[1] == b.pawn[1] &&
           a.wallsLeft[0] == b.wallsLeft[0] &&
           a.wallsLeft[1] == b.wallsLeft[1] && a.turn == b.turn &&
           a.hash == b.hash;
}

struct RecordingAccumulator : AccumulatorQuant {
    std::vector<int> operations;

    void addFeature(int feature) {
        operations.push_back(feature + 1);
        AccumulatorQuant::addFeature(feature);
    }

    void removeFeature(int feature) {
        operations.push_back(-feature - 1);
        AccumulatorQuant::removeFeature(feature);
    }
};

void checkDeltaBound(const State& before, const Move& move,
                     bool viewerIsMover, PlayerPathCacheTable* paths) {
    const int perspective = viewerIsMover ? before.turn : 1 - before.turn;
    RecordingAccumulator recorder;
    static_cast<AccumulatorQuant&>(recorder) =
        buildAccumulatorQuant(before, perspective, paths);
    updateAccumulatorForMoveQuantKernel(recorder, viewerIsMover, before, move, paths);
    assert(recorder.operations.size() <= 64);
    if (recorder.operations.size() > 64)
        fail("delta-operation-bound", perspective,
             (int)recorder.operations.size(), (long long)recorder.operations.size(), 64);
}

void verifyMove(const State& before, const Move& move, const char* label,
                PlayerPathCacheTable* paths, int& comparisons) {
    PlayerPathCacheTable* pathModes[] = {nullptr, paths};
    for (int pathMode = 0; pathMode < 2; ++pathMode) {
        PlayerPathCacheTable* activePaths = pathModes[pathMode];
        for (int viewerIsMover = 0; viewerIsMover <= 1; ++viewerIsMover) {
            const int perspective = viewerIsMover ? before.turn : 1 - before.turn;
            AccumulatorQuant reference =
                buildAccumulatorQuant(before, perspective, activePaths);
            AccumulatorQuant candidate = reference;
            updateAccumulatorForMoveQuantKernel(reference, viewerIsMover != 0,
                                                before, move, activePaths);
            updateAccumulatorForMoveQuant(candidate, viewerIsMover != 0,
                                          before, move, activePaths);
            sameAccumulator(reference, candidate, label, perspective);
            ++comparisons;

            // The cached object represents an ordered transformation. Apply
            // the same edge to another live vector to reject accumulator snapshots.
            AccumulatorQuant shiftedReference =
                buildAccumulatorQuant(before, perspective, activePaths);
            AccumulatorQuant shiftedCandidate = shiftedReference;
            for (int i = 0; i < HIDDEN; ++i) {
                const int32_t shift =
                    (int32_t)((i * 17 + perspective * 29) % 101) - 50;
                shiftedReference.v[(size_t)i] += shift;
                shiftedCandidate.v[(size_t)i] += shift;
            }
            updateAccumulatorForMoveQuantKernel(shiftedReference, viewerIsMover != 0,
                                                before, move, activePaths);
            updateAccumulatorForMoveQuant(shiftedCandidate, viewerIsMover != 0,
                                          before, move, activePaths);
            sameAccumulator(shiftedReference, shiftedCandidate,
                            "altered-live-vector", perspective);
            ++comparisons;
            checkDeltaBound(before, move, viewerIsMover != 0, activePaths);
        }
    }
}

void resolveReference(AccPair& pair, int perspective,
                      PlayerPathCacheTable* paths) {
    if (!pair.pending[perspective]) return;
    updateAccumulatorForMoveQuantKernel(
        pair.acc[perspective], pair.pendViewerIsMover[perspective],
        pair.pendBefore[perspective], pair.pendMove[perspective], paths);
    pair.pending[perspective] = false;
}

void makeChildReference(AccPair& parent, AccPair& child, const State& before,
                        const Move& move, PlayerPathCacheTable* paths) {
    const int mover = before.turn;
    const int opponent = 1 - mover;
    resolveReference(parent, opponent, paths);
    child.acc[opponent] = parent.acc[opponent];
    child.pending[opponent] = false;
    updateAccumulatorForMoveQuantKernel(child.acc[opponent], false, before, move, paths);
    child.acc[mover] = parent.acc[mover];
    child.pending[mover] = true;
    child.pendBefore[mover] = before;
    child.pendMove[mover] = move;
    child.pendViewerIsMover[mover] = true;
}

void samePair(const AccPair& a, const AccPair& b, const char* label) {
    for (int side = 0; side < 2; ++side) {
        sameAccumulator(a.acc[side], b.acc[side], label, side);
        if (a.pending[side] != b.pending[side])
            fail(label, side, -10, a.pending[side], b.pending[side]);
        if (a.pending[side] &&
            (!sameState(a.pendBefore[side], b.pendBefore[side]) ||
             !(a.pendMove[side] == b.pendMove[side]) ||
             a.pendViewerIsMover[side] != b.pendViewerIsMover[side]))
            fail(label, side, -11, 1, 0);
    }
}

State randomReachableState(std::mt19937_64& rng, int plies) {
    State state = initialState();
    for (int ply = 0; ply < plies && winner(state) == -1; ++ply) {
        MoveList moves = legalMoves(state);
        if (moves.empty()) break;
        std::vector<size_t> pawns, walls;
        for (size_t i = 0; i < moves.size(); ++i)
            (moves[i].isWall ? walls : pawns).push_back(i);
        size_t selected = 0;
        if (!pawns.empty() && (rng() % 4 != 0))
            selected = pawns[(size_t)(rng() % pawns.size())];
        else if (!walls.empty())
            selected = walls[(size_t)(rng() % walls.size())];
        else
            selected = (size_t)(rng() % moves.size());
        state = applyMove(state, moves[selected]);
    }
    return state;
}

bool containsMove(const MoveList& moves, const Move& target) {
    for (const Move& move : moves)
        if (move == target) return true;
    return false;
}

void verifyTransposedPaths(int& comparisons, PlayerPathCacheTable* paths) {
    State root = initialState();
    MoveList rootMoves = legalMoves(root);
    std::vector<Move> walls;
    for (const Move& move : rootMoves)
        if (move.isWall) walls.push_back(move);

    int transpositions = 0;
    for (size_t i = 0; i < walls.size() && transpositions < 32; ++i) {
        for (size_t j = i + 1; j < walls.size() && transpositions < 32; ++j) {
            const Move first = walls[i], second = walls[j];
            const State afterFirst = applyMove(root, first);
            const State afterSecond = applyMove(root, second);
            if (!containsMove(legalMoves(afterFirst), second) ||
                !containsMove(legalMoves(afterSecond), first))
                continue;
            const State finalA = applyMove(afterFirst, second);
            const State finalB = applyMove(afterSecond, first);
            if (!sameState(finalA, finalB)) continue;

            AccPair candidateA = buildAccPairRoot(root, paths);
            AccPair referenceA = candidateA;
            AccPair childA, childAReference;
            makeChildAccPair(candidateA, childA, root, first, paths);
            makeChildReference(referenceA, childAReference, root, first, paths);
            makeChildAccPair(childA, candidateA, afterFirst, second, paths);
            makeChildReference(childAReference, referenceA, afterFirst, second, paths);
            samePair(candidateA, referenceA, "transposed-path-a");

            AccPair candidateB = buildAccPairRoot(root, paths);
            AccPair referenceB = candidateB;
            AccPair childB, childBReference;
            makeChildAccPair(candidateB, childB, root, second, paths);
            makeChildReference(referenceB, childBReference, root, second, paths);
            makeChildAccPair(childB, candidateB, afterSecond, first, paths);
            makeChildReference(childBReference, referenceB, afterSecond, first, paths);
            samePair(candidateB, referenceB, "transposed-path-b");

            MoveList continuation = legalMoves(finalA);
            if (continuation.empty()) continue;
            const Move third = continuation[(size_t)(transpositions % continuation.size())];
            verifyMove(finalA, third, "same-edge-distinct-paths", paths, comparisons);
            AccPair candidateChildA, referenceChildA;
            AccPair candidateChildB, referenceChildB;
            makeChildAccPair(candidateA, candidateChildA, finalA, third, paths);
            makeChildReference(referenceA, referenceChildA, finalA, third, paths);
            makeChildAccPair(candidateB, candidateChildB, finalB, third, paths);
            makeChildReference(referenceB, referenceChildB, finalB, third, paths);
            samePair(candidateChildA, referenceChildA, "transposed-child-a");
            samePair(candidateChildB, referenceChildB, "transposed-child-b");
            ++transpositions;
        }
    }
    if (transpositions < 16) {
        std::fprintf(stderr, "insufficient commuting wall paths: %d\n", transpositions);
        std::exit(8);
    }
    comparisons += transpositions;
}

void verifyLazyHistories(std::mt19937_64& rng, int& comparisons,
                         PlayerPathCacheTable* paths) {
    for (int game = 0; game < 36; ++game) {
        State state = randomReachableState(rng, 6 + (int)(rng() % 24));
        AccPair candidate = buildAccPairRoot(state, paths);
        AccPair reference = candidate;
        for (int ply = 0; ply < 48 && winner(state) == -1; ++ply) {
            MoveList moves = legalMoves(state);
            if (moves.empty()) break;
            const Move move = moves[(size_t)(rng() % moves.size())];
            verifyMove(state, move, "live-path-edge", paths, comparisons);
            AccPair candidateChild, referenceChild;
            makeChildAccPair(candidate, candidateChild, state, move, paths);
            makeChildReference(reference, referenceChild, state, move, paths);
            samePair(candidate, reference, "lazy-parent-mutation");
            samePair(candidateChild, referenceChild, "lazy-child-state");
            candidate = candidateChild;
            reference = referenceChild;
            state = applyMove(state, move);
            if ((ply % 5) == 4) {
                for (int side = 0; side < 2; ++side) {
                    resolvePending(candidate, side, paths);
                    resolveReference(reference, side, paths);
                }
                samePair(candidate, reference, "lazy-pending-resolution");
            }
        }
        for (int side = 0; side < 2; ++side) {
            resolvePending(candidate, side, paths);
            resolveReference(reference, side, paths);
        }
        samePair(candidate, reference, "lazy-final-resolution");
    }
}

bool writeModifiedWeights(const std::string& source, const std::string& target,
                          int feature) {
    std::ifstream input(source, std::ios::binary);
    if (!input) return false;
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    const size_t offset = 2 * sizeof(int32_t) +
        ((size_t)feature * HIDDEN) * sizeof(int16_t);
    if (offset + sizeof(int16_t) > bytes.size()) return false;
    int16_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    value = value == INT16_MAX ? (int16_t)(value - 1) : (int16_t)(value + 1);
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
    std::ofstream output(target, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), (std::streamsize)bytes.size());
    return output.good();
}

void verifyWeightReload(const std::string& weights, const State& before,
                        const Move& move, PlayerPathCacheTable* paths,
                        int& comparisons) {
    const int perspective = before.turn;
    RecordingAccumulator recorder;
    static_cast<AccumulatorQuant&>(recorder) =
        buildAccumulatorQuant(before, perspective, paths);
    updateAccumulatorForMoveQuantKernel(recorder, true, before, move, paths);
    if (recorder.operations.empty()) return;
    const int feature = std::abs(recorder.operations.front()) - 1;
    const std::string changed = weights + ".edge-delta-modified.tmp";
    if (!writeModifiedWeights(weights, changed, feature)) {
        std::fprintf(stderr, "could not write modified weights: %s\n", changed.c_str());
        std::exit(4);
    }
    // Warm the old dense entry, then load different weights. The generation
    // change must force the cache to rebuild its feature transform.
    verifyMove(before, move, "weight-reload-warm", paths, comparisons);
    if (!loadWeightsQuant(changed)) {
        std::fprintf(stderr, "failed to load modified weights: %s\n", changed.c_str());
        std::exit(5);
    }
    verifyMove(before, move, "weight-reload-changed", paths, comparisons);
    if (!loadWeightsQuant(weights)) {
        std::fprintf(stderr, "failed to reload weights: %s\n", weights.c_str());
        std::exit(6);
    }
    std::remove(changed.c_str());
    verifyMove(before, move, "weight-reload-restored", paths, comparisons);
}

void verifyBfsAliasPreservation(int& comparisons) {
    constexpr uint64_t corner = 1ull << 63;
    constexpr int start = cellIdx(7, 7);
    constexpr int viewer = 0;
    PlayerPathCacheTable paths;

    const int aliasedKeyH = (int)distLenCached(corner, 0, start, viewer, &paths);
    const int directKeyV = (int)distLenCached(0, corner, start, viewer, nullptr);
    if (playerPathCacheKey(corner, 0, start, viewer) !=
            playerPathCacheKey(0, corner, start, viewer) ||
        aliasedKeyH == directKeyV) {
        std::fprintf(stderr, "BFS alias regression setup changed: h=%d v=%d\n",
                     aliasedKeyH, directKeyV);
        std::exit(9);
    }
    const int aliasedKeyV =
        (int)distLenCached(0, corner, start, viewer, &paths);
    if (aliasedKeyV != aliasedKeyH) {
        std::fprintf(stderr, "BFS alias behavior changed: expected=%d got=%d\n",
                     aliasedKeyH, aliasedKeyV);
        std::exit(10);
    }

    State before = initialState();
    before.pawn[0] = (uint8_t)start;
    before.pawn[1] = (uint8_t)cellIdx(0, 4);
    before.turn = 1;
    before.wallsLeft[1] = 9;
    const Zobrist& z = zobrist();
    before.hash = z.pawnKey[0][before.pawn[0]] ^
                  z.pawnKey[1][before.pawn[1]] ^ z.turnKey;
    const Move move = Move::wall(1, 7, 7);
    const State after = applyMove(before, move);
    const int trueAfterDistance =
        distLenCached(after.wallsH, after.wallsV, after.pawn[viewer], viewer, nullptr);
    const int cachedAfterDistance =
        distLenCached(after.wallsH, after.wallsV, after.pawn[viewer], viewer, &paths);
    if (trueAfterDistance != directKeyV || cachedAfterDistance != aliasedKeyH) {
        std::fprintf(stderr,
            "BFS edge alias setup changed: direct=%d cached=%d expected=%d\n",
            trueAfterDistance, cachedAfterDistance, aliasedKeyH);
        std::exit(11);
    }

    AccumulatorQuant reference = buildAccumulatorQuant(before, viewer, nullptr);
    AccumulatorQuant candidate = reference;
    updateAccumulatorForMoveQuantKernel(reference, false, before, move, &paths);
    updateAccumulatorForMoveQuant(candidate, false, before, move, &paths);
    sameAccumulator(reference, candidate, "preloaded-bfs-alias", viewer);
    if (candidate.ownDistBucket != distBucket(aliasedKeyH))
        fail("bfs-alias-bucket", viewer, -1, candidate.ownDistBucket,
             distBucket(aliasedKeyH));
    ++comparisons;
}

} // namespace

int main(int argc, char** argv) {
#if !ZQ_EXP_EDGE_FEATURE_DELTA
    std::fprintf(stderr, "compile with ZQ_EXP_EDGE_FEATURE_DELTA=1\n");
    return 1;
#endif
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <weights_int8.bin>\n", argv[0]);
        return 1;
    }
    if (!loadWeightsQuant(argv[1])) {
        std::fprintf(stderr, "failed to load weights: %s\n", argv[1]);
        return 2;
    }

    std::mt19937_64 rng(0xED6EDE17A2026106ull);
    int comparisons = 0;
    PlayerPathCacheTable paths;
    for (int sample = 0; sample < 240; ++sample) {
        State before = randomReachableState(rng, (int)(rng() % 38));
        if (winner(before) != -1) continue;
        MoveList moves = legalMoves(before);
        if (moves.empty()) continue;
        const size_t count = std::min<size_t>(moves.size(), 12);
        for (size_t i = 0; i < count; ++i)
            verifyMove(before, moves[i], "random-legal-edge", &paths, comparisons);
    }

    verifyTransposedPaths(comparisons, &paths);
    verifyLazyHistories(rng, comparisons, &paths);

    State reloadState = initialState();
    MoveList reloadMoves = legalMoves(reloadState);
    if (reloadMoves.empty()) return 7;
    verifyWeightReload(argv[1], reloadState, reloadMoves[0], &paths, comparisons);
    verifyBfsAliasPreservation(comparisons);

    std::printf("EDGE_FEATURE_DELTA_OK comparisons=%d hidden=%d features=%d contact=%d dense=%d\n",
                comparisons, HIDDEN, NUM_FEATURES,
                ZQ_NNUE_CONTACT_FEATURES, ZQ_EXP_EDGE_DENSE_DELTA);
    return 0;
}
