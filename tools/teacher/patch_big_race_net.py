#!/usr/bin/env python3
"""Patch src/nnue.hpp in-place for the experimental big race-aware NNUE.

The experiment keeps ZQuoridor's normal sparse/incremental NNUE machinery but
changes the representation from 354->256 to 1340->512.  The 986 added inputs
make distance/resource interactions explicit instead of asking one hidden layer
to discover all of them from independent one-hot features.
"""
from __future__ import annotations

import argparse
from pathlib import Path


ARCH_BLOCK_OLD = """constexpr int NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS;  // 354
constexpr int HIDDEN = 256;
"""

ARCH_BLOCK_NEW = """// Experimental race-aware architecture (Claustrophobia teacher v1).
// Base inputs stay bit-for-bit identical to production (354).  Added inputs:
//   4 x (distance bucket x wall-stock bucket) = 4 * 21 * 11 = 924
//   race-margin bucket (oppDist-ownDist, clipped -20..20) = 41
//   wall-margin bucket (ownWalls-oppWalls, clipped -10..10) = 21
// Total: 354 + 924 + 41 + 21 = 1340.
constexpr int BASE_NUM_FEATURES = N * N + N * N + WS * WS * 2
                                + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS;  // 354
constexpr int DIST_WALL_JOINT = DIST_BUCKETS * WALLS_LEFT_BUCKETS;            // 231
constexpr int OWN_DIST_OWN_WALL_BASE = BASE_NUM_FEATURES;                     // 354
constexpr int OWN_DIST_OPP_WALL_BASE = OWN_DIST_OWN_WALL_BASE + DIST_WALL_JOINT;
constexpr int OPP_DIST_OWN_WALL_BASE = OWN_DIST_OPP_WALL_BASE + DIST_WALL_JOINT;
constexpr int OPP_DIST_OPP_WALL_BASE = OPP_DIST_OWN_WALL_BASE + DIST_WALL_JOINT;
constexpr int RACE_MARGIN_BUCKETS = 2 * DIST_BUCKETS - 1;                      // 41
constexpr int RACE_MARGIN_BASE = OPP_DIST_OPP_WALL_BASE + DIST_WALL_JOINT;    // 1278
constexpr int WALL_MARGIN_BUCKETS = 2 * WALLS_PER_PLAYER + 1;                 // 21
constexpr int WALL_MARGIN_BASE = RACE_MARGIN_BASE + RACE_MARGIN_BUCKETS;      // 1319
constexpr int NUM_FEATURES = WALL_MARGIN_BASE + WALL_MARGIN_BUCKETS;           // 1340
constexpr int HIDDEN = 512;
"""

FEATURE_ANCHOR = """inline int featOppWallsLeft(int bucket) { return WALLS_LEFT_FEAT_BASE + WALLS_LEFT_BUCKETS + bucket; }
"""

FEATURE_HELPERS = r'''

inline int clampInt(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
inline int featOwnDistOwnWalls(int d, int w) {
    return OWN_DIST_OWN_WALL_BASE + d * WALLS_LEFT_BUCKETS + w;
}
inline int featOwnDistOppWalls(int d, int w) {
    return OWN_DIST_OPP_WALL_BASE + d * WALLS_LEFT_BUCKETS + w;
}
inline int featOppDistOwnWalls(int d, int w) {
    return OPP_DIST_OWN_WALL_BASE + d * WALLS_LEFT_BUCKETS + w;
}
inline int featOppDistOppWalls(int d, int w) {
    return OPP_DIST_OPP_WALL_BASE + d * WALLS_LEFT_BUCKETS + w;
}
inline int featRaceMargin(int ownDist, int oppDist) {
    int delta = clampInt(oppDist - ownDist, -(DIST_BUCKETS - 1), DIST_BUCKETS - 1);
    return RACE_MARGIN_BASE + delta + (DIST_BUCKETS - 1);
}
inline int featWallMargin(int ownWalls, int oppWalls) {
    int delta = clampInt(ownWalls - oppWalls, -WALLS_PER_PLAYER, WALLS_PER_PLAYER);
    return WALL_MARGIN_BASE + delta + WALLS_PER_PLAYER;
}
'''

INTERACTION_HELPERS = r'''
// Six interaction rows are always active.  Updating them explicitly makes the
// representation conditional on wall stock without sacrificing NNUE's sparse
// incremental update path.
template <typename Acc>
inline void addRaceInteractionFeatures(Acc& acc) {
    acc.addFeature(featOwnDistOwnWalls(acc.ownDistBucket, acc.ownWallsLeftBucket));
    acc.addFeature(featOwnDistOppWalls(acc.ownDistBucket, acc.oppWallsLeftBucket));
    acc.addFeature(featOppDistOwnWalls(acc.oppDistBucket, acc.ownWallsLeftBucket));
    acc.addFeature(featOppDistOppWalls(acc.oppDistBucket, acc.oppWallsLeftBucket));
    acc.addFeature(featRaceMargin(acc.ownDistBucket, acc.oppDistBucket));
    acc.addFeature(featWallMargin(acc.ownWallsLeftBucket, acc.oppWallsLeftBucket));
}

template <typename Acc>
inline void removeRaceInteractionFeatures(Acc& acc) {
    acc.removeFeature(featOwnDistOwnWalls(acc.ownDistBucket, acc.ownWallsLeftBucket));
    acc.removeFeature(featOwnDistOppWalls(acc.ownDistBucket, acc.oppWallsLeftBucket));
    acc.removeFeature(featOppDistOwnWalls(acc.oppDistBucket, acc.ownWallsLeftBucket));
    acc.removeFeature(featOppDistOppWalls(acc.oppDistBucket, acc.oppWallsLeftBucket));
    acc.removeFeature(featRaceMargin(acc.ownDistBucket, acc.oppDistBucket));
    acc.removeFeature(featWallMargin(acc.ownWallsLeftBucket, acc.oppWallsLeftBucket));
}

'''


def function_bounds(text: str, signature: str) -> tuple[int, int]:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for i in range(brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i
    raise ValueError(f"unbalanced function: {signature}")


def patch_update_function(text: str, signature: str) -> str:
    start, close = function_bounds(text, signature)
    body = text[start:close]
    marker = "    int mover = before.turn, opp = 1 - mover;\n"
    if marker not in body:
        raise ValueError(f"mover marker not found in {signature}")
    body = body.replace(marker, marker + "    removeRaceInteractionFeatures(acc);\n", 1)
    # Recompute closing location after the insertion by replacing the original body.
    text = text[:start] + body + text[close:]
    _, close = function_bounds(text, signature)
    text = text[:close] + "    addRaceInteractionFeatures(acc);\n" + text[close:]
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="src/nnue.hpp")
    args = ap.parse_args()
    path = Path(args.path)
    text = path.read_text(encoding="utf-8")

    if "constexpr int NUM_FEATURES = WALL_MARGIN_BASE + WALL_MARGIN_BUCKETS" in text:
        print("big-race architecture already applied")
        return 0
    if text.count(ARCH_BLOCK_OLD) != 1:
        raise SystemExit("production architecture anchor changed; refusing unsafe patch")
    text = text.replace(ARCH_BLOCK_OLD, ARCH_BLOCK_NEW, 1)

    if text.count(FEATURE_ANCHOR) != 1:
        raise SystemExit("feature anchor changed; refusing unsafe patch")
    text = text.replace(FEATURE_ANCHOR, FEATURE_ANCHOR + FEATURE_HELPERS, 1)

    acc_anchor = "// recomputa do zero -- só usada ao entrar numa posição \"fria\" (raiz da busca)\n"
    if text.count(acc_anchor) != 1:
        raise SystemExit("accumulator helper anchor changed; refusing unsafe patch")
    text = text.replace(acc_anchor, INTERACTION_HELPERS + acc_anchor, 1)

    build_tail = "    acc.addFeature(featOppWallsLeft(acc.oppWallsLeftBucket));\n    return acc;"
    if text.count(build_tail) != 2:
        raise SystemExit(f"expected two accumulator build tails, found {text.count(build_tail)}")
    text = text.replace(
        build_tail,
        "    acc.addFeature(featOppWallsLeft(acc.oppWallsLeftBucket));\n"
        "    addRaceInteractionFeatures(acc);\n"
        "    return acc;",
    )

    text = patch_update_function(text, "inline void updateAccumulatorForMove(")
    text = patch_update_function(text, "inline void updateAccumulatorForMoveQuant(")

    path.write_text(text, encoding="utf-8")
    print("patched", path)
    print("architecture: NUM_FEATURES=1340 HIDDEN=512 VALUE_HIDDEN=32 POLICY_OUT=209")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
