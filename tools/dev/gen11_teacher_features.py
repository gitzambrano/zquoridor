#!/usr/bin/env python3
"""Build temporary Gen11 NNUE experiment variants in a clean checkout.

The script keeps production files unchanged in Git. CI applies one variant in
its workspace, trains it, benchmarks it, and discards the modified checkout.
This allows architecture A/B tests from one reproducible experiment branch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NNUE = ROOT / "src" / "nnue.hpp"
TRAIN = ROOT / "training" / "train_nnue.py"
QUANT = ROOT / "training" / "quantize_nnue.py"
SELFPLAY = ROOT / "tools" / "selfplay" / "selfplay.hpp"

BASE_FEATURES = 354
BASE_HIDDEN = 256
TARGET_HIDDEN = 320
VALUE_HIDDEN = 32
POLICY_OUT = 209
RACE_DIFF_BUCKETS = 41
WALL_DIFF_BUCKETS = 21
RACE_WALL_BUCKETS = 25
RACE_FEATURES = BASE_FEATURES + RACE_DIFF_BUCKETS + WALL_DIFF_BUCKETS + RACE_WALL_BUCKETS


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _function_span(text: str, signature: str) -> tuple[int, int]:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for idx in range(brace, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, idx + 1
    raise RuntimeError(f"unclosed function: {signature}")


def _insert_before_function_close(text: str, signature: str, code: str) -> str:
    start, end = _function_span(text, signature)
    segment = text[start:end]
    close = segment.rfind("}")
    segment = segment[:close] + code + segment[close:]
    return text[:start] + segment + text[end:]


def _insert_before_return(text: str, signature: str, code: str) -> str:
    start, end = _function_span(text, signature)
    segment = text[start:end]
    marker = "    return acc;"
    count = segment.count(marker)
    if count != 1:
        raise RuntimeError(f"{signature}: expected one return acc, found {count}")
    segment = segment.replace(marker, code + marker, 1)
    return text[:start] + segment + text[end:]


def patch_teacher() -> None:
    text = _read(SELFPLAY)
    old = """                if (!ranked.empty() && sumVisits > 0.0)\n                    rec.policyTopProb[0] = (uint16_t)(rec.policyTopProb[0] + (65535u - assigned));\n"""
    new = old + """\n                // Use the searched root value as a value teacher when this\n                // sample came from a valid MCAB root. The sample field stays\n                // architecture-neutral: it still stores white win probability.\n                if (rootNode->totalN > 0) {\n                    double sumW = 0.0;\n                    for (float w : rootNode->W) sumW += (double)w;\n                    double qMover = sumW / (double)rootNode->totalN;\n                    qMover = std::max(0.0, std::min(1.0, qMover));\n                    double qWhite = (mover == 0) ? qMover : (1.0 - qMover);\n                    rec.evalNNUE = (uint16_t)std::lround(qWhite * (double)EV_SCALE);\n                }\n"""
    text = _replace_once(text, old, new, "MCAB value teacher injection")
    _write(SELFPLAY, text)
    print("patched self-play: MCAB top-8 policy teacher + searched root value teacher")


def _patch_width() -> None:
    text = _read(NNUE)
    text = _replace_once(text, "constexpr int HIDDEN = 256;", "constexpr int HIDDEN = 320;", "nnue HIDDEN")
    _write(NNUE, text)

    text = _read(TRAIN)
    text = _replace_once(text, "HIDDEN = 256\n", "HIDDEN = 320\n", "trainer HIDDEN")
    _write(TRAIN, text)

    text = _read(QUANT)
    text = _replace_once(text, "HIDDEN = 256\n", "HIDDEN = 320\n", "quantizer HIDDEN")
    _write(QUANT, text)


def _patch_race_features_cpp() -> None:
    text = _read(NNUE)
    old_constants = """constexpr int NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS;  // 354\nconstexpr int HIDDEN = 320;\n"""
    new_constants = """constexpr int BASE_NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS;\nconstexpr int RACE_DIFF_BUCKETS = 41;   // clamp(oppDist-ownDist, -20, 20)\nconstexpr int WALL_DIFF_BUCKETS = 21;   // clamp(ownWalls-oppWalls, -10, 10)\nconstexpr int RACE_WALL_BUCKETS = 25;   // 5x5 coarse interaction grid\nconstexpr int NUM_FEATURES = BASE_NUM_FEATURES + RACE_DIFF_BUCKETS + WALL_DIFF_BUCKETS + RACE_WALL_BUCKETS;\nconstexpr int HIDDEN = 320;\n"""
    text = _replace_once(text, old_constants, new_constants, "C++ race constants")

    marker = "inline int featOppWallsLeft(int bucket) { return WALLS_LEFT_FEAT_BASE + WALLS_LEFT_BUCKETS + bucket; }\n"
    helper = marker + """constexpr int RACE_DIFF_FEAT_BASE = WALLS_LEFT_FEAT_BASE + 2 * WALLS_LEFT_BUCKETS;\nconstexpr int WALL_DIFF_FEAT_BASE = RACE_DIFF_FEAT_BASE + RACE_DIFF_BUCKETS;\nconstexpr int RACE_WALL_FEAT_BASE = WALL_DIFF_FEAT_BASE + WALL_DIFF_BUCKETS;\n\ninline int raceDiffBucketOf(int ownDist, int oppDist) {\n    int d = oppDist - ownDist;\n    if (d < -20) d = -20;\n    if (d > 20) d = 20;\n    return d + 20;\n}\ninline int wallDiffBucketOf(int ownWalls, int oppWalls) {\n    int d = ownWalls - oppWalls;\n    if (d < -10) d = -10;\n    if (d > 10) d = 10;\n    return d + 10;\n}\ninline int coarseDiff5(int d) {\n    if (d <= -4) return 0;\n    if (d <= -1) return 1;\n    if (d == 0) return 2;\n    if (d <= 3) return 3;\n    return 4;\n}\ninline int raceWallBucketOf(int ownDist, int oppDist, int ownWalls, int oppWalls) {\n    int race = coarseDiff5(oppDist - ownDist);\n    int reserve = coarseDiff5(ownWalls - oppWalls);\n    return race * 5 + reserve;\n}\ninline int featRaceDiff(int bucket) { return RACE_DIFF_FEAT_BASE + bucket; }\ninline int featWallDiff(int bucket) { return WALL_DIFF_FEAT_BASE + bucket; }\ninline int featRaceWall(int bucket) { return RACE_WALL_FEAT_BASE + bucket; }\n"""
    text = _replace_once(text, marker, helper, "C++ race feature helpers")

    def add_fields(source: str, struct_name: str) -> str:
        struct_pos = source.index(f"struct {struct_name} {{")
        marker_pos = source.index("    void addFeature(int featIdx) {", struct_pos)
        fields = """    int raceDiffBucket = 0;\n    int wallDiffBucket = 0;\n    int raceWallBucket = 0;\n\n"""
        return source[:marker_pos] + fields + source[marker_pos:]

    text = add_fields(text, "Accumulator")
    text = add_fields(text, "AccumulatorQuant")

    float_struct_end = text.index("// recomputa do zero", text.index("struct Accumulator {"))
    refresh = """template <typename AccT>\ninline void refreshDerivedRaceFeatures(AccT& acc) {\n    int rd = raceDiffBucketOf(acc.ownDistBucket, acc.oppDistBucket);\n    int wd = wallDiffBucketOf(acc.ownWallsLeftBucket, acc.oppWallsLeftBucket);\n    int rw = raceWallBucketOf(acc.ownDistBucket, acc.oppDistBucket,\n                              acc.ownWallsLeftBucket, acc.oppWallsLeftBucket);\n    if (rd != acc.raceDiffBucket) {\n        acc.removeFeature(featRaceDiff(acc.raceDiffBucket));\n        acc.addFeature(featRaceDiff(rd));\n        acc.raceDiffBucket = rd;\n    }\n    if (wd != acc.wallDiffBucket) {\n        acc.removeFeature(featWallDiff(acc.wallDiffBucket));\n        acc.addFeature(featWallDiff(wd));\n        acc.wallDiffBucket = wd;\n    }\n    if (rw != acc.raceWallBucket) {\n        acc.removeFeature(featRaceWall(acc.raceWallBucket));\n        acc.addFeature(featRaceWall(rw));\n        acc.raceWallBucket = rw;\n    }\n}\n\n"""
    text = text[:float_struct_end] + refresh + text[float_struct_end:]

    init_code = """    acc.raceDiffBucket = raceDiffBucketOf(acc.ownDistBucket, acc.oppDistBucket);\n    acc.wallDiffBucket = wallDiffBucketOf(acc.ownWallsLeftBucket, acc.oppWallsLeftBucket);\n    acc.raceWallBucket = raceWallBucketOf(acc.ownDistBucket, acc.oppDistBucket,\n                                           acc.ownWallsLeftBucket, acc.oppWallsLeftBucket);\n    acc.addFeature(featRaceDiff(acc.raceDiffBucket));\n    acc.addFeature(featWallDiff(acc.wallDiffBucket));\n    acc.addFeature(featRaceWall(acc.raceWallBucket));\n"""
    text = _insert_before_return(text, "inline Accumulator buildAccumulator(", init_code)
    text = _insert_before_return(text, "inline AccumulatorQuant buildAccumulatorQuant(", init_code)
    text = _insert_before_function_close(
        text,
        "inline void updateAccumulatorForMove(Accumulator&",
        "    refreshDerivedRaceFeatures(acc);\n",
    )
    text = _insert_before_function_close(
        text,
        "inline void updateAccumulatorForMoveQuant(AccumulatorQuant&",
        "    refreshDerivedRaceFeatures(acc);\n",
    )
    _write(NNUE, text)


def _patch_race_features_python() -> None:
    text = _read(TRAIN)
    old = """NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS  # 354\nHIDDEN = 320\n"""
    new = """BASE_NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS\nRACE_DIFF_BUCKETS = 41\nWALL_DIFF_BUCKETS = 21\nRACE_WALL_BUCKETS = 25\nRACE_DIFF_FEAT_BASE = BASE_NUM_FEATURES\nWALL_DIFF_FEAT_BASE = RACE_DIFF_FEAT_BASE + RACE_DIFF_BUCKETS\nRACE_WALL_FEAT_BASE = WALL_DIFF_FEAT_BASE + WALL_DIFF_BUCKETS\nNUM_FEATURES = RACE_WALL_FEAT_BASE + RACE_WALL_BUCKETS  # 441\nHIDDEN = 320\n"""
    text = _replace_once(text, old, new, "trainer race constants")

    marker = "    x[np.arange(n), wl_base + WALLS_LEFT_BUCKETS + opp_wl_bucket] = 1.0\n"
    feature_code = marker + """\n    # Cheap derived race features. They reuse the distance and wall-stock\n    # values already present in each sample and require no extra path search.\n    race_raw = opp_bucket - own_bucket\n    wall_raw = own_wl_bucket - opp_wl_bucket\n    race_bucket = np.clip(race_raw, -20, 20) + 20\n    wall_bucket = np.clip(wall_raw, -10, 10) + 10\n    x[np.arange(n), RACE_DIFF_FEAT_BASE + race_bucket] = 1.0\n    x[np.arange(n), WALL_DIFF_FEAT_BASE + wall_bucket] = 1.0\n\n    def coarse5(v):\n        return np.select(\n            [v <= -4, v <= -1, v == 0, v <= 3],\n            [0, 1, 2, 3],\n            default=4,\n        ).astype(np.int64)\n\n    race_wall_bucket = coarse5(race_raw) * 5 + coarse5(wall_raw)\n    x[np.arange(n), RACE_WALL_FEAT_BASE + race_wall_bucket] = 1.0\n"""
    text = _replace_once(text, marker, feature_code, "trainer race feature encoding")
    _write(TRAIN, text)

    text = _read(QUANT)
    old = """NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS  # 354\nHIDDEN = 320\n"""
    new = """BASE_NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS\nRACE_DIFF_BUCKETS = 41\nWALL_DIFF_BUCKETS = 21\nRACE_WALL_BUCKETS = 25\nNUM_FEATURES = BASE_NUM_FEATURES + RACE_DIFF_BUCKETS + WALL_DIFF_BUCKETS + RACE_WALL_BUCKETS  # 441\nHIDDEN = 320\n"""
    text = _replace_once(text, old, new, "quantizer race constants")
    _write(QUANT, text)


def patch_model(variant: str) -> None:
    if variant not in {"wide320", "race320"}:
        raise SystemExit(f"unknown variant: {variant}")
    _patch_width()
    if variant == "race320":
        _patch_race_features_cpp()
        _patch_race_features_python()
    features = RACE_FEATURES if variant == "race320" else BASE_FEATURES
    print(f"patched model: variant={variant} features={features} hidden={TARGET_HIDDEN}")


def _read_base_weights(path: Path) -> dict[str, np.ndarray]:
    expected_floats = (
        BASE_FEATURES * BASE_HIDDEN
        + BASE_HIDDEN
        + BASE_HIDDEN * VALUE_HIDDEN
        + VALUE_HIDDEN
        + VALUE_HIDDEN
        + 1
        + POLICY_OUT * BASE_HIDDEN
        + POLICY_OUT
    )
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != expected_floats:
        raise RuntimeError(
            f"{path}: expected {expected_floats} float32 values for the 354x256 baseline, got {raw.size}"
        )
    pos = 0

    def take(count: int) -> np.ndarray:
        nonlocal pos
        out = raw[pos : pos + count]
        pos += count
        return out

    return {
        "w1": take(BASE_FEATURES * BASE_HIDDEN).reshape(BASE_FEATURES, BASE_HIDDEN),
        "b1": take(BASE_HIDDEN),
        "wv1": take(BASE_HIDDEN * VALUE_HIDDEN).reshape(BASE_HIDDEN, VALUE_HIDDEN),
        "bv1": take(VALUE_HIDDEN),
        "wv2": take(VALUE_HIDDEN),
        "bv2": take(1),
        "wp": take(POLICY_OUT * BASE_HIDDEN).reshape(POLICY_OUT, BASE_HIDDEN),
        "bp": take(POLICY_OUT),
    }


def expand_weights(variant: str, src: Path, dst: Path, seed: int) -> None:
    if variant not in {"wide320", "race320"}:
        raise SystemExit(f"unknown variant: {variant}")
    source = _read_base_weights(src)
    target_features = RACE_FEATURES if variant == "race320" else BASE_FEATURES
    rng = np.random.default_rng(seed)

    w1 = np.zeros((target_features, TARGET_HIDDEN), dtype=np.float32)
    w1[:BASE_FEATURES, :BASE_HIDDEN] = source["w1"]
    # New hidden units start with nonzero activations, while every outgoing\n    # connection from them starts at zero. The expanded network therefore\n    # reproduces the baseline output exactly at step zero but new units can\n    # acquire outgoing gradients immediately.
    w1[:, BASE_HIDDEN:] = rng.normal(0.0, 0.02, size=(target_features, TARGET_HIDDEN - BASE_HIDDEN)).astype(np.float32)
    b1 = np.zeros(TARGET_HIDDEN, dtype=np.float32)
    b1[:BASE_HIDDEN] = source["b1"]
    b1[BASE_HIDDEN:] = np.float32(0.05)

    wv1 = np.zeros((TARGET_HIDDEN, VALUE_HIDDEN), dtype=np.float32)
    wv1[:BASE_HIDDEN] = source["wv1"]
    wp = np.zeros((POLICY_OUT, TARGET_HIDDEN), dtype=np.float32)
    wp[:, :BASE_HIDDEN] = source["wp"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        for array in (
            w1,
            b1,
            wv1,
            source["bv1"],
            source["wv2"],
            source["bv2"],
            wp,
            source["bp"],
        ):
            f.write(np.ascontiguousarray(array, dtype="<f4").tobytes())
    print(
        f"expanded baseline weights: variant={variant} features={target_features} "
        f"hidden={TARGET_HIDDEN} -> {dst}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("patch-teacher")

    p_model = sub.add_parser("patch-model")
    p_model.add_argument("--variant", required=True, choices=["wide320", "race320"])

    p_expand = sub.add_parser("expand-weights")
    p_expand.add_argument("--variant", required=True, choices=["wide320", "race320"])
    p_expand.add_argument("--src", required=True, type=Path)
    p_expand.add_argument("--dst", required=True, type=Path)
    p_expand.add_argument("--seed", type=int, default=200809)

    args = parser.parse_args()
    if args.command == "patch-teacher":
        patch_teacher()
    elif args.command == "patch-model":
        patch_model(args.variant)
    elif args.command == "expand-weights":
        expand_weights(args.variant, args.src, args.dst, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
