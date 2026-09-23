from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import numpy as np
import pytest
from mirror_augmentation import (
    PAWN_MIRROR_H,
    WALL_SLOT_MIRROR_H,
    ACTION_MIRROR_H,
    mirror_pawn_cell_h,
    mirror_wall_slot_h,
    mirror_action_h,
    mirror_wall_bitboard_h,
    mirror_sample_chunk_h,
    mirror_dataset_dict_h,
)
from training.read_selfplay import SAMPLE_DTYPE


def test_pawn_mirror_involution_and_boundaries():
    assert len(PAWN_MIRROR_H) == 81
    assert np.array_equal(PAWN_MIRROR_H[PAWN_MIRROR_H], np.arange(81))
    # Top-left (0,0) -> top-right (0,8)
    assert mirror_pawn_cell_h(0) == 8
    assert mirror_pawn_cell_h(8) == 0
    # Center column (4,4) -> center column (4,4)
    center = 4 * 9 + 4
    assert mirror_pawn_cell_h(center) == center
    # Bottom-right (8,8) -> bottom-left (8,0)
    assert mirror_pawn_cell_h(80) == 72
    assert mirror_pawn_cell_h(72) == 80


def test_wall_slot_mirror_involution_and_boundaries():
    assert len(WALL_SLOT_MIRROR_H) == 64
    assert np.array_equal(WALL_SLOT_MIRROR_H[WALL_SLOT_MIRROR_H], np.arange(64))
    # Slot (0,0) -> (0,7)
    assert mirror_wall_slot_h(0) == 7
    assert mirror_wall_slot_h(7) == 0
    # Slot (7,7) -> (7,0)
    assert mirror_wall_slot_h(63) == 56
    assert mirror_wall_slot_h(56) == 63


def test_action_mirror_involution():
    assert len(ACTION_MIRROR_H) == 209
    assert np.array_equal(ACTION_MIRROR_H[ACTION_MIRROR_H], np.arange(209))
    # Pawn moves
    for c in range(81):
        assert mirror_action_h(c) == PAWN_MIRROR_H[c]
    # Horizontal walls
    for slot in range(64):
        assert mirror_action_h(81 + slot) == 81 + WALL_SLOT_MIRROR_H[slot]
    # Vertical walls
    for slot in range(64):
        assert mirror_action_h(145 + slot) == 145 + WALL_SLOT_MIRROR_H[slot]


def test_mirror_wall_bitboard():
    # Slot 0 set: bit 0 -> bit 7 (slot 7)
    bb = np.uint64(1 << 0)
    assert mirror_wall_bitboard_h(bb) == (1 << 7)

    # Double mirror is identity
    rng = np.random.default_rng(2026)
    random_bbs = rng.integers(0, 2**64, size=50, dtype=np.uint64)
    mirrored = mirror_wall_bitboard_h(random_bbs)
    double_mirrored = mirror_wall_bitboard_h(mirrored)
    assert np.array_equal(double_mirrored, random_bbs)


def test_mirror_sample_chunk():
    chunk = np.zeros(2, dtype=SAMPLE_DTYPE)
    chunk[0]["own_pawn"] = 0
    chunk[0]["opp_pawn"] = 80
    chunk[0]["walls_h"] = np.uint64(1 << 3)
    chunk[0]["walls_v"] = np.uint64(1 << 11)
    chunk[0]["policy_target"] = 0
    chunk[0]["policy_top_idx"] = [0, 81, 145, 4, 10, 20, 30, 40]
    chunk[0]["policy_top_prob"] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    mirrored = mirror_sample_chunk_h(chunk)
    assert mirrored[0]["own_pawn"] == 8
    assert mirrored[0]["opp_pawn"] == 72
    assert mirrored[0]["policy_target"] == 8
    assert mirrored[0]["policy_top_idx"][0] == 8
    assert mirrored[0]["policy_top_idx"][1] == 81 + WALL_SLOT_MIRROR_H[0]
    assert mirrored[0]["policy_top_idx"][2] == 145 + WALL_SLOT_MIRROR_H[0]
    # Probabilities unchanged
    assert np.array_equal(mirrored[0]["policy_top_prob"], chunk[0]["policy_top_prob"])


def test_mirror_dataset_dict():
    data = {
        "own_pawn": np.array([0, 80], dtype=np.uint8),
        "opp_pawn": np.array([8, 72], dtype=np.uint8),
        "walls_h": np.array([1 << 0, 1 << 7], dtype=np.uint64),
        "walls_v": np.array([0, 0], dtype=np.uint64),
        "policy": np.zeros((2, 209), dtype=np.float32),
        "value": np.array([0.5, -0.5], dtype=np.float32),
        "weight": np.array([1.0, 1.0], dtype=np.float32),
    }
    data["policy"][0, 0] = 1.0  # pawn to 0 -> mirror should be pawn to 8
    data["policy"][1, 81] = 1.0  # H-wall slot 0 -> mirror should be H-wall slot 7 (81 + 7 = 88)

    m = mirror_dataset_dict_h(data)
    assert m["own_pawn"].tolist() == [8, 72]
    assert m["opp_pawn"].tolist() == [0, 80]
    assert m["policy"][0, 8] == 1.0
    assert m["policy"][1, 88] == 1.0
    assert np.array_equal(m["value"], data["value"])
