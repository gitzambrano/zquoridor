"""Horizontal (left-right) board symmetry transformations for Quoridor.

The Quoridor board has exact horizontal symmetry along the vertical axis (col -> 8 - col).
This module precomputes exact bijective mappings for pawn cells, wall slots, wall bitboards,
and the canonical 209 action policy indices.
"""
from __future__ import annotations
import numpy as np

# 9x9 cells: cell = r * 9 + c  ->  mirrored = r * 9 + (8 - c)
PAWN_MIRROR_H = np.array([r * 9 + (8 - c) for r in range(9) for c in range(9)], dtype=np.int64)

# 8x8 wall slots: slot = r * 8 + c  ->  mirrored = r * 8 + (7 - c)
WALL_SLOT_MIRROR_H = np.array([r * 8 + (7 - c) for r in range(8) for c in range(8)], dtype=np.int64)

# 209 Actions:
# 0..80: pawn dest cell (81 actions)
# 81..144: horizontal wall slot (64 actions)
# 145..208: vertical wall slot (64 actions)
ACTION_MIRROR_H = np.zeros(209, dtype=np.int64)
ACTION_MIRROR_H[:81] = PAWN_MIRROR_H
ACTION_MIRROR_H[81:145] = 81 + WALL_SLOT_MIRROR_H
ACTION_MIRROR_H[145:209] = 145 + WALL_SLOT_MIRROR_H

# Lookup table for reversing the 8 bits in a byte
_REV_BYTE = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], dtype=np.uint8)


def mirror_pawn_cell_h(cell: int | np.ndarray) -> int | np.ndarray:
    """Return the horizontally mirrored pawn cell index."""
    if isinstance(cell, (int, np.integer)):
        return int(PAWN_MIRROR_H[cell])
    return PAWN_MIRROR_H[np.asarray(cell, dtype=np.int64)]


def mirror_wall_slot_h(slot: int | np.ndarray) -> int | np.ndarray:
    """Return the horizontally mirrored wall slot index."""
    if isinstance(slot, (int, np.integer)):
        return int(WALL_SLOT_MIRROR_H[slot])
    return WALL_SLOT_MIRROR_H[np.asarray(slot, dtype=np.int64)]


def mirror_action_h(action: int | np.ndarray) -> int | np.ndarray:
    """Return the horizontally mirrored 209-action index."""
    if isinstance(action, (int, np.integer)):
        return int(ACTION_MIRROR_H[action])
    return ACTION_MIRROR_H[np.asarray(action, dtype=np.int64)]


def mirror_wall_bitboard_h(bb: int | np.ndarray) -> int | np.ndarray:
    """Horizontally mirror 64-bit wall bitboards.

    Each row r (bits r*8..r*8+7) represents wall slots across columns 0..7.
    Reflecting horizontally reverses the 8 bits within each byte.
    """
    is_scalar = isinstance(bb, (int, np.integer))
    arr = np.ascontiguousarray([bb] if is_scalar else bb, dtype=np.uint64)
    b = arr.view(np.uint8).reshape(-1, 8)
    b_rev = _REV_BYTE[b]
    mirrored = b_rev.reshape(-1).view(np.uint64).copy()
    return int(mirrored[0]) if is_scalar else mirrored


def mirror_sample_chunk_h(chunk: np.ndarray) -> np.ndarray:
    """Return a horizontally mirrored copy of a structured selfplay sample array.

    Works with V2 (32-byte) and V3 (64-byte) sample dtypes.
    """
    mirrored = chunk.copy()
    mirrored["own_pawn"] = PAWN_MIRROR_H[chunk["own_pawn"].astype(np.int64)].astype(chunk["own_pawn"].dtype)
    mirrored["opp_pawn"] = PAWN_MIRROR_H[chunk["opp_pawn"].astype(np.int64)].astype(chunk["opp_pawn"].dtype)
    mirrored["walls_h"] = mirror_wall_bitboard_h(chunk["walls_h"])
    mirrored["walls_v"] = mirror_wall_bitboard_h(chunk["walls_v"])
    mirrored["policy_target"] = ACTION_MIRROR_H[chunk["policy_target"].astype(np.int64)].astype(chunk["policy_target"].dtype)
    if "policy_top_idx" in chunk.dtype.names:
        top_idx = chunk["policy_top_idx"].astype(np.int64)
        mirrored["policy_top_idx"] = ACTION_MIRROR_H[top_idx].astype(chunk["policy_top_idx"].dtype)
    # own_dist, opp_dist, walls_left_own, walls_left_opp, nnue_eval, game_result, mover
    # are invariant under horizontal reflection.
    return mirrored


def mirror_dataset_dict_h(data: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Horizontally mirror a training dataset dictionary for the given indices (or all).

    Returns a new dict containing the mirrored slice.
    """
    if indices is None:
        indices = np.arange(len(data["value"]))
    n = len(indices)
    mirrored = {}
    for k, v in data.items():
        if k in ("own_pawn", "opp_pawn"):
            mirrored[k] = PAWN_MIRROR_H[v[indices].astype(np.int64)].astype(v.dtype)
        elif k in ("walls_h", "walls_v"):
            mirrored[k] = mirror_wall_bitboard_h(v[indices])
        elif k == "policy":
            mirrored[k] = v[indices][:, ACTION_MIRROR_H].astype(v.dtype)
        else:
            mirrored[k] = v[indices].copy()
    return mirrored


def stochastic_mirror_dict_h(data: dict[str, np.ndarray], indices: np.ndarray, flip_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Horizontally mirror selected samples where flip_mask is True in a new dict slice."""
    batch = {k: v[indices].copy() for k, v in data.items()}
    if not flip_mask.any():
        return batch
    m = np.flatnonzero(flip_mask)
    if "own_pawn" in batch:
        batch["own_pawn"][m] = PAWN_MIRROR_H[batch["own_pawn"][m].astype(np.int64)]
    if "opp_pawn" in batch:
        batch["opp_pawn"][m] = PAWN_MIRROR_H[batch["opp_pawn"][m].astype(np.int64)]
    if "walls_h" in batch:
        batch["walls_h"][m] = mirror_wall_bitboard_h(batch["walls_h"][m])
    if "walls_v" in batch:
        batch["walls_v"][m] = mirror_wall_bitboard_h(batch["walls_v"][m])
    if "policy" in batch:
        batch["policy"][m] = batch["policy"][m][:, ACTION_MIRROR_H]
    return batch

