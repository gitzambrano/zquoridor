"""Contract tests for the one-way historical self-play migration."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import migrate_selfplay_v3 as migration  # noqa: E402
from read_selfplay import SAMPLE_DTYPE_LEGACY  # noqa: E402
import prepare_replay  # noqa: E402


def test_legacy_conversion_marks_turn_and_mirrors_second_player_policy():
    rows = np.zeros(2, dtype=SAMPLE_DTYPE_LEGACY)
    # A complete game starts in the normal physical board frame.
    rows[0] = (4, 76, 0, 0, 10, 10, 123, 1, 4, 8, 8)
    # The next row is player 1 in the old physical frame.  It contains one
    # horizontal wall and a pawn target that must be mirrored vertically.
    rows[1] = (13, 67, np.uint64(1 << 1), 0, 10, 10, 456, -1, 13, 7, 7)

    converted, dropped = migration.convert_legacy(rows)

    assert dropped == 0
    assert converted.dtype.itemsize == 64
    assert converted["mover"].tolist() == [0, 1]
    assert converted["nnue_eval"].tolist() == [32768, 32768]
    assert int(converted[1]["own_pawn"]) == 67
    assert int(converted[1]["opp_pawn"]) == 13
    assert int(converted[1]["walls_h"]) == (1 << 57)
    assert int(converted[1]["policy_target"]) == 67


def test_legacy_partial_prefix_is_discarded_before_turn_inference():
    rows = np.zeros(3, dtype=SAMPLE_DTYPE_LEGACY)
    rows[0] = (5, 76, 0, 0, 10, 10, 0, 0, 5, 8, 8)
    rows[1] = (4, 76, 0, 0, 10, 10, 0, 1, 4, 8, 8)
    rows[2] = (13, 76, 0, 0, 10, 10, 0, 1, 13, 7, 8)

    converted, dropped = migration.convert_legacy(rows)

    assert len(converted) == 2
    assert dropped == 1
    assert converted["mover"].tolist() == [0, 1]


def test_replay_filters_crossed_or_path_blocking_wall_topologies():
    assert not prepare_replay.legal_wall_topology(1, 1, 4, 76)
    # Eight horizontal walls form a complete fence between rows zero and one.
    assert not prepare_replay.legal_wall_topology((1 << 8) - 1, 0, 4, 76)
    assert prepare_replay.legal_wall_topology(1, 0, 4, 76)
