from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.teacher import run_central_weakness_rollouts as campaign


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        games_per_shard=512,
        threads=4,
        depth=40,
        time_ms=80,
        max_plies=160,
        seed=20260930,
        cheap_time_ms=20,
        replay_max_positions=12_000_000,
        val_fraction=0.15,
        stored_gamma=1.0,
        stored_outcome_weight=0.5,
    )


def test_move_mirror_is_an_involution():
    moves = ("e2", "d4", "a8", "d4h", "b6v", "h1h")
    for move in moves:
        assert campaign._mirror_move_text(campaign._mirror_move_text(move)) == move


def test_metadata_game_ids_are_namespaced_per_shard(tmp_path: Path):
    path = tmp_path / "sample.meta"
    rows = np.zeros(4, dtype=campaign.META_DTYPE)
    rows["game"] = np.asarray([0, 0, 5, 511], dtype=np.uint64)
    rows.tofile(path)

    campaign._namespace_metadata_game_ids(path, shard_index=3, games_per_shard=512)

    updated = np.fromfile(path, dtype=campaign.META_DTYPE)
    assert updated["game"].tolist() == [1536, 1536, 1541, 2047]


def test_wide_profile_selfplay_command_keeps_search_targets_untempered(tmp_path: Path):
    args = _args()
    command, profile, shard_seed = campaign._build_selfplay_command(
        args,
        executable=tmp_path / "selfplay",
        weights=tmp_path / "weights.bin",
        bank_path=tmp_path / "openings.jsonl",
        shard_index=0,
        out_bin=tmp_path / "shard.bin",
        out_meta=tmp_path / "shard.meta",
    )

    assert profile.name == "wide"
    assert shard_seed == args.seed
    assert "--mc-mode" in command
    assert "--playout-cap" in command
    assert command[command.index("--mc-temp-opening") + 1] == "1.6"
    assert command[command.index("--full-search-prob") + 1] == "0.42"
    assert command[command.index("--mcab-root-noise-epsilon") + 1] == "0.35"
    assert str(tmp_path / "shard.meta") == command[command.index("--meta-out") + 1]


def test_replay_command_is_literal_half_result_half_search_value(tmp_path: Path):
    args = _args()
    command = campaign._build_replay_command(args, tmp_path)

    assert command[command.index("--mode") + 1] == "stored_search"
    assert command[command.index("--stored-outcome-weight") + 1] == "0.5"
    assert command[command.index("--stored-gamma") + 1] == "1.0"
    assert command[command.index("--max-positions") + 1] == "12000000"


def test_pair_record_count_requires_aligned_v3_and_metadata(tmp_path: Path):
    v3 = tmp_path / "shard.bin"
    meta = tmp_path / "shard.meta"
    v3.write_bytes(b"\0" * (campaign.V3_RECORD_BYTES * 3))
    np.zeros(3, dtype=campaign.META_DTYPE).tofile(meta)

    assert campaign._pair_record_count(v3, meta) == 3