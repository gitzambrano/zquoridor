from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.teacher import run_central_weakness_rollouts as campaign
from tools.teacher.selfplay_unique_store import V3_DTYPE


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        games_per_shard=512,
        threads=4,
        depth=40,
        time_ms=200,
        max_plies=160,
        seed=20260930,
        cheap_time_ms=20,
        full_search_opening_plies=16,
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
    assert command[command.index("--time-ms") + 1] == "200"
    assert command[command.index("--full-search-opening-plies") + 1] == "16"
    assert command[command.index("--cheap-time-ms") + 1] == "20"
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


def test_source_quotas_reserve_and_architectures():
    sources = campaign.SELFPLAY_SOURCES
    assert [(s["name"], s["target_positions"], s["features"]) for s in sources] == [
        ("production_bucketed", 8_000_000, 504),
        ("contact_bucketed", 4_000_000, 858),
    ]
    assert sources[0]["weights"] != sources[1]["weights"]
    assert sources[0]["executable"] != sources[1]["executable"]
    assert "architecture_manifest" in sources[1]
    assert campaign.RESERVE_UNASSIGNED == 4_000_000
    assert sum(s["target_positions"] for s in sources) + campaign.RESERVE_UNASSIGNED == 16_000_000


def test_profiles_are_identical_by_local_shard_for_every_source(tmp_path: Path):
    args = _args()
    for shard in range(15):
        commands = [campaign._build_selfplay_command(
            args, executable=tmp_path / s["executable"], weights=tmp_path / s["weights"],
            bank_path=tmp_path / "shared_bank.jsonl", shard_index=shard, source_index=i,
            out_bin=tmp_path / f"{i}.bin", out_meta=tmp_path / f"{i}.meta")
            for i, s in enumerate(campaign.SELFPLAY_SOURCES)]
        assert commands[0][1] == commands[1][1]
        assert commands[0][2] != commands[1][2]
        for command, _, _ in commands:
            assert command[command.index("--positions") + 1] == str(tmp_path / "shared_bank.jsonl")
            assert "--time-ms" in command and command[command.index("--time-ms") + 1] == "200"


def test_game_ids_do_not_collide_between_sources(tmp_path: Path):
    ids = []
    for source_index in range(2):
        path = tmp_path / f"{source_index}.meta"
        rows = np.zeros(3, dtype=campaign.META_DTYPE)
        rows["game"] = [0, 1, 511]
        rows.tofile(path)
        campaign._namespace_metadata_game_ids(path, source_index * (1 << 32), 512)
        ids.extend(np.fromfile(path, dtype=campaign.META_DTYPE)["game"].tolist())
    assert len(set(ids)) == 6


def test_resume_is_independent_and_rejects_partial_shards(tmp_path: Path):
    for name, count in (("production_bucketed", 2), ("contact_bucketed", 3)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "central_weakness_00000.bin").write_bytes(b"\0" * 64 * count)
        np.zeros(count, dtype=campaign.META_DTYPE).tofile(folder / "central_weakness_00000.meta")
    assert campaign._existing_shards(tmp_path / "production_bucketed")[:2] == (2, 1)
    assert campaign._existing_shards(tmp_path / "contact_bucketed")[:2] == (3, 1)
    (tmp_path / "production_bucketed" / "central_weakness_00001.bin").write_bytes(b"\0")
    import pytest
    with pytest.raises(ValueError):
        campaign._existing_shards(tmp_path / "production_bucketed")


def test_partial_admission_is_preserved_and_not_counted(tmp_path: Path):
    folder = tmp_path / "production_bucketed"
    folder.mkdir()
    (folder / "central_weakness_00000.bin").write_bytes(b"\0" * 64)
    campaign._recover_partial_shards(folder)
    assert campaign._existing_shards(folder)[:2] == (0, 0)
    assert list((folder / "staging").rglob("central_weakness_00000.bin"))


def test_contact_manifest_matches_weights_and_flags():
    source = campaign.SELFPLAY_SOURCES[1]
    artifacts = campaign._source_artifacts(source)
    assert "-DZQ_NNUE_CONTACT_FEATURES=1" in artifacts["cpp_flags"]
    assert artifacts["architecture_manifest_sha256"]


def test_campaign_identity_tracks_reserve_and_provenance():
    args = _args()
    args.games_per_shard = 512
    identity = campaign._campaign_identity(args, [dict(campaign.SELFPLAY_SOURCES[0])],
                                          {"fingerprint": "bank", "bank_sha256": "sha"})
    assert identity["reserve_unassigned"] == 4_000_000
    assert identity["sources"][0]["architecture"] == "multipath_phase_bucketed"
    assert identity["time_ms"] == 200


def test_incomplete_search_targets_are_removed_as_aligned_pairs(tmp_path: Path):
    bin_path, meta_path = tmp_path / "shard.bin", tmp_path / "shard.meta"
    rows = np.zeros(3, dtype=V3_DTYPE)
    rows["policy_top_prob"][:, 0] = [65535, 65535, 0]
    rows.tofile(bin_path)
    metadata = np.zeros(3, dtype=campaign.META_DTYPE)
    metadata["flags"] = [2, 1, 2]
    metadata["root"] = [0.6, np.nan, 0.4]
    metadata.tofile(meta_path)
    assert campaign._retain_complete_search_records(bin_path, meta_path) == 2
    assert campaign._pair_record_count(bin_path, meta_path) == 1
    assert np.fromfile(meta_path, dtype=campaign.META_DTYPE)["root"].tolist() == [0.6000000238418579]


def test_dry_run_reports_quotas_reserve_and_search_window(capsys):
    assert campaign.main(["--dry-run"]) == 0
    report = capsys.readouterr().out
    assert "8,000,000" in report and "4,000,000" in report
    assert "Reserve: unassigned" in report
    assert "200 ms for first 16 plies" in report
    assert "production_bucketed" in report and "contact_bucketed" in report


def test_wrong_feature_layout_is_rejected_before_selfplay():
    import pytest
    swapped = dict(campaign.SELFPLAY_SOURCES[0],
                   weights=campaign.SELFPLAY_SOURCES[1]["weights"])
    with pytest.raises(ValueError, match="layout/feature count mismatch"):
        campaign._source_artifacts(swapped)


def test_time_budget_for_recorded_search_is_fixed():
    import pytest
    with pytest.raises(SystemExit):
        campaign._resolve_args(["--time-ms", "80"])


def test_runner_recovers_interrupted_admission_before_counting(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "production_bucketed"
    source_dir.mkdir()
    (source_dir / "central_weakness_00000.bin").write_bytes(b"partial")
    executable = campaign._source_path(campaign.SELFPLAY_SOURCES[0]["executable"])
    monkeypatch.setattr(campaign, "_ensure_executable", lambda source, artifacts:
                        (executable, {"executable_sha256": "test-build"}))
    monkeypatch.setattr(campaign, "_generate_opening_bank", lambda *args:
                        {"fingerprint": "test-bank", "bank_sha256": "test-bank-sha"})

    def fake_selfplay(command, **kwargs):
        rows = np.zeros(1, dtype=V3_DTYPE)
        rows["policy_top_prob"][:, 0] = 65535
        rows.tofile(command[command.index("--out") + 1])
        metadata = np.zeros(1, dtype=campaign.META_DTYPE)
        metadata["flags"] = 2
        metadata["root"] = 0.5
        metadata.tofile(command[command.index("--meta-out") + 1])
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_selfplay)
    assert campaign.main(["--no-dry-run", "--no-prepare-replay", "--source", "production_bucketed",
                          "--target-positions", "1", "--out", str(tmp_path)]) == 0
    assert campaign._existing_shards(source_dir)[:2] == (1, 1)
    preserved = list((source_dir / "staging").rglob("central_weakness_00000.bin"))
    assert len(preserved) == 1 and preserved[0].read_bytes() == b"partial"
    assert campaign.CONFIG["threads"] == 10
