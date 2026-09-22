import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tools.teacher.build_selfplay_seed_schedule import REQUIRED_FAMILIES
from tools.teacher.run_four_million_selfplay import (
    OutputLock,
    TargetCounts,
    _counts_from_store,
    _game_entries,
    counts_for,
    discover_resume_state,
    reconcile_store,
    run_controller,
    select_thread_count,
    select_next_pool,
    targets_met,
)
from tools.teacher.selfplay_unique_store import UniqueStateStore, V3_DTYPE


def _counts(*, total, central, broad, family):
    return counts_for(total=total, central=central, broad=broad, family=family)


def test_stop_requires_total_composition_and_family_floors():
    family = {name: 400_000 for name in REQUIRED_FAMILIES}
    family["front_wall"] = 399_999
    counts = _counts(total=4_100_000, central=2_500_000, broad=1_600_000, family=family)

    assert not targets_met(counts)

    counts.family["front_wall"] = 400_000
    assert targets_met(counts)


def test_adaptive_selection_uses_the_largest_proportional_deficit():
    family = {name: 400_000 for name in REQUIRED_FAMILIES}
    family["pawn_jump"] = 100_000
    counts = _counts(total=2_300_000, central=1_000_000, broad=1_300_000, family=family)

    assert select_next_pool(counts) == ("central", "pawn_jump")


def test_imported_unclassified_central_states_reduce_family_capacity():
    family = {name: 0 for name in REQUIRED_FAMILIES}
    family["front_wall"] = 1
    counts = _counts(total=6, central=2, broad=0, family=family)

    # One imported central state is not assigned to a required family. The
    # controller must reserve that state inside the central target.
    from tools.teacher.run_four_million_selfplay import _capacity

    capacity = _capacity(
        counts, "central", "front_wall", total_target=7,
        central_target=6, broad_target=1, family_floor=1,
    )
    assert capacity == 0


def test_thread_helper_rejects_an_unsafe_reservation():
    with pytest.raises(ValueError, match="exceed the CPU limit"):
        select_thread_count(True, during_fine_tune=10, fine_tune_threads=7,
                           cpu_thread_limit=16)


def _write_pair(out_dir, index, *, meta=True, accepted=True):
    shard = out_dir / f"shard_{index:06d}.bin"
    sidecar = out_dir / f"shard_{index:06d}.meta"
    np.zeros(1, dtype=V3_DTYPE).tofile(shard)
    if meta:
        sidecar.write_bytes(bytes(20))
    if accepted:
        return {
            "index": index,
            "v3": shard.name,
            "meta": sidecar.name,
            "source": "central",
            "family": "front_wall",
            "records": 1,
        }
    return None


def test_resume_uses_accepted_manifest_and_quarantines_incomplete_pairs(tmp_path):
    out_dir = tmp_path / "campaign"
    out_dir.mkdir()
    accepted = _write_pair(out_dir, 4)
    _write_pair(out_dir, 9, meta=False, accepted=False)
    staging = out_dir / "staging"
    staging.mkdir()
    (staging / "shard_000010_0000.bin").write_bytes(bytes(64))
    (out_dir / "manifest.json").write_text(
        json.dumps({"schema": "zquoridor.four_million_selfplay.v1", "accepted_shards": [accepted]}),
        encoding="utf-8",
    )

    state = discover_resume_state(out_dir)

    assert state.next_shard == 5
    assert state.counts == TargetCounts(1, 1, 0, {"front_wall": 1})
    assert (out_dir / "quarantine" / "shard_000009.bin").exists()
    assert (out_dir / "quarantine" / "shard_000010_0000.bin").exists()


def test_output_lock_rejects_a_second_controller(tmp_path):
    out_dir = tmp_path / "campaign"
    with OutputLock(out_dir):
        with pytest.raises(RuntimeError, match="controller lock"):
            with OutputLock(out_dir):
                pass


def test_uncommitted_shards_do_not_change_durable_counts(tmp_path):
    out_dir = tmp_path / "campaign"
    out_dir.mkdir()
    _write_pair(out_dir, 2, accepted=False)
    (out_dir / "manifest.json").write_text(
        json.dumps({"schema": "zquoridor.four_million_selfplay.v1", "accepted_shards": []}),
        encoding="utf-8",
    )

    state = discover_resume_state(out_dir)

    assert state.counts == TargetCounts(0, 0, 0, {})
    assert (out_dir / "quarantine" / "shard_000002.bin").exists()


def test_reconcile_keeps_the_optional_weakness_corpus_in_central_counts(tmp_path):
    out_dir = tmp_path / "campaign"
    weakness = tmp_path / "weakness"
    weakness.mkdir()
    np.zeros(1, dtype=V3_DTYPE).tofile(weakness / "legacy.bin")

    store = reconcile_store(out_dir, (), weakness_corpora=(weakness,))
    try:
        assert _counts_from_store(store) == TargetCounts(1, 1, 0, {})
    finally:
        store.close()


def _write_fake_generator(path):
    path.write_text(
        """import struct
import sys
from pathlib import Path
import numpy as np

args = sys.argv[1:]
def value(name):
    return args[args.index(name) + 1]

dtype = np.dtype([
    ('own_pawn', 'u1'), ('opp_pawn', 'u1'), ('walls_h', '<u8'), ('walls_v', '<u8'),
    ('walls_left_own', 'i1'), ('walls_left_opp', 'i1'), ('nnue_eval', '<u2'),
    ('game_result', 'i1'), ('policy_target', '<u2'), ('own_dist', 'u1'),
    ('opp_dist', 'u1'), ('mover', 'u1'), ('own_cat_total', '<i2'),
    ('opp_cat_total', '<i2'), ('policy_top_idx', '<u2', (8,)),
    ('policy_top_prob', '<u2', (8,)),
])
index = int(value('--start-shard'))
row = np.zeros(2 if '--duplicate' in args else 1, dtype=dtype)
row['own_pawn'] = index + 1
row['walls_left_own'] = 10
row['walls_left_opp'] = 10
row['mover'] = 0
row['game_result'] = 1
row.tofile(Path(value('--out')))
Path(value('--meta-out')).write_bytes(struct.pack('<QfHHBBH', index, 0.5, 0, 1, 0, 0, 0) * len(row))
""",
        encoding="utf-8",
    )


def _write_schedule(path):
    rows = [{"opening_category": family, "history": ["e2"]} for family in REQUIRED_FAMILIES]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_fake_generator_completes_mixed_central_target_and_keeps_game_manifests(tmp_path):
    fake = tmp_path / "fake_generator.py"
    _write_fake_generator(fake)
    central = tmp_path / "central.jsonl"
    broad = tmp_path / "broad.jsonl"
    _write_schedule(central)
    broad.write_text(json.dumps({"history": ["e2"]}) + "\n", encoding="utf-8")
    weakness = tmp_path / "weakness"
    weakness.mkdir()
    legacy = np.zeros(1, dtype=V3_DTYPE)
    legacy["walls_left_own"] = 10
    legacy["walls_left_opp"] = 10
    legacy.tofile(weakness / "legacy.bin")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    out_dir = tmp_path / "campaign"

    args = [
        "--out", str(out_dir), "--exe", sys.executable, "--exe-arg", str(fake),
        "--weights", str(weights), "--central-schedule", str(central),
        "--broad-schedule", str(broad), "--weakness-corpus", str(weakness),
        "--total-target", "7", "--central-target", "6", "--broad-target", "1",
        "--family-floor", "1", "--games-per-shard", "1", "--seed-rows-per-shard", "1",
        "--threads-during-fine-tune", "1", "--threads-after-fine-tune", "1",
        "--fine-tune-threads", "0", "--cpu-thread-limit", "16",
    ]

    from tools.teacher.run_four_million_selfplay import parse_args

    assert run_controller(parse_args(args)) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["counts"] == {"total": 7, "central": 6, "broad": 1,
                                  "family": {name: 1 for name in REQUIRED_FAMILIES}}
    assert len(manifest["accepted_shards"]) == 6
    assert len(manifest["games"]) == 6
    assert all(entry["games"][0]["offsets"] == [0] for entry in manifest["accepted_shards"])
    assert all(entry["games"][0]["winner"] == 0 for entry in manifest["accepted_shards"])
    assert all(entry["games"][0]["result"] == 1.0 for entry in manifest["accepted_shards"])
    assert all(entry["games"][0]["termination"] == "goal" for entry in manifest["accepted_shards"])


def test_launch_intent_is_durable_before_child_spawn(tmp_path, monkeypatch):
    fake = tmp_path / "fake_generator.py"
    _write_fake_generator(fake)
    central = tmp_path / "central.jsonl"
    broad = tmp_path / "broad.jsonl"
    _write_schedule(central)
    broad.write_text(json.dumps({"history": ["e2"]}) + "\n", encoding="utf-8")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    out_dir = tmp_path / "campaign"
    from tools.teacher.run_four_million_selfplay import parse_args

    args = parse_args([
        "--out", str(out_dir), "--exe", sys.executable, "--exe-arg", str(fake),
        "--weights", str(weights), "--central-schedule", str(central),
        "--broad-schedule", str(broad), "--total-target", "2",
        "--central-target", "1", "--broad-target", "1", "--family-floor", "0",
        "--games-per-shard", "1", "--seed-rows-per-shard", "1",
        "--threads-during-fine-tune", "1", "--threads-after-fine-tune", "1",
        "--fine-tune-threads", "0",
    ])

    real_popen = subprocess.Popen
    observed = {}

    def checked_popen(command, **kwargs):
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        observed.update(manifest)
        return real_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", checked_popen)
    assert run_controller(args) == 0
    assert "launch_intent" in observed
    assert "active_shard" not in observed


def test_resume_marks_pid_reuse_stale_when_child_identity_differs(tmp_path, monkeypatch):
    out_dir = tmp_path / "campaign"
    staging = out_dir / "staging"
    staging.mkdir(parents=True)
    staged = staging / "shard_000000_0000.bin"
    staged.write_bytes(bytes(64))
    (out_dir / "manifest.json").write_text(json.dumps({
        "schema": "zquoridor.four_million_selfplay.v1",
        "accepted_shards": [],
        "active_shard": {
            "pid": 321,
            "process_identity": {"command_hash": "old"},
            "v3": "staging/shard_000000_0000.bin",
            "meta": "staging/shard_000000_0000.meta",
        },
    }), encoding="utf-8")
    monkeypatch.setattr("tools.teacher.run_four_million_selfplay._process_exists",
                        lambda pid: True)
    monkeypatch.setattr("tools.teacher.run_four_million_selfplay._process_identity",
                        lambda pid: {"command_hash": "new"})

    state = discover_resume_state(out_dir)

    assert state.next_shard == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["orphaned_shards"][0]["stale_reason"] == "child_identity_mismatch"
    assert (out_dir / "quarantine" / staged.name).exists()


def test_resume_blocks_an_unbound_launch_intent(tmp_path):
    out_dir = tmp_path / "campaign"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({
        "schema": "zquoridor.four_million_selfplay.v1",
        "accepted_shards": [],
        "launch_intent": {
            "index": 0,
            "attempt": 0,
            "v3": "staging/shard_000000_0000.bin",
            "meta": "staging/shard_000000_0000.meta",
        },
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unbound launch intent"):
        discover_resume_state(out_dir)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stale_launch_intents"][0]["index"] == 0


def test_game_entries_derive_winner_result_and_termination(tmp_path):
    v3_path = tmp_path / "game.bin"
    meta_path = tmp_path / "game.meta"
    rows = np.zeros(2, dtype=V3_DTYPE)
    rows[0]["game_result"] = 1
    rows[0]["mover"] = 0
    rows[1]["game_result"] = -1
    rows[1]["mover"] = 1
    rows.tofile(v3_path)
    np.asarray([(7, 0.5, 2, 4, 0, 0, 0), (7, 0.5, 1, 4, 0, 0, 0)], dtype=np.dtype([
        ("game", "<u8"), ("root", "<f4"), ("plies", "<u2"), ("length", "<u2"),
        ("source", "u1"), ("flags", "u1"), ("reserved", "<u2"),
    ])).tofile(meta_path)

    entries = _game_entries(v3_path, meta_path, seed_id=99, max_plies=140)

    assert entries == [{
        "game_id": 7,
        "seed_id": 99,
        "offsets": [0, 1],
        "mover_results": [-1, 1],
        "winner": 0,
        "result": 1.0,
        "termination": "goal",
        "termination_basis": "V3 game_result identifies the winner.",
    }]


def test_game_entries_label_draw_at_max_plies(tmp_path):
    v3_path = tmp_path / "game.bin"
    meta_path = tmp_path / "game.meta"
    rows = np.zeros(1, dtype=V3_DTYPE)
    rows[0]["game_result"] = 0
    rows[0]["mover"] = 1
    rows.tofile(v3_path)
    np.asarray([(8, 0.5, 3, 12, 0, 0, 0)], dtype=np.dtype([
        ("game", "<u8"), ("root", "<f4"), ("plies", "<u2"), ("length", "<u2"),
        ("source", "u1"), ("flags", "u1"), ("reserved", "<u2"),
    ])).tofile(meta_path)

    entry = _game_entries(v3_path, meta_path, seed_id=100, max_plies=12)[0]

    assert entry["winner"] == -1
    assert entry["result"] == 0.5
    assert entry["termination"] == "max_plies"


def test_resume_refuses_a_live_active_child_with_matching_identity(tmp_path, monkeypatch):
    out_dir = tmp_path / "campaign"
    staging = out_dir / "staging"
    staging.mkdir(parents=True)
    (staging / "shard_000000_0000.bin").write_bytes(bytes(64))
    (out_dir / "manifest.json").write_text(json.dumps({
        "schema": "zquoridor.four_million_selfplay.v1",
        "accepted_shards": [],
        "active_shard": {"pid": os.getpid(), "process_identity": {"command_hash": "current"},
                         "v3": "staging/shard_000000_0000.bin",
                         "meta": "staging/shard_000000_0000.meta"},
    }), encoding="utf-8")
    monkeypatch.setattr("tools.teacher.run_four_million_selfplay._process_identity",
                        lambda pid: {"command_hash": "current"})

    with pytest.raises(RuntimeError, match="active child"):
        discover_resume_state(out_dir)

    assert (staging / "shard_000000_0000.bin").exists()


def test_resume_records_and_quarantines_an_orphaned_child(tmp_path, monkeypatch):
    out_dir = tmp_path / "campaign"
    staging = out_dir / "staging"
    staging.mkdir(parents=True)
    (staging / "shard_000000_0000.bin").write_bytes(bytes(64))
    (out_dir / "manifest.json").write_text(json.dumps({
        "schema": "zquoridor.four_million_selfplay.v1",
        "accepted_shards": [],
        "active_shard": {"pid": 99999, "v3": "staging/shard_000000_0000.bin",
                         "meta": "staging/shard_000000_0000.meta"},
    }), encoding="utf-8")
    monkeypatch.setattr("tools.teacher.run_four_million_selfplay._process_exists",
                        lambda pid: False)

    discover_resume_state(out_dir)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["orphaned_shards"][0]["pid"] == 99999
    assert (out_dir / "quarantine" / "shard_000000_0000.bin").exists()


def test_fake_generator_admits_duplicate_observations_and_keeps_final_refs(tmp_path):
    fake = tmp_path / "fake_generator.py"
    _write_fake_generator(fake)
    central = tmp_path / "central.jsonl"
    broad = tmp_path / "broad.jsonl"
    _write_schedule(central)
    broad.write_text(json.dumps({"history": ["e2"]}) + "\n", encoding="utf-8")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    out_dir = tmp_path / "campaign"
    from tools.teacher.run_four_million_selfplay import parse_args

    args = parse_args([
        "--out", str(out_dir), "--exe", sys.executable, "--exe-arg", str(fake),
        "--exe-arg=--duplicate", "--weights", str(weights),
        "--central-schedule", str(central), "--broad-schedule", str(broad),
        "--total-target", "6", "--central-target", "5", "--broad-target", "1",
        "--family-floor", "1", "--games-per-shard", "1", "--seed-rows-per-shard", "1",
        "--threads-during-fine-tune", "1", "--threads-after-fine-tune", "1",
        "--fine-tune-threads", "0",
    ])

    assert run_controller(args) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all(entry["accepted_records"] == 2 for entry in manifest["accepted_shards"])
    assert all(entry["unique_records"] == 1 for entry in manifest["accepted_shards"])
    assert all(entry["equal_records"] == 1 for entry in manifest["accepted_shards"])
    store = UniqueStateStore(out_dir / "states.sqlite")
    try:
        ref = store.lookup(UniqueStateStore.canonical_key(np.memmap(
            out_dir / "shard_000000.bin", dtype=V3_DTYPE, mode="r")[0]))
        assert ref is not None
        assert ref.shard == str(out_dir / "shard_000000.bin")
    finally:
        store.close()


def test_fake_generator_preserves_the_first_pool_label_on_a_stronger_duplicate(tmp_path):
    fake = tmp_path / "fake_generator.py"
    _write_fake_generator(fake)
    central = tmp_path / "central.jsonl"
    broad = tmp_path / "broad.jsonl"
    _write_schedule(central)
    broad.write_text(json.dumps({"history": ["e2"]}) + "\n", encoding="utf-8")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    out_dir = tmp_path / "campaign"

    from tools.teacher.run_four_million_selfplay import parse_args

    args = parse_args([
        "--out", str(out_dir), "--exe", sys.executable, "--exe-arg", str(fake),
        "--weights", str(weights), "--central-schedule", str(central),
        "--broad-schedule", str(broad), "--total-target", "6",
        "--central-target", "5", "--broad-target", "1", "--family-floor", "1",
        "--games-per-shard", "1", "--seed-rows-per-shard", "1",
        "--threads-during-fine-tune", "1", "--threads-after-fine-tune", "1",
        "--fine-tune-threads", "0", "--reliability", "2",
    ])

    assert run_controller(args) == 0
    store = UniqueStateStore(out_dir / "states.sqlite")
    try:
        from tools.teacher.selfplay_unique_store import RecordRef

        key = b"cross-pool-duplicate"
        assert store.accept(key, 1.0, RecordRef("old.bin", 0, "central", "front_wall")).inserted
        assert store.accept(key, 2.0, RecordRef("new.bin", 0, "broad", None)).replaced
        labels = store.db.execute(
            "SELECT source, family FROM states ORDER BY rowid"
        ).fetchall()
        assert ("central", "front_wall") in labels
    finally:
        store.close()


def test_invalid_thread_reservation_is_rejected_before_file_checks(tmp_path):
    from tools.teacher.run_four_million_selfplay import parse_args

    args = parse_args([
        "--out", str(tmp_path / "campaign"), "--exe", "missing.exe", "--weights", "missing.bin",
        "--central-schedule", "central.jsonl", "--broad-schedule", "broad.jsonl",
        "--threads-during-fine-tune", "10", "--fine-tune-threads", "7",
    ])

    with pytest.raises(ValueError, match="exceed the CPU limit"):
        run_controller(args)


def test_direct_script_help_uses_the_repository_import_path():
    completed = subprocess.run(
        [sys.executable, "tools/teacher/run_four_million_selfplay.py", "--help"],
        cwd="C:/Projetos/Zquoridor",
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--central-schedule" in completed.stdout
