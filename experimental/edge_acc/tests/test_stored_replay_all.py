import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training.prepare_stored_replay_all import META_DTYPE, META_ROOT_VALID, _partition, prepare
from training.read_selfplay import SAMPLE_DTYPE


def make_rows(specs):
    rows = np.zeros(len(specs), dtype=SAMPLE_DTYPE)
    meta = np.zeros(len(specs), dtype=META_DTYPE)
    for i, spec in enumerate(specs):
        state, game, root, result, action, plies = spec
        rows[i]["own_pawn"] = state
        rows[i]["opp_pawn"] = 80 - state
        rows[i]["own_dist"] = 8
        rows[i]["opp_dist"] = 9
        rows[i]["walls_left_own"] = 5
        rows[i]["walls_left_opp"] = 5
        rows[i]["game_result"] = result
        rows[i]["policy_top_idx"] = np.arange(8, dtype=np.uint16)
        rows[i]["policy_top_prob"] = 0
        rows[i]["policy_top_idx"][0] = action
        rows[i]["policy_top_prob"][0] = 65535
        meta[i]["game"] = game
        meta[i]["root"] = root
        meta[i]["plies"] = plies
        meta[i]["length"] = 10
        meta[i]["flags"] = META_ROOT_VALID
    return rows, meta


def write_campaign(root: Path, specs, extras=()):
    root.mkdir(parents=True)
    entries = []
    for binary_name, meta_name, source_name, rowspecs in [
            ("production_bucketed/part.bin", "part.meta", "local", specs), *extras]:
        rows, meta = make_rows(rowspecs)
        binary, sidecar = root / binary_name, root / meta_name
        binary.parent.mkdir(parents=True, exist_ok=True)
        sidecar = binary.with_suffix(".meta")
        rows.tofile(binary)
        meta.tofile(sidecar)
        entries.append({"v3": str(Path(binary_name)), "meta": meta_name, "source": source_name,
                        "seed": 44, "bin_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "meta_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()})
    manifest = {
        "schema": "frozen-test-campaign.v1",
        "status": "assigned_sources_complete_reserve_unassigned",
        "seed": 44,
        "reserve_unassigned": 4000,
        "games_per_shard": 512,
        "shards": entries,
    }
    (root / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return binary, sidecar


def test_disk_replay_averages_targets_and_keeps_whole_game_splits(tmp_path):
    seed = 73
    train_game = next(i for i in range(1, 10000) if not _partition(seed, str(i), 0.5)[0])
    val_game = next(i for i in range(train_game + 1, 20000) if _partition(seed, str(i), 0.5)[0])
    # State 5 sorts before retained states and must be excluded from both splits.
    # Source provenance must advance past its removed rows. State 10 is
    # duplicated inside one training game; its policy/value are averaged.
    specs = [
        (10, train_game, 0.25, 1, 3, 0),
        (10, train_game, 0.75, -1, 5, 1),
        (5, train_game, 0.50, 1, 6, 2),
        (20, val_game, 0.50, -1, 7, 1),
        (5, val_game, 0.50, 1, 8, 2),
    ]
    source = tmp_path / "frozen"
    write_campaign(source, specs, extras=[("contact_bucketed/part2.bin", "part2.meta", "colab", [
        (10, train_game, 0.9, 1, 4, 3),
        (20, val_game, 0.8, -1, 9, 2),
    ])])
    out = tmp_path / "replay-all-50-50" / "dataset"
    manifest = prepare(source, out, seed=seed, val_fraction=0.5, chunk_size=2)
    assert manifest["stored_gamma"] == 1.0
    assert manifest["stored_outcome_weight"] == 0.5
    assert manifest["cross_split_conflict_states_removed"] == 1
    assert manifest["samples"] == 2
    assert manifest["train_samples"] == 1
    assert manifest["val_samples"] == 1

    values = np.load(out / "fields" / "value.npy", mmap_mode="r")
    policy = np.load(out / "fields" / "policy.npy", mmap_mode="r")
    pawns = np.load(out / "fields" / "own_pawn.npy", mmap_mode="r")
    split = np.load(out / "fields" / "is_val.npy", mmap_mode="r")
    groups = np.load(out / "fields" / "group_id.npy", mmap_mode="r")
    row = int(np.flatnonzero(pawns == 10)[0])
    assert values[row] == pytest.approx(0.3, abs=1e-6)  # all roots/results contribute equally
    assert policy[row, 3] == pytest.approx(1 / 3)
    assert policy[row, 4] == pytest.approx(1 / 3)
    assert policy[row, 5] == pytest.approx(1 / 3)
    assert split[row] is np.False_
    assert groups[row].decode("ascii")
    assert manifest["cross_split_conflict_records_removed"] == 2
    source_count = np.load(out / "fields" / "network_source_count.npy", mmap_mode="r")
    assert source_count[row].tolist() == [1, 2]  # colab, local in sorted source order
    assert source_count.sum() == 5
    assert manifest["shared_states_across_sources"] == 2
    assert manifest["mean_pairwise_root_disagreement"] == pytest.approx(0.35)
    assert manifest["profile_counts"] == {"unknown": 7}
    assert sum(manifest["rollout_ply_counts"].values()) == 7
    assert sum(manifest["opening_family_counts"].values()) == 4
    assert manifest["opening_family_counts"] == {"unknown": 4}
    replay = json.loads((out.parent / "replay_manifest.json").read_text(encoding="utf-8"))
    assert replay["dataset_manifest_sha256"] == hashlib.sha256(
        (out / "dataset.manifest.json").read_bytes()).hexdigest()


def test_refuses_missing_or_misaligned_sidecar(tmp_path):
    root = tmp_path / "frozen"
    binary, sidecar = write_campaign(root, [(2, 3, 0.5, 1, 1, 0)])
    sidecar.write_bytes(b"")
    manifest_path = root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["meta_sha256"] = hashlib.sha256(b"").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="counts are not aligned"):
        prepare(root, tmp_path / "out" / "dataset", seed=4, val_fraction=0.5, chunk_size=2)


def test_resume_checks_output_sha_and_requires_frozen_campaign(tmp_path):
    seed = 9
    train_game = next(i for i in range(1, 1000) if not _partition(seed, str(i), 0.5)[0])
    val_game = next(i for i in range(train_game + 1, 2000) if _partition(seed, str(i), 0.5)[0])
    root = tmp_path / "frozen"
    write_campaign(root, [(1, train_game, 0.4, 1, 1, 1),
                          (2, val_game, 0.6, -1, 2, 1)])
    out = tmp_path / "resume" / "dataset"
    result = prepare(root, out, seed=seed, val_fraction=0.5, chunk_size=1)
    replay = json.loads((out.parent / "replay_manifest.json").read_text(encoding="utf-8"))
    assert replay["complete"] is True
    assert replay["config"]["stored_gamma"] == 1.0
    assert replay["config"]["stored_outcome_weight"] == 0.5
    assert prepare(root, out, seed=seed, val_fraction=0.5, chunk_size=1)["identity_sha256"] == result["identity_sha256"]
    (out.parent / "replay_manifest.json").unlink()
    prepare(root, out, seed=seed, val_fraction=0.5, chunk_size=1)
    assert (out.parent / "replay_manifest.json").is_file()  # recover interrupted finalization
    arrays_file = out / result["arrays"]["policy"]["filename"]
    arrays_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA mismatch"):
        prepare(root, out, seed=seed, val_fraction=0.5, chunk_size=1)
