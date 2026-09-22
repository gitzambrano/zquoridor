import numpy as np
import pytest

from prepare_replay import normalize_visits, stored_value_target, METADATA_DTYPE, sample_stored_states, CONFIG


def test_normalize_visits_returns_dense_policy():
    result = normalize_visits([3, 1], [4, 2])
    assert result.shape == (209,)
    np.testing.assert_allclose(result[[3, 1]], [2 / 3, 1 / 3])
    assert float(result.sum()) == pytest.approx(1.0)


def test_stored_value_bootstraps_terminal_result_with_remaining_ply_discount():
    # Signed root evidence and terminal evidence are symmetric. The discount
    # reduces the magnitude of both wins and losses.
    assert stored_value_target(0.75, 1, 4, 0.5) == pytest.approx(0.28125)
    assert stored_value_target(0.75, -1, 4, 0.5) == pytest.approx(0.21875)
    assert stored_value_target(0.75, 0, 4, 0.5) == pytest.approx(0.25)


def test_metadata_layout_is_the_packed_v1_sidecar():
    assert METADATA_DTYPE.itemsize == 20


def test_signed_targets_preserve_perspective_and_draw():
    assert stored_value_target(0.75, 1, 4, 0.99) == pytest.approx(
        -stored_value_target(0.25, -1, 4, 0.99))
    assert stored_value_target(0.5, 0, 4, 0.99) == 0
    with pytest.raises(ValueError):
        stored_value_target(float("nan"), 1, 4, 0.99)


def test_stored_manifest_filters_duplicates_and_transient_files(tmp_path):
    import json
    from tools.teacher.selfplay_unique_store import V3_DTYPE
    rows = np.zeros(5, dtype=V3_DTYPE)
    rows["own_pawn"] = [4, 5, 6, 4, 7]
    rows["opp_pawn"] = 76
    rows["walls_left_own"] = rows["walls_left_opp"] = 10
    rows["own_dist"] = rows["opp_dist"] = 8
    rows["policy_top_idx"][:, 0] = 13
    rows["policy_top_prob"][:, 0] = 65535
    rows["game_result"] = 1
    meta = np.zeros(5, dtype=METADATA_DTYPE)
    meta["game"] = [10, 11, 12, 13, 14]
    meta["root"] = 0.6
    meta["flags"] = 2
    meta["plies"] = 3
    meta["length"] = 10
    meta["flags"][-1] = 1
    meta["root"][-1] = np.nan
    rows.tofile(tmp_path / "shard.bin")
    meta.tofile(tmp_path / "shard.meta")
    (tmp_path / "manifest.json").write_text(json.dumps({"accepted_shards": [{"v3": "shard.bin"}]}))
    (tmp_path / "quarantine").mkdir()
    (tmp_path / "quarantine" / "bad.bin").write_bytes(b"invalid")
    data, report = sample_stored_states(dict(CONFIG, source=str(tmp_path), max_positions=10))
    assert len(data["id"]) == 3
    assert report["skipped_missing_root"] == 1
    assert not (set(data["group_id"][data["is_val"]]) & set(data["group_id"][~data["is_val"]]))
    assert len(set(data["own_pawn"])) == 3


def test_zero_mass_unused_indices_and_invalid_visits():
    assert normalize_visits([13, 65535], [1, 0])[13] == 1
    with pytest.raises(ValueError):
        normalize_visits([209], [1])
    with pytest.raises(ValueError):
        normalize_visits([13], [float("nan")])
