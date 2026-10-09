from pathlib import Path

import numpy as np
import pytest

from training.mix_experimental_datasets import load_arrays, mix_datasets


def state(pawn, dist=3):
    return {
        "own_pawn": pawn,
        "opp_pawn": 72 - pawn,
        "walls_h": np.uint64(1 << (pawn % 40)),
        "walls_v": np.uint64(1 << ((pawn + 1) % 40)),
        "walls_left_own": 4,
        "walls_left_opp": 5,
        "own_dist": dist,
        "opp_dist": dist + 1,
    }


def write_npz(path: Path, rows):
    data = {key: [] for key in (*state(0).keys(), "policy", "value", "weight", "is_val", "group_id")}
    for board, value, action, weight, is_val, group in rows:
        for key, item in board.items():
            data[key].append(item)
        policy = np.zeros(209, dtype=np.float32)
        policy[action] = 1.0
        data["policy"].append(policy)
        data["value"].append(value)
        data["weight"].append(weight)
        data["is_val"].append(is_val)
        data["group_id"].append(group)
    arrays = {k: np.asarray(v, dtype="U32" if k == "group_id" else None) for k, v in data.items()}
    np.savez(path, **arrays)


def test_deduplicates_neutrally_and_preserves_split_mass(tmp_path):
    x, y, z, v = state(10), state(11), state(12), state(13)
    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    write_npz(old, [
        (x, 0.0, 0, 1.0, False, "old-x0"),
        (x, 1.0, 1, 100.0, False, "old-x1"),
        (y, -0.5, 2, 1.0, False, "old-y"),
        (z, 0.4, 3, 1.0, False, "old-z-train"),
        (state(14), 0.1, 4, 1.0, False, "group-leak"),
        (state(15), -0.1, 5, 1.0, True, "group-leak"),
        (v, 0.25, 6, 1.0, True, "old-v"),
    ])
    write_npz(new, [
        (x, 0.2, 7, 1.0, False, "new-x"),
        (y, 0.5, 8, 1.0, False, "new-y"),
        (z, -0.8, 9, 1.0, True, "new-z-val"),
        (v, 0.75, 10, 1.0, True, "new-v"),
    ])
    out = tmp_path / "mixed"
    manifest = mix_datasets([
        {"name": "old", "path": str(old), "fraction": 0.5, "weight_cap": 30},
        {"name": "new", "path": str(new), "fraction": 0.5, "weight_cap": None},
    ], out, chunk_size=2)

    arrays = load_arrays(out)
    assert manifest["train_samples"] == 2  # x and y; z is cross-split contaminated
    assert manifest["val_samples"] == 1  # v
    assert manifest["sources"][0]["removed_group_count"] == 1
    assert manifest["sources"][0]["removed_split_conflict_rows"] == 1
    assert manifest["sources"][1]["removed_split_conflict_rows"] == 1
    assert np.allclose(arrays["weight"], arrays["source_mass"].sum(axis=1), atol=1e-6)

    xrow = int(np.flatnonzero(arrays["own_pawn"] == 10)[0])
    # Old-source x target is 1:30 weighted after the cap, then merged with new.
    # Old raw train mass is 32, new raw train mass is 2, and each source owns
    # one unit of the two-row train mixture.
    expected_x = ((1 / 32) * 0.0 + (30 / 32) * 1.0 + 0.5 * 0.2) / ((31 / 32) + 0.5)
    assert arrays["value"][xrow] == pytest.approx(expected_x, abs=1e-6)
    total_x_mass = (31 / 32) + 0.5
    assert arrays["policy"][xrow, 0] == pytest.approx((1 / 32) / total_x_mass, abs=1e-6)
    assert arrays["policy"][xrow, 1] == pytest.approx((30 / 32) / total_x_mass, abs=1e-6)
    assert arrays["policy"][xrow, 7] == pytest.approx(0.5 / total_x_mass, abs=1e-6)

    for source in manifest["sources"]:
        assert source["train_effective_mass"] == pytest.approx(1.0)
        assert source["val_effective_mass"] == pytest.approx(0.5)
    assert manifest["group_separation_verified"] is True
    assert manifest["source_names"] == ["old", "new"]


def test_cache_resume_identity_and_source_change_rejection(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    board = state(20)
    rows = [(board, 0.2, 1, 1.0, False, "t"),
            (state(21), -0.1, 2, 1.0, True, "v")]
    write_npz(a, rows)
    write_npz(b, rows)
    out = tmp_path / "out"
    sources = [{"name": "a", "path": str(a), "fraction": 0.5, "weight_cap": None},
               {"name": "b", "path": str(b), "fraction": 0.5, "weight_cap": None}]
    first = mix_datasets(sources, out, chunk_size=1)
    second = mix_datasets(sources, out, chunk_size=1)
    assert first["identity_sha256"] == second["identity_sha256"]
    with np.load(b) as z:
        changed = {k: z[k] for k in z.files}
    changed["value"] = changed["value"].copy()
    changed["value"][0] = 0.8
    np.savez(b, **changed)
    with pytest.raises(ValueError, match="identity"):
        mix_datasets(sources, out, chunk_size=1)


def test_pool_uses_all_records_proportional_to_eligible_weight_mass(tmp_path):
    old, local, colab = (tmp_path / name for name in ("old.npz", "local.npz", "colab.npz"))
    # Even extreme historical weights remain within the 25% global budget.
    write_npz(old, [(state(1), 0.1, 0, 1000.0, False, "ot"),
                    (state(2), -0.1, 1, 34.1, True, "ov")])
    write_npz(local, [(state(10), 0.2, 2, 1.0, False, "lt0"),
                      (state(11), 0.3, 3, 1.0, False, "lt1"),
                      (state(12), 0.4, 4, 1.0, True, "lv")])
    write_npz(colab, [(state(20), -0.2, 5, 1.0, False, "ct"),
                      (state(21), -0.4, 6, 1.0, True, "cv")])
    manifest = mix_datasets([
        {"name": "historical", "path": str(old), "fraction": 0.25, "weight_cap": 30},
        {"name": "local", "path": str(local), "fraction": None, "pool": "new",
         "pool_fraction": 0.75, "weight_cap": None},
        {"name": "colab", "path": str(colab), "fraction": None, "pool": "new",
         "pool_fraction": 0.75, "weight_cap": None},
    ], tmp_path / "pool-mix", chunk_size=2)
    by_name = {row["name"]: row for row in manifest["sources"]}
    assert by_name["historical"]["effective_fraction_train"] == pytest.approx(0.25)
    assert by_name["historical"]["effective_fraction_val"] == pytest.approx(0.25)
    assert by_name["local"]["effective_fraction_train"] == pytest.approx(0.5)
    assert by_name["colab"]["effective_fraction_train"] == pytest.approx(0.25)
    assert by_name["local"]["effective_fraction_val"] == pytest.approx(0.375)
    assert by_name["colab"]["effective_fraction_val"] == pytest.approx(0.375)
