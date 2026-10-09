import struct

import pytest

import numpy as np

from tools.teacher.selfplay_unique_store import (META_RECORD_SIZE, QUALITY_MISSING_ROOT_VALUE,
                                                 RecordRef, UniqueStateStore, V3_DTYPE)


KEY = b"canonical-state-key"


def ref(i):
    return RecordRef("shard.bin", i, "central", "front_wall")


def test_duplicate_state_does_not_increase_unique_count(tmp_path):
    store = UniqueStateStore(tmp_path / "states.sqlite")
    assert store.accept(KEY, 1.0, ref(0)).inserted
    assert not store.accept(KEY, 0.5, ref(1)).inserted
    assert store.unique_count == 1


def test_higher_reliability_replaces_record_reference(tmp_path):
    store = UniqueStateStore(tmp_path / "states.sqlite")
    store.accept(KEY, 1.0, ref(0))
    decision = store.accept(KEY, 2.0, ref(1))
    assert decision.replaced
    assert store.lookup(KEY) == ref(1)


def test_equal_reliability_does_not_increase_unique_count(tmp_path):
    store = UniqueStateStore(tmp_path / "states.sqlite")
    store.accept(KEY, 1.0, ref(0))
    assert store.accept(KEY, 1.0, ref(1)).equal
    assert store.unique_count == 1


def test_invalid_shard_rolls_back(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(bytes(64))
    store = UniqueStateStore(tmp_path / "states.sqlite")
    with pytest.raises(ValueError):
        store.admit_shard(path, keys=[], reliability=1.0)
    assert store.unique_count == 0


def test_reopen_preserves_counts(tmp_path):
    path = tmp_path / "states.sqlite"
    store = UniqueStateStore(path)
    store.accept(KEY, 1.0, ref(0))
    store.close()
    reopened = UniqueStateStore(path)
    assert reopened.unique_count == 1
    assert reopened.counts()["raw"] == 1


def test_canonical_key_excludes_labels(tmp_path):
    row = np.zeros((), dtype=V3_DTYPE)
    row["own_pawn"], row["opp_pawn"], row["mover"] = 3, 75, 1
    row["walls_left_own"], row["walls_left_opp"] = 4, 7
    a = UniqueStateStore.canonical_key(row)
    row["game_result"], row["policy_target"], row["nnue_eval"] = -1, 208, 123
    assert a == UniqueStateStore.canonical_key(row)


def test_populated_top_policy_index_is_validated(tmp_path):
    path = tmp_path / "bad.bin"
    row = np.zeros(1, dtype=V3_DTYPE)
    row["policy_top_idx"][0, 0] = 209
    row["policy_top_prob"][0, 0] = 1
    row.tofile(path)
    with pytest.raises(ValueError):
        UniqueStateStore(tmp_path / "states.sqlite").admit_shard(path)


def test_terminal_distance_is_validated(tmp_path):
    path = tmp_path / "bad.bin"
    meta = tmp_path / "bad.meta"
    np.zeros(1, dtype=V3_DTYPE).tofile(path)
    meta.write_bytes(struct.pack("<QfHHBBH", 1, 0.5, 4, 3, 0, 0, 0))
    with pytest.raises(ValueError):
        UniqueStateStore(tmp_path / "states.sqlite").admit_shard(path, meta)


def test_rejected_counters_persist_by_scope(tmp_path):
    path = tmp_path / "states.sqlite"
    store = UniqueStateStore(path)
    bad = ref(0)
    store.accept(b"", 1.0, bad)
    store.accept(KEY, float("nan"), bad)
    store.close()
    reopened = UniqueStateStore(path)
    assert reopened.counts() == {"rejected": 2}
    assert reopened.counts("source:central")["rejected"] == 2
    assert reopened.counts("family:front_wall")["rejected"] == 2


def test_transaction_rolls_back_after_partial_mutation(tmp_path):
    class FailingStore(UniqueStateStore):
        def accept(self, key, reliability, ref, quality_flags=0):
            result = super().accept(key, reliability, ref, quality_flags)
            if ref.offset == 1:
                raise RuntimeError("injected failure")
            return result

    path = tmp_path / "states.sqlite"
    v3 = tmp_path / "rows.bin"
    np.zeros(2, dtype=V3_DTYPE).tofile(v3)
    store = FailingStore(path)
    with pytest.raises(RuntimeError):
        store.admit_shard(v3, keys=[b"a", b"b"], source="central", family="front_wall",
                          missing_metadata=True)
    assert store.unique_count == 0
    assert store.counts() == {}
    assert store.db.execute("SELECT COUNT(*) FROM state_metadata").fetchone()[0] == 0


def test_missing_metadata_flag_persists(tmp_path):
    v3 = tmp_path / "rows.bin"
    np.zeros(1, dtype=V3_DTYPE).tofile(v3)
    store = UniqueStateStore(tmp_path / "states.sqlite")
    store.admit_shard(v3, keys=[b"canonical"], missing_metadata=True)
    assert store.db.execute("SELECT quality_flags FROM state_metadata").fetchone()[0] == QUALITY_MISSING_ROOT_VALUE


def test_higher_reliability_real_metadata_replaces_legacy_flag(tmp_path):
    path = tmp_path / "states.sqlite"
    store = UniqueStateStore(path)
    store.accept(KEY, 1.0, ref(0), QUALITY_MISSING_ROOT_VALUE)
    store.accept(KEY, 2.0, ref(1), 0)
    assert store.lookup(KEY) == ref(1)
    assert store.db.execute("SELECT quality_flags FROM state_metadata WHERE key=?", (KEY,)).fetchone()[0] == 0
    store.close()
    reopened = UniqueStateStore(path)
    assert reopened.lookup(KEY) == ref(1)
    assert reopened.db.execute("SELECT quality_flags FROM state_metadata WHERE key=?", (KEY,)).fetchone()[0] == 0


def test_equal_reliability_duplicate_preserves_selected_missing_root_flag(tmp_path):
    store = UniqueStateStore(tmp_path / "states.sqlite")
    store.accept(KEY, 1.0, ref(0), QUALITY_MISSING_ROOT_VALUE)
    decision = store.accept(KEY, 1.0, ref(1), 0)
    assert decision.equal
    assert store.lookup(KEY) == ref(0)
    assert store.db.execute("SELECT quality_flags FROM state_metadata WHERE key=?", (KEY,)).fetchone()[0] == QUALITY_MISSING_ROOT_VALUE


def test_lower_reliability_duplicate_preserves_selected_missing_root_flag(tmp_path):
    store = UniqueStateStore(tmp_path / "states.sqlite")
    store.accept(KEY, 2.0, ref(0), QUALITY_MISSING_ROOT_VALUE)
    decision = store.accept(KEY, 1.0, ref(1), 0)
    assert decision.rejected
    assert store.lookup(KEY) == ref(0)
    assert store.db.execute("SELECT quality_flags FROM state_metadata WHERE key=?", (KEY,)).fetchone()[0] == QUALITY_MISSING_ROOT_VALUE
