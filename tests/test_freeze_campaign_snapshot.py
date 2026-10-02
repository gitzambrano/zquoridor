import hashlib
import json

import numpy as np
import pytest

from training.freeze_campaign_snapshot import freeze_campaign_snapshot
from training.prepare_stored_replay_all import (
    META_DTYPE,
    _opening_bank,
    _opening_family,
    _resolve_source,
)
from training.read_selfplay import SAMPLE_DTYPE


def test_freezes_partial_campaign_without_changing_live_manifest(tmp_path):
    source = tmp_path / "source"
    shard_dir = source / "production_bucketed"
    shard_dir.mkdir(parents=True)
    binary = shard_dir / "sample.bin"
    metadata = shard_dir / "sample.meta"
    np.zeros(2, dtype=SAMPLE_DTYPE).tofile(binary)
    np.zeros(2, dtype=META_DTYPE).tofile(metadata)
    bank = source / "opening_bank_8to10ply.jsonl"
    bank.write_text('{"opening_index":"family-1"}\n', encoding="utf-8")
    bank_sha = hashlib.sha256(bank.read_bytes()).hexdigest()

    campaign = {
        "schema": "test.campaign",
        "status": "running",
        "assigned_target": 12000000,
        "campaign_total_target": 16000000,
        "reserve_unassigned": 4000000,
        "games_per_shard": 2,
        "bank_fingerprint": "bank-fingerprint",
        "bank_sha256": bank_sha,
        "profiles": [{"name": "wide"}],
        "sources": [{"name": "production_bucketed"}],
        "source_plan": {"production_bucketed": 12000000},
        "shards": [{
            "index": 0,
            "source": "production_bucketed",
            "profile": "wide",
            "seed": 123,
            "v3": "production_bucketed/sample.bin",
            "meta": "sample.meta",
            "records": 2,
            "games": 2,
        }],
    }
    campaign_path = source / "campaign_manifest.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    original_bytes = campaign_path.read_bytes()

    snapshot_path = freeze_campaign_snapshot(source)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot_path.name == "frozen_partial_manifest.json"
    assert campaign_path.read_bytes() == original_bytes
    assert snapshot["complete"] is True
    assert snapshot["campaign_complete"] is False
    assert snapshot["completion_scope"] == "all_listed_accepted_shards_frozen_and_hash_verified"
    assert snapshot["original_target"]["campaign_total_target"] == 16000000
    assert snapshot["original_target"]["reserve_unassigned"] == 4000000
    assert snapshot["total_positions"] == 2
    assert snapshot["profiles"] == campaign["profiles"]
    assert snapshot["source_plan"] == campaign["source_plan"]
    assert snapshot["opening_bank"]["sha256"] == bank_sha
    shard = snapshot["accepted_shards"][0]
    assert shard["bin_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert shard["meta_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()

    resolved = _resolve_source(snapshot_path)
    assert resolved["campaign"] is False
    bank_info = _opening_bank(resolved)
    assert bank_info is not None
    assert _opening_family(bank_info, 0, resolved["shards"][0], resolved) == "family-1"

    binary.write_bytes(binary.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA mismatch"):
        _resolve_source(snapshot_path)


def test_rejects_unaligned_or_record_count_mismatch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    binary = source / "sample.bin"
    metadata = source / "sample.meta"
    binary.write_bytes(b"x" * SAMPLE_DTYPE.itemsize)
    metadata.write_bytes(b"x" * META_DTYPE.itemsize)
    campaign_path = source / "campaign_manifest.json"
    campaign_path.write_text(json.dumps({
        "status": "running",
        "shards": [{"v3": "sample.bin", "meta": "sample.meta", "records": 2}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="record count does not match"):
        freeze_campaign_snapshot(source)
    assert not (source / "frozen_partial_manifest.json").exists()
