#!/usr/bin/env python3
"""Freeze the currently accepted shards of a local self-play campaign.

The output is a separate immutable input manifest. Its ``complete`` flag means
the listed snapshot files and hashes are complete; ``campaign_complete`` stays
false because the campaign target may still be partial.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "source": "data/selfplay/central-weakness-rollouts-16m",
    "out_manifest": None,
}
MANIFEST_NAME = "frozen_partial_manifest.json"
V3_RECORD_BYTES = 64
META_RECORD_BYTES = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while snapshotting: {path}")
    return after.st_size, digest


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"campaign path escapes source directory: {value}")
    return path


def _metadata_path(root: Path, binary: Path, entry: dict) -> Path:
    name = entry.get("meta_path") or entry.get("meta")
    if isinstance(name, dict):
        name = name.get("path")
    if not isinstance(name, str):
        return binary.with_suffix(".meta")
    candidate = _inside(root, name)
    if candidate.is_file():
        return candidate
    if Path(name).name == name:
        return _inside(root, str(binary.relative_to(root).parent / name))
    return candidate


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze_campaign_snapshot(source, out_manifest=None) -> Path:
    """Write a hash-verified accepted-shard snapshot beside a campaign.

    The campaign manifest is only read. Raw shards remain in place and are
    checked again by the replay preparer against the hashes in the snapshot.
    """
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path = source_path.resolve()
    root = source_path.parent if source_path.is_file() else source_path
    campaign_path = source_path if source_path.is_file() else root / "campaign_manifest.json"
    if campaign_path.name != "campaign_manifest.json" or not campaign_path.is_file():
        raise ValueError(f"source must be a campaign directory or campaign_manifest.json: {source_path}")

    stat_before = campaign_path.stat()
    raw_manifest = campaign_path.read_bytes()
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    campaign = json.loads(raw_manifest)
    stat_after = campaign_path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError(f"campaign manifest changed while snapshotting: {campaign_path}")

    entries = campaign.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("campaign has no accepted shards to freeze")

    accepted = []
    source_positions = {}
    profile_positions = {}
    total_records = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("campaign shard entry must be an object")
        binary_name = entry.get("v3") or entry.get("bin") or entry.get("path")
        if not isinstance(binary_name, str):
            raise ValueError("campaign shard entry lacks a V3 path")
        binary = _inside(root, binary_name)
        metadata = _metadata_path(root, binary, entry)
        binary_size, binary_sha = _stable_file(binary)
        metadata_size, metadata_sha = _stable_file(metadata)
        if binary_size % V3_RECORD_BYTES:
            raise ValueError(f"V3 shard is not {V3_RECORD_BYTES}-byte aligned: {binary}")
        if metadata_size % META_RECORD_BYTES:
            raise ValueError(f"metadata shard is not {META_RECORD_BYTES}-byte aligned: {metadata}")
        records = binary_size // V3_RECORD_BYTES
        if metadata_size // META_RECORD_BYTES != records:
            raise ValueError(f"V3 and metadata record counts differ: {binary}")
        declared_records = entry.get("records")
        if declared_records is not None and int(declared_records) != records:
            raise ValueError(f"campaign record count does not match shard size: {binary}")

        frozen = dict(entry)
        frozen.update(v3=binary.relative_to(root).as_posix(),
                      meta_path=metadata.relative_to(root).as_posix(),
                      records=records, bin_sha256=binary_sha, meta_sha256=metadata_sha)
        accepted.append(frozen)
        total_records += records
        source = str(entry.get("source", "unknown"))
        profile = str(entry.get("profile", "unknown"))
        source_positions[source] = source_positions.get(source, 0) + records
        profile_positions[profile] = profile_positions.get(profile, 0) + records

    if _sha256(campaign_path) != manifest_sha:
        raise RuntimeError(f"campaign manifest changed while snapshotting: {campaign_path}")

    bank_path = root / "opening_bank_8to10ply.jsonl"
    bank_info = {"path": bank_path.name, "available": False, "sha256": None}
    expected_bank_sha = campaign.get("bank_sha256")
    if bank_path.is_file():
        _, bank_sha = _stable_file(bank_path)
        if expected_bank_sha and bank_sha != expected_bank_sha:
            raise ValueError(f"campaign opening bank SHA mismatch: {bank_path}")
        bank_info.update(available=True, sha256=bank_sha)
    elif expected_bank_sha:
        raise FileNotFoundError(f"campaign opening bank is missing: {bank_path}")

    out_path = Path(out_manifest) if out_manifest else root / MANIFEST_NAME
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path = out_path.resolve()
    if out_path.parent != root:
        raise ValueError("snapshot manifest must be beside the campaign so shard paths stay contained")

    payload = {
        "schema": "zquoridor.selfplay.v3.frozen_partial_snapshot.v1",
        "complete": True,
        "completion_scope": "all_listed_accepted_shards_frozen_and_hash_verified",
        "campaign_complete": False,
        "campaign_status_at_snapshot": campaign.get("status"),
        "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign_manifest": {
            "path": campaign_path.name,
            "sha256": manifest_sha,
        },
        "campaign_provenance": {key: value for key, value in campaign.items() if key != "shards"},
        "original_target": {
            "assigned_target": campaign.get("assigned_target"),
            "campaign_total_target": campaign.get("campaign_total_target"),
            "reserve_unassigned": campaign.get("reserve_unassigned"),
            "progress": campaign.get("progress"),
        },
        "total_shards": len(accepted),
        "total_positions": total_records,
        "source_positions": source_positions,
        "profile_positions": profile_positions,
        "games_per_shard": campaign.get("games_per_shard"),
        "seed": campaign.get("seed"),
        "bank_fingerprint": campaign.get("bank_fingerprint"),
        "bank_sha256": bank_info["sha256"],
        "opening_bank": bank_info,
        "profiles": campaign.get("profiles"),
        "sources": campaign.get("sources"),
        "source_plan": campaign.get("source_plan"),
        "accepted_shards": accepted,
    }

    # Never overwrite a prior snapshot unless it already describes this exact
    # frozen campaign input; a new snapshot should get a new path.
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if (existing.get("source_campaign_manifest", {}).get("sha256") != manifest_sha
                or existing.get("accepted_shards") != accepted):
            raise FileExistsError(f"different frozen snapshot already exists: {out_path}")
        return out_path
    _atomic_json(out_path, payload)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=CONFIG["source"])
    parser.add_argument("--out-manifest", default=CONFIG["out_manifest"])
    args = parser.parse_args(argv)
    result = freeze_campaign_snapshot(args.source, args.out_manifest)
    print(json.dumps({"manifest": str(result), "complete": True,
                      "campaign_complete": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
