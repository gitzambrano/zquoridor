#!/usr/bin/env python3
"""Migrate every local self-play generation to one verified V3 contract.

Input may contain legacy 27-byte raw-perspective records, canonical 32-byte
V2 records, or canonical 64-byte V3 records.  Output is always V3 (64-byte)
with the mover-mirrored board, mover-relative policy index, explicit mover,
and a manifest beside each migrated generation.  Source files are never
modified.  Edit CONFIG or pass matching CLI options.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from read_selfplay import SAMPLE_DTYPE, SAMPLE_DTYPE_LEGACY, SAMPLE_DTYPE_V2, _detect_format

CONFIG = {
    "source": str(ROOT / "data" / "selfplay"),
    "out_dir": str(ROOT / "data" / "selfplay_canonical_v3"),
    "overwrite": False,
    "include_partial": False,
    "chunk_records": 262144,
}

SCHEMA = "zquoridor.selfplay.v3.canonical.mover-mirrored.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mirror_cell(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint16, copy=False)
    return ((8 - values // 9) * 9 + values % 9).astype(np.uint8)


def mirror_slots(bits: np.ndarray) -> np.ndarray:
    out = np.zeros(len(bits), dtype=np.uint64)
    for slot in range(64):
        selected = ((bits >> np.uint64(slot)) & np.uint64(1)) != 0
        if selected.any():
            target = (7 - slot // 8) * 8 + slot % 8
            out[selected] |= np.uint64(1) << np.uint64(target)
    return out


def mirror_policy(indices: np.ndarray) -> np.ndarray:
    out = indices.astype(np.uint16, copy=True)
    pawn = indices < 81
    if pawn.any():
        out[pawn] = mirror_cell(indices[pawn])
    horizontal = (indices >= 81) & (indices < 145)
    if horizontal.any():
        slots = indices[horizontal] - 81
        out[horizontal] = 81 + ((7 - slots // 8) * 8 + slots % 8)
    vertical = indices >= 145
    if vertical.any():
        slots = indices[vertical] - 145
        out[vertical] = 145 + ((7 - slots // 8) * 8 + slots % 8)
    return out


def legacy_movers(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer mover from complete legacy games; leading partial rows are dropped."""
    starts = ((rows["own_pawn"] == 4) & (rows["opp_pawn"] == 76)
              & (rows["walls_h"] == 0) & (rows["walls_v"] == 0)
              & (rows["walls_left_own"] == 10) & (rows["walls_left_opp"] == 10)
              & (rows["own_dist"] == 8) & (rows["opp_dist"] == 8))
    start_idx = np.flatnonzero(starts)
    if not len(start_idx):
        return np.zeros(0, np.uint8), np.zeros(0, bool)
    keep = np.arange(len(rows)) >= start_idx[0]
    position = np.arange(len(rows), dtype=np.int64)
    last = np.maximum.accumulate(np.where(starts, position, -1))
    movers = ((position - np.maximum(last, 0)) & 1).astype(np.uint8)
    return movers, keep


def valid_rows(rows: np.ndarray) -> np.ndarray:
    return ((rows["own_pawn"] < 81) & (rows["opp_pawn"] < 81)
            & (rows["own_pawn"] != rows["opp_pawn"])
            & (rows["walls_left_own"] >= 0) & (rows["walls_left_own"] <= 10)
            & (rows["walls_left_opp"] >= 0) & (rows["walls_left_opp"] <= 10)
            & (rows["policy_target"] <= 208))


def convert_legacy(rows: np.ndarray) -> tuple[np.ndarray, int]:
    movers, complete = legacy_movers(rows)
    keep = complete & valid_rows(rows)
    src = rows[keep]
    side = movers[keep]
    out = np.zeros(len(src), dtype=SAMPLE_DTYPE)
    flip = side == 1
    out["own_pawn"] = src["own_pawn"]
    out["opp_pawn"] = src["opp_pawn"]
    out["walls_h"] = src["walls_h"]
    out["walls_v"] = src["walls_v"]
    out["policy_target"] = src["policy_target"]
    if flip.any():
        out["own_pawn"][flip] = mirror_cell(src["own_pawn"][flip])
        out["opp_pawn"][flip] = mirror_cell(src["opp_pawn"][flip])
        out["walls_h"][flip] = mirror_slots(src["walls_h"][flip])
        out["walls_v"][flip] = mirror_slots(src["walls_v"][flip])
        out["policy_target"][flip] = mirror_policy(src["policy_target"][flip])
    for name in ("walls_left_own", "walls_left_opp", "game_result", "own_dist", "opp_dist"):
        out[name] = src[name]
    out["nnue_eval"] = 32768  # legacy field was heuristic, never a NNUE probability
    out["mover"] = side
    return out, int((~keep).sum())


def convert_file(source: Path, destination: Path) -> dict:
    dtype, size = _detect_format(str(source))
    n = source.stat().st_size // size
    raw = np.memmap(source, dtype=dtype, mode="r", shape=(n,))
    if size == 27:
        converted, dropped = convert_legacy(np.asarray(raw))
    else:
        mask = valid_rows(raw)
        converted = np.zeros(int(mask.sum()), dtype=SAMPLE_DTYPE)
        for name in dtype.names:
            converted[name] = raw[name][mask]
        dropped = int((~mask).sum())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with temp.open("wb") as output:
        converted.tofile(output)
    temp.replace(destination)
    return dict(source=str(source.resolve()), source_sha256=sha256(source), source_record_bytes=size,
                output=str(destination.resolve()), records_in=int(n), records_out=int(len(converted)), dropped=dropped,
                output_sha256=sha256(destination))


def migrate_generation(source: Path, destination: Path, config: dict) -> dict:
    files = sorted(source.glob("*.bin"))
    if not files:
        return {}
    if not config["include_partial"] and source.name == "gen12-standard-control-v2":
        files = [p for p in files if p.name != "selfplay_008.bin"]
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not config["overwrite"]:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if destination.exists() and config["overwrite"]:
        shutil.rmtree(destination)
    records = [convert_file(path, destination / path.name) for path in files]
    manifest = dict(schema=SCHEMA, generation=source.name, records=records,
                    records_in=sum(r["records_in"] for r in records),
                    records_out=sum(r["records_out"] for r in records),
                    dropped=sum(r["dropped"] for r in records),
                    output_record_bytes=64, legacy_nnue_eval="neutral; train with outcome-only k=1.0")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs = {"default": argparse.SUPPRESS}
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(value)
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    config = dict(CONFIG, **vars(parser.parse_args(argv)))
    source, output = Path(config["source"]), Path(config["out_dir"])
    generations = [p for p in sorted(source.iterdir()) if p.is_dir() and p.resolve() != output.resolve()]
    results = [migrate_generation(g, output / g.name, config) for g in generations]
    results = [r for r in results if r]
    root_manifest = dict(schema=SCHEMA, generations=[r["generation"] for r in results], records_out=sum(r["records_out"] for r in results), dropped=sum(r["dropped"] for r in results))
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(root_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
