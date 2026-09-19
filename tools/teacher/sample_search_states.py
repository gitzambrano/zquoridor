#!/usr/bin/env python3
"""Sample states from canonical V3 shards formatted as @state for MCTS and MCAB bridges.

Supports both:
- mode='crisis': filters for wall-depletion asymmetry (|ow - pw| >= 2 or <= 3 walls) and dense boards (>= 8 walls placed).
- mode='generic': uniform balanced sampling across game plies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from read_selfplay import _detect_format
from prepare_replay import legal_wall_topology, STATE_FIELDS


def sample_canonical_states(
    source_dir: Path,
    out_file: Path,
    target_count: int,
    mode: str = "crisis",
    seed: int = 20260919,
) -> dict:
    files = sorted(source_dir.rglob("*.bin"))
    valid = []
    for path in files:
        try:
            dtype, size = _detect_format(str(path))
            if size in (32, 64):
                valid.append((path, dtype, size))
        except Exception:
            continue

    if not valid:
        raise ValueError(f"no valid canonical shards in {source_dir}")

    rng = np.random.default_rng(seed)
    rng.shuffle(valid)

    # Block benchmark opening positions
    openings_path = ROOT / "tools/external/openings_titanium.jsonl"
    blocked = set()
    if openings_path.exists():
        from teachers.teaching_pipeline import _tool
        from build_teacher_soft import encode_states
        openings = [json.loads(line)["moves"] for line in openings_path.read_text().splitlines() if line.strip()]
        histories = {tuple(moves[:end]) for moves in openings for end in range(len(moves) + 1)}
        encoder = _tool("teacher_encode_state", "tools/teacher/encode_state.cpp")
        blocked_data = encode_states([{"history": list(h)} for h in histories], encoder)
        blocked = {tuple(int(blocked_data[k][i]) for k in STATE_FIELDS) for i in range(len(histories))}

    quota_per_shard = math.ceil(target_count / len(valid))
    seen = set(blocked)
    records = []

    print(f"Sampling {target_count:,} '{mode}' states from {len(valid)} shards...", flush=True)

    for file_idx, (path, dtype, size) in enumerate(valid):
        data = np.memmap(path, dtype=dtype, mode="r")
        n_data = len(data)
        if n_data == 0:
            continue

        perm = rng.permutation(n_data)
        accepted_this_shard = 0

        for idx in perm:
            row = data[idx]
            own_p = int(row["own_pawn"])
            opp_p = int(row["opp_pawn"])
            if own_p >= 72 or opp_p <= 8:
                continue

            ow = int(row["walls_left_own"])
            pw = int(row["walls_left_opp"])
            if not (0 <= ow <= 10 and 0 <= pw <= 10):
                continue

            wh = int(row["walls_h"])
            wv = int(row["walls_v"])
            key = (own_p, opp_p, wh, wv, ow, pw)
            if key in seen:
                continue

            if mode == "crisis":
                is_asymmetric = abs(ow - pw) >= 2 or ow <= 3 or pw <= 3
                is_dense = (20 - ow - pw) >= 8
                if not is_asymmetric and not is_dense:
                    continue
            elif mode == "generic":
                # Balanced sample: do not oversample asymmetry
                pass

            if not legal_wall_topology(wh, wv, own_p, opp_p):
                continue

            seen.add(key)
            # Format as @state string
            state_text = f"@state {own_p} {opp_p} {wh} {wv} {ow} {pw}"
            sid = hashlib.sha256(f"{own_p}_{opp_p}_{wh}_{wv}_{ow}_{pw}".encode("ascii")).hexdigest()[:24]

            records.append({
                "id": sid,
                "schema": "zquoridor.position.v1",
                "side_to_move": 0,  # Canonical V3 states are already from mover perspective (side 0)
                "history": [state_text],
                "mode": mode,
            })

            accepted_this_shard += 1
            if accepted_this_shard >= quota_per_shard or len(records) >= target_count:
                break

        del data
        if len(records) >= target_count:
            break

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    manifest = {
        "schema": "zquoridor.sampled_search_states.v1",
        "out": str(out_file),
        "target_count": target_count,
        "actual_count": len(records),
        "mode": mode,
    }
    manifest_path = out_file.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records):,} '{mode}' states to {out_file}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/selfplay_canonical_v3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--mode", choices=["crisis", "generic"], default="crisis")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    sample_canonical_states(args.source, args.out, args.count, args.mode, args.seed)


if __name__ == "__main__":
    main()
