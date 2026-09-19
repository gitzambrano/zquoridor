#!/usr/bin/env python3
"""Extract unique generic positions with move histories from benchmark games."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]


def sample_id(history: Sequence[str]) -> str:
    payload = " ".join(move.strip().lower() for move in history).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def extract_positions(
    root_dir: Path,
    out_file: Path,
    max_positions: int = 200000,
    seed: int = 20260919,
) -> dict:
    rng = random.Random(seed)
    openings_path = ROOT / "tools/external/openings_titanium.jsonl"
    blocked_openings = set()
    if openings_path.exists():
        with openings_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    o = json.loads(line)
                    blocked_openings.add(tuple(o.get("moves", [])))

    game_files = list(root_dir.glob("**/games.jsonl"))
    print(f"Found {len(game_files)} games.jsonl files under {root_dir}", flush=True)

    seen = set()
    positions = []

    for gf in game_files:
        try:
            with gf.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    g = json.loads(line)
                    moves = g.get("moves", [])
                    if len(moves) < 2:
                        continue
                    # Skip terminal ply, sample midgame plies
                    for ply in range(1, len(moves)):
                        h = tuple(moves[:ply])
                        if h in blocked_openings:
                            continue
                        sid = sample_id(h)
                        if sid in seen:
                            continue
                        seen.add(sid)
                        positions.append({
                            "id": sid,
                            "schema": "zquoridor.position.v1",
                            "side_to_move": ply % 2,
                            "history": list(h),
                            "ply": ply,
                            "source": str(gf.relative_to(ROOT)),
                        })
        except Exception:
            continue

    print(f"Extracted {len(positions):,} unique generic positions", flush=True)
    rng.shuffle(positions)
    selected = positions[:max_positions]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for p in selected:
            fh.write(json.dumps(p) + "\n")

    manifest = {
        "schema": "zquoridor.generic_positions_manifest.v1",
        "out": str(out_file),
        "total_extracted": len(positions),
        "selected": len(selected),
    }
    manifest_path = out_file.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(selected):,} positions to {out_file}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "data/teaching/generic-search-200k/positions.jsonl")
    parser.add_argument("--max-positions", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    extract_positions(args.root, args.out, args.max_positions, args.seed)


if __name__ == "__main__":
    main()
