#!/usr/bin/env python3
"""
Audit local self-play shards before a NNUE training cycle.

The current 32-byte format stores the board and policy target in the canonical
mover perspective. The legacy 27-byte format does not. train_nnue.py cannot
repair that perspective after the legacy record is loaded because the old
record does not store the mover identity.

Use this tool before a training cycle. A legacy shard must not enter a modern
NNUE training population.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
DATA_ROOT = ROOT / "data" / "selfplay"

sys.path.insert(0, str(TRAINING_DIR))
from read_selfplay import (  # noqa: E402
    SAMPLE_DTYPE,
    SAMPLE_DTYPE_LEGACY,
    _is_legacy_file,
    game_start_mask,
)


def generation_name(path: Path) -> str:
    try:
        rel = path.relative_to(DATA_ROOT)
    except ValueError:
        return str(path.parent)
    return rel.parts[0] if len(rel.parts) > 1 else rel.parent.name


def modern_stats(path: Path) -> dict:
    arr = np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")
    if len(arr) == 0:
        return dict(samples=0, games=0, risk=0, no_walls=0)

    own_walls = arr["walls_left_own"].astype(np.int16)
    opp_walls = arr["walls_left_opp"].astype(np.int16)
    own_dist = arr["own_dist"].astype(np.int16)
    opp_dist = arr["opp_dist"].astype(np.int16)

    no_walls = (own_walls == 0) & (opp_walls >= 4)
    asymmetric = (
        (own_walls <= 2)
        & (opp_walls - own_walls >= 4)
        & (own_dist + 2 <= opp_dist)
    )
    return dict(
        samples=len(arr),
        games=int(game_start_mask(arr).sum()),
        risk=int((no_walls | asymmetric).sum()),
        no_walls=int(no_walls.sum()),
    )


def audit(paths: list[Path]) -> tuple[dict, list[Path]]:
    rows = defaultdict(
        lambda: dict(
            files=0,
            modern=0,
            legacy=0,
            samples=0,
            games=0,
            risk=0,
            no_walls=0,
        )
    )
    legacy_paths: list[Path] = []

    for path in paths:
        gen = generation_name(path)
        row = rows[gen]
        row["files"] += 1

        if _is_legacy_file(str(path)):
            row["legacy"] += 1
            legacy_paths.append(path)
            size = path.stat().st_size
            row["samples"] += size // SAMPLE_DTYPE_LEGACY.itemsize
            continue

        row["modern"] += 1
        st = modern_stats(path)
        for key in ("samples", "games", "risk", "no_walls"):
            row[key] += st[key]

    return dict(sorted(rows.items())), legacy_paths


def print_report(rows: dict, legacy_paths: list[Path]) -> None:
    print(
        f"{'generation':28s} {'files':>6s} {'modern':>7s} {'legacy':>7s} "
        f"{'samples':>12s} {'games':>8s} {'risk-pos':>10s}"
    )
    print("-" * 86)
    for gen, row in rows.items():
        print(
            f"{gen:28.28s} {row['files']:6d} {row['modern']:7d} {row['legacy']:7d} "
            f"{row['samples']:12,d} {row['games']:8,d} {row['risk']:10,d}"
        )

    total_files = sum(r["files"] for r in rows.values())
    total_legacy = sum(r["legacy"] for r in rows.values())
    total_samples = sum(r["samples"] for r in rows.values())
    print("-" * 86)
    print(
        f"total: {total_files:,} file(s), {total_samples:,} sample(s), "
        f"{total_legacy:,} legacy file(s)"
    )

    if legacy_paths:
        print("\nLEGACY SHARDS. DO NOT MIX THESE SHARDS INTO MODERN NNUE TRAINING:")
        for path in legacy_paths:
            print(f"  - {path}")
        print(
            "\nThe loader can upcast the record size, but it cannot reconstruct the "
            "canonical mover perspective of the old board and policy fields."
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="*",
        help="files or directories to audit. Default: data/selfplay",
    )
    p.add_argument(
        "--fail-on-legacy",
        action="store_true",
        help="return exit code 2 if at least one legacy shard is found",
    )
    return p.parse_args()


def expand_inputs(items: list[str]) -> list[Path]:
    roots = [Path(x) for x in items] if items else [DATA_ROOT]
    out: list[Path] = []
    for root in roots:
        root = root if root.is_absolute() else (ROOT / root)
        if root.is_file() and root.suffix == ".bin":
            out.append(root)
        elif root.is_dir():
            out.extend(sorted(root.rglob("*.bin")))
        else:
            raise SystemExit(f"path does not exist: {root}")
    return sorted(set(out))


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.paths)
    if not paths:
        raise SystemExit("no .bin self-play files were found")

    rows, legacy_paths = audit(paths)
    print_report(rows, legacy_paths)
    if args.fail_on_legacy and legacy_paths:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
