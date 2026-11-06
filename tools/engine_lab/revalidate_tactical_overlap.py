#!/usr/bin/env python3
"""Revalidate seeded tactical sample histories against every frozen evaluation book.

This script uses the same deterministic sample IDs as the search audit.
It does not repeat inference and does not assign training targets.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.engine_lab.tactical_consensus_audit import (
    CONFIG, detect_benchmark_overlap, load_book,
)

SAMPLES = {"normal": 14, "center": 14, "targeted": 14, "irregular": 20}
SEED_OFFSETS = {"normal": 11, "center": 13, "targeted": 17, "irregular": 19}
SEED = 20261106


def main() -> None:
    frozen_by_book = {
        path: [tuple(record["moves"]) for record in load_book(ROOT / path)]
        for path in CONFIG["frozen_evaluation_paths"]
    }
    all_frozen = [opening for book in frozen_by_book.values() for opening in book]

    # Regression guard: all evaluation openings must identify themselves.
    for name, openings in frozen_by_book.items():
        for opening in openings:
            assert detect_benchmark_overlap(list(opening), all_frozen), name

    results = {}
    for category, count in SAMPLES.items():
        book = load_book(ROOT / CONFIG["book_paths"][category])
        indexes = list(range(len(book)))
        random.Random(SEED + SEED_OFFSETS[category]).shuffle(indexes)
        samples = []
        for i in indexes[:count]:
            history = book[i]["moves"]
            overlapped = [
                Path(name).name for name, openings in frozen_by_book.items()
                if detect_benchmark_overlap(history, openings)
            ]
            samples.append({
                "source_index": i,
                "benchmark_overlap": bool(overlapped),
                "overlapping_books": overlapped,
            })
        results[category] = {
            "count": count,
            "overlap_count": sum(row["benchmark_overlap"] for row in samples),
            "nonoverlap_count": sum(not row["benchmark_overlap"] for row in samples),
            "samples": samples,
        }
        print("SUMMARY " + json.dumps({
            "source": category,
            "samples": count,
            "overlap": results[category]["overlap_count"],
            "nonoverlap": results[category]["nonoverlap_count"],
        }), flush=True)
    out = ROOT / "results/tactical-consensus/eligibility-recheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "evaluation_books": list(frozen_by_book),
        "results": results,
        "diagnostic_only": True,
        "description": "Opening-history prefix overlap, not a training-quality label.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
