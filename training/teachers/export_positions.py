#!/usr/bin/env python3
"""Export de-duplicated architecture-neutral positions from teacher/DAgger shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from select_samples import iter_records
from targets import dumps, make_position, sample_id


def export_positions(corpus: Path, out: Path) -> dict:
    """Export unique histories while preserving split and provenance."""
    seen: set[str] = set()
    count = 0
    duplicate_count = 0
    sources: dict[str, int] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in iter_records(corpus):
            history = record.get("history")
            if not isinstance(history, list) or not all(isinstance(x, str) for x in history):
                raise ValueError("source record lacks a valid move history")
            sid = str(record.get("id") or sample_id(history))
            if sid in seen:
                duplicate_count += 1
                continue
            seen.add(sid)
            source = str(record.get("trajectory_source") or record.get("teacher") or "unknown")
            position = make_position(
                history,
                split=str(record.get("split", "train")),
                source=source,
                opening_index=int(record.get("opening_index", -1)),
                ply=int(record.get("ply", len(history))),
                metadata={
                    "origin_schema": record.get("schema"),
                    "teacher_bestmove": record.get("bestmove"),
                    "played_move": record.get("played_move"),
                    "student": record.get("student"),
                    "student_bestmove": record.get("diagnostics", {}).get("student_bestmove"),
                    "student_agrees": record.get("diagnostics", {}).get("student_agrees"),
                },
            )
            if position["id"] != sid:
                raise ValueError("source id does not match normalized move history")
            fh.write(dumps(position) + "\n")
            count += 1
            sources[source] = sources.get(source, 0) + 1
    manifest = {
        "schema": "zquoridor.position_manifest.v1",
        "source": str(corpus),
        "out": str(out),
        "positions": count,
        "duplicates_removed": duplicate_count,
        "sources": sources,
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = export_positions(args.corpus, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
