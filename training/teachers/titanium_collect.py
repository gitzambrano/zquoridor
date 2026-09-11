#!/usr/bin/env python3
"""Collect architecture-neutral teacher data from pinned Titanium."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Keep the Titanium CLI and call the generic teacher collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titanium", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--movetime", type=int, default=200)
    parser.add_argument("--budgets")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--val-mod", type=int, default=5)
    parser.add_argument("--student")
    parser.add_argument("--student-engine")
    parser.add_argument("--student-engine-arg", action="append", default=[])
    parser.add_argument("--student-movetime", type=int, default=200)
    args = parser.parse_args(argv)

    budgets = args.budgets if args.budgets is not None else str(args.movetime)
    forwarded = [
        "--teacher",
        "Titanium",
        "--protocol",
        "titanium",
        "--engine",
        args.titanium,
        "--openings",
        args.openings,
        "--out-dir",
        args.out_dir,
        "--budgets",
        budgets,
        "--repeats",
        str(args.repeats),
        "--workers",
        str(args.workers),
        "--max-plies",
        str(args.max_plies),
        "--val-mod",
        str(args.val_mod),
    ]
    if args.student is not None or args.student_engine is not None:
        if args.student is None or args.student_engine is None:
            raise SystemExit("set both --student and --student-engine, or set neither")
        forwarded.extend(
            [
                "--student",
                args.student,
                "--student-engine",
                args.student_engine,
                "--student-movetime",
                str(args.student_movetime),
            ]
        )
        for engine_arg in args.student_engine_arg:
            forwarded.append(f"--student-engine-arg={engine_arg}")
    return collect.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
