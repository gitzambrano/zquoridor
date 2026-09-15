"""Prepare pinned external bots for local jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.external.bot_setup import ROOT, ensure_bot


CONFIG = {
    "bots": ["titanium", "claustrophobia"],
    "root": ROOT,
    "build": True,
    "dry_run": False,
}


def parse_args(argv: list[str] | None = None, config: dict | None = None) -> argparse.Namespace:
    """Parse command-line values over the CONFIG defaults."""
    defaults = CONFIG if config is None else config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bots",
        nargs="+",
        choices=("titanium", "claustrophobia"),
        default=list(defaults["bots"]),
        help="Select one or more bots.",
    )
    parser.add_argument(
        "--root",
        type=lambda value: Path(value).resolve(),
        default=Path(defaults["root"]).resolve(),
        help="Set the ZQuoridor project root.",
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["build"]),
        help="Build the bot executables.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["dry_run"]),
        help="Print the planned paths without file changes.",
    )
    return parser.parse_args(argv)


def _planned_paths(name: str, root: Path) -> dict[str, Path]:
    suffix = ".exe" if __import__("os").name == "nt" else ""
    if name == "titanium":
        base = root / "external_bots" / "titanium" / "repo" / "target" / "release"
        return {"executable": base / f"titanium{suffix}"}
    checkout = root / "external_bots" / "claustrophobia" / "repo"
    release = checkout / "target" / "release"
    return {
        "checkout": checkout,
        "checkpoint": root / "external_bots" / "claustrophobia" / "champion.pt",
        "encode_bridge": release / f"zq_encode_bridge{suffix}",
        "search_bridge": release / f"zq_search_bridge{suffix}",
        "benchmark_bridge": release / f"zq_benchmark_bridge{suffix}",
    }


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    results = {}
    for name in options.bots:
        paths = (
            _planned_paths(name, options.root)
            if options.dry_run
            else ensure_bot(name, root=options.root, build=options.build)
        )
        results[name] = {key: str(value.resolve()) for key, value in paths.items()}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
