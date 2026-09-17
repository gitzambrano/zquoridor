#!/usr/bin/env python3
"""Mix a direct replay dataset and a search-teaching dataset by sample weight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.run_campaign import combine_datasets

# Edit this block for normal local use. Command-line options override these values.
CONFIG = {
    "replay": str(ROOT / "data" / "teaching" / "replay-historical-2m-cuda" / "dataset.npz"),
    "teaching": str(ROOT / "data" / "teaching" / "search-priority-gen1-conservative" / "dataset.npz"),
    "out": str(ROOT / "data" / "teaching" / "historical2m-search10-conservative" / "dataset.npz"),
    "teaching_fraction": 0.10,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay")
    parser.add_argument("--teaching")
    parser.add_argument("--out")
    parser.add_argument("--teaching-fraction", type=float)
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    config.update({key: value for key, value in vars(args).items() if value is not None})
    for key in ("replay", "teaching"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} does not exist: {config[key]}")
    if not 0 < float(config["teaching_fraction"]) < 1:
        raise ValueError("teaching_fraction must be in (0, 1)")
    return config


def run(config: dict) -> dict:
    output = Path(config["out"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = combine_datasets(
        Path(config["replay"]).resolve(), Path(config["teaching"]).resolve(), output,
        float(config["teaching_fraction"]),
    )
    manifest = {"schema": "zquoridor.teacher.mix.v1", "replay": str(Path(config["replay"]).resolve()),
                "teaching": str(Path(config["teaching"]).resolve()),
                "teaching_fraction": float(config["teaching_fraction"]), **report}
    (output.parent / "dataset.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    try:
        run(resolve_config(build_parser().parse_args(argv)))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"dataset mix error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
