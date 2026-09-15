#!/usr/bin/env python3
"""Collect and cache direct or searched teaching targets for NNUE students."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "teachers"))
import teaching_pipeline as pipeline

# Edit any setting here. Supplied CLI arguments override only those settings.
CONFIG = dict(pipeline.CONFIG)
CONFIG.update({
    "mode": "direct",
    "games": 24,
    "max_positions": 512,
    "gamma": 1.0,
    "outcome_weight": 0.0,
    "bootstrap_weight": 0.0,
    "policy_weights": {"old-direct": 1.0, "claustro-direct": 1.0},
    "value_weights": {"old-direct": 3.0, "claustro-direct": 1.0},
})


def main(argv=None):
    pipeline.CONFIG.update(CONFIG)
    return pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
