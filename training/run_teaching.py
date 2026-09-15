#!/usr/bin/env python3
"""Create a multi-teacher dataset for an NNUE student.

When ``--positions`` is supplied, consume an architecture-neutral JSONL file.
Without it, generate fresh ZQuoridor trajectories from the configured number
of games.  ``direct`` combines the old NNUE and direct Claustrophobia outputs;
``search`` combines ZQuoridor search and Claustrophobia MCTS; ``mixed`` uses
all four sources.  Temporal discount, outcome targets, own-search bootstrap,
temperatures, teacher weights, CPU/GPU device, and resumable caches are
configurable in the top-level ``CONFIG`` block or through CLI overrides.

Output is ``positions.jsonl`` (when generated), source caches under ``cache/``,
``teacher_targets.npz``, the final ``dataset.npz`` consumed by
``train_nnue.py``, and ``manifest.json``.  This script does not train or
promote a network.
"""
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
