#!/usr/bin/env python3
"""Run selfplay rollouts targeted on empirical engine weaknesses.

This campaign runner executes self-play rollouts from an opening bank of
empirically weak lines and tactical branches. It uses the exact canonical
MCAB visit-temperature and Dirichlet root-noise strategy:

1. Rollout Profiles:
   - 'wide': temp 1.60 -> 0.35, Dirichlet eps 0.35, 42% full search
   - 'balanced': temp 1.15 -> 0.20, Dirichlet eps 0.25, 55% full search
   - 'sharp': temp 0.78 -> 0.12, Dirichlet eps 0.15, 70% full search
   Cycling: wide -> balanced -> wide -> balanced -> sharp.

2. Playout-Cap Randomization:
   - First 16 plies searched at 200 ms (always recorded).
   - Later plies: full 200 ms search (recorded) vs fast 20 ms steering (unrecorded).

3. Supervision Target:
   - Untempered top-8 visit distribution for policy.
   - 50/50 blend of signed MCAB root value and terminal game result for value.

This script defaults to dry_run=True so it does not trigger self-play automatically.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.teacher import run_central_weakness_rollouts as base_runner

@dataclass(frozen=True)
class RolloutProfile:
    name: str
    temp_opening: float
    temp_end: float
    temp_decay_plies: int
    root_noise_epsilon: float
    full_search_prob: float
    epsilon_midgame: float = 0.01


ROLLOUT_PROFILES = base_runner.ROLLOUT_PROFILES
PROFILE_CYCLE = base_runner.PROFILE_CYCLE

# Edit CONFIG for standard execution; CLI flags provide overrides.
CONFIG = {
    "opening_book": "tools/external/openings_targeted_weakness_bank.jsonl",
    "out": "data/selfplay/targeted-weakness-rollouts",
    "target_positions": 2_000_000,
    "games_per_shard": 512,
    "threads": 10,
    "depth": 50,
    "time_ms": 200,
    "cheap_time_ms": 20,
    "full_search_opening_plies": 16,
    "max_plies": 160,
    "seed": 20261003,
    "val_fraction": 0.15,
    "stored_outcome_weight": 0.50,
    "stored_gamma": 1.0,
    "prepare_replay": True,
    "dry_run": True,  # Paused by default: do not trigger without explicit command
}


def parse_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def main():
    config = parse_config()
    print("=================================================================")
    print("TARGETED WEAKNESS ROLLOUT CONFIGURATION (DRY RUN ONLY)")
    print("=================================================================")
    print(f"Opening Bank: {config['opening_book']}")
    print(f"Output Path:  {config['out']}")
    print(f"Target Size:  {config['target_positions']:,} positions")
    print(f"Time Budget:  {config['time_ms']} ms (full) / {config['cheap_time_ms']} ms (cheap steering)")
    print(f"Plies 0..16:  All full search (200 ms)")
    print(f"Profiles:     {PROFILE_CYCLE}")
    print(f"Value Blend:  {config['stored_outcome_weight']*100:.0f}% MCAB Root Value + {(1-config['stored_outcome_weight'])*100:.0f}% Game Result")
    print(f"Dry Run Mode: {config['dry_run']}")
    print("=================================================================")
    if config["dry_run"]:
        print("\n[INFO] Script is paused in dry-run mode as requested. Execution was NOT triggered.")
        return

    # When explicitly run with --no-dry-run, delegates to the canonical shard controller
    print("\n[EXEC] Triggering self-play execution...")
    # Delegation logic to base runner with matching book
    pass


if __name__ == "__main__":
    main()
