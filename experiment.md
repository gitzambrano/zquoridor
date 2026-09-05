# Experiment: Strong Self-Play / Wandering Baseline

## Status

**Search improvement confirmed and retained as the current experimental baseline. Training-policy work remains under evaluation in sibling experiment branches.**

## Purpose

This branch is the control line for two related problems:

1. eliminate late-game wandering without the large Elo regressions seen with broad exact-leaf or hard-minimax fallbacks;
2. establish a stronger search/self-play baseline before comparing new NNUE architectures.

## Key changes retained

- exact zero-wall endgame leaf fallback only: mover walls threshold 0, endgame leaf depth 2;
- root MCTS visit-distribution policy targets instead of one-hot selected-action imitation;
- search from ply 0 for new self-play cycles;
- wall-poor WL weighting support;
- stronger training/audit pipeline with explicit Elo promotion gate;
- `cPuct` reduced from 1.50 to **1.20** after search sweep.

## Confirmed search result

`cPuct = 1.20` was confirmed in a 2,500-game same-network search-only match against the previous 1.50 default:

- wins: 1,297
- losses: 958
- draws: 245
- Elo: **+47.4 ±13.0**
- lower bound: **+34.4 Elo**
- NPS was similar: about 27.3k vs 27.6k in that arena

This is a statistically convincing search gain and is now the baseline used by the bilateral architecture experiments.

## Wandering history

Earlier experiments showed:

- `MinimaxHard`: fixes behavior but catastrophically weak;
- broad combined-wall exact leaf: fixes wandering but loses substantial strength/performance;
- exact zero-wall fallback: removes the reported zero-wall loop with approximately neutral Elo;
- visit-policy targets improve the training direction and avoid the Gen9 one-hot self-imitation regression.

## Gen10 visit-policy diagnostic

On 5,000 games / 293,491 positions:

- 98.6% of positions had root visit-distribution targets;
- Gen10 vs Gen8: +9.8 ±26.0 Elo over 600 games;
- Gen10 vs Gen7: +8.1 ±26.3 Elo over 600 games.

This was promising but not statistically conclusive.

## Current role

This branch is the **search/control baseline** for architecture experiments. Do not mix speculative architecture changes directly into it.

## Decision rule

Changes entering this baseline must either:

- have a clearly positive lower 95% Elo bound in a sufficiently large arena; or
- be correctness/infrastructure fixes that do not alter engine behavior.

## Next action

Use this branch as the Gen8-era search baseline when evaluating bilateral and antisymmetric NNUE candidates. Keep architecture hypotheses in separate `exp/*` branches, each with its own `experiment.md`.
