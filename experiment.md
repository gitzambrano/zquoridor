# Experiment: Wall Economy Gen11

## Status

**In progress. Teacher generation is complete; scratch and Gen8 warm-start candidates are training on the larger targeted dataset. No candidate is promotable yet.**

Workflow run: `34043150939`.

## Hypothesis

The current Gen8 network has learned local wall patterns but not a sufficiently robust long-horizon representation of walls as a scarce strategic resource. The observed failure family is:

1. spend walls aggressively or inefficiently;
2. enter a severe wall-count imbalance;
3. reach a pawn race with little or no defensive resource;
4. rely on the search-side zero-wall fallback to avoid terminal wandering.

The experiment tests whether substantially more search-generated data, deeper teacher examples, and hard-example replay improve this behavior without changing the Gen8 architecture.

## Baseline

Production/control branch: `fix/strong-selfplay-wandering`.

Current production search includes `cPuct=1.20`, the exact zero-wall endgame leaf fallback, and the exact lazy zero-wall policy-row optimization. Production commit at experiment start includes `12ea0136e0c1d73ac63847b4d72d0171b4ff9473`.

Network baseline: Gen8 from run `33973129847`, artifact `gen8-search-ci-completed`.

## Wall-economy diagnostics

The diagnostic tooling measures:

- fraction of moves that place walls;
- immediate wall efficiency `delta(opponent distance) - delta(own distance)`;
- zero/negative-gain wall placements;
- wall depletion milestones and the game outcome after reaching 5/3/1/0 walls;
- same-player wall-placement chains;
- wall-poor race positions;
- root visit mass assigned to walls;
- matched candidate-vs-Gen8 behavior samples.

A counterfactual benchmark also fixes the board geometry and varies only `wallsLeft`. It measures whether the value head respects the resource-ordering expectation that an extra own wall should not reduce optimal value and an extra opponent wall should not improve our value.

## Teacher data

Four parallel teacher jobs completed successfully:

- teacher-0 job `101513392361`;
- teacher-1 job `101513392393`;
- teacher-2 job `101513392332`;
- teacher-3 job `101513392414`.

Each partition generated:

- 5,000 broad search-driven games at 20 ms/move;
- 500 deeper games at 100 ms/move.

Combined total: **22,000 games**:

- 20,000 broad games;
- 2,000 deeper games.

One complete broad shard is held out for validation. The 100 ms teacher shards are duplicated in the training source to increase their sampling weight without changing labels.

## Hard-example replay

`training/wall_economy.py` mines complete games containing strategically critical wall states, including low-resource positions, large wall imbalance, wall-poor races, depletion events, and ambiguous wall-vs-pawn decisions. Selected games are replayed twice with a maximum focused fraction of 12% of the source positions.

The older 5k dataset already showed the failure family clearly: roughly 28% of moves were walls and about 13.5% of wall transitions had non-positive local efficiency. Outcomes also degraded sharply after reaching one or zero walls. These numbers motivated the larger targeted campaign but are not by themselves causal estimates of wall value.

## Candidates

Two candidates use the exact same Gen8 architecture.

### Scratch

Job: `101520018937`.

- random initialization;
- 50 epochs;
- LR 5e-4 -> 2e-5 cosine;
- monitor `val_outcome`;
- `wl-gamma=0.9975`.

### Warm start

Job: `101520018956`.

- initialized from Gen8 float weights;
- new optimizer state;
- 40 epochs;
- LR 2e-5 -> 1e-6 cosine;
- monitor `val_outcome`;
- `wl-gamma=0.9975`.

Both train on the same broad + deeper-replay + wall-critical replay population and the same held-out validation shard.

## Gates

After training, each candidate must pass:

1. counterfactual wall-resource benchmark;
2. 5-case wandering race suite;
3. 1,600-game arena against Gen8 at 20 ms/move, 4 threads, random plies 4;
4. matched 500-game behavior sampling for candidate and Gen8 with identical seeds/configuration.

No promotion is based on offline loss alone.

## Decision rule

A candidate is considered for promotion only if it shows convincing playing-strength evidence and no behavioral regression. If the 1,600-game result is promising but statistically inconclusive, run a longer confirmation before changing production weights.

Scratch versus warm start is itself part of the experiment: if scratch fails while warm start succeeds, the dataset improves the existing representation but is not sufficient to relearn the whole network from random initialization. If scratch succeeds, the larger data regime is sufficient to learn the Gen8 architecture independently.

## Current state

Teacher generation, validation split, wall-economy mining, and deeper-data replay preparation have all completed. At the latest check both `train-scratch` and `train-warm` are in the training step. All downstream gates are pending.

## Next action

Let both training arms complete, compare their counterfactual/behavioral metrics and Elo, then long-confirm the strongest candidate if warranted. Also compare it against the best old-data hard-replay control from `exp/wall-monotonic-gen12` before any production promotion.
