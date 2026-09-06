# Experiment: Targeted Wall Reanalysis Gen13

## Status

**In progress. Reanalysis reconstruction smoke passed; full targeted teacher + policy-only dose-response experiment is being launched.**

Branch: `exp/wall-reanalysis-gen13`.

## Baseline

Production/control engine: `fix/strong-selfplay-wandering` with Gen8 NNUE from run `33973129847`, artifact `gen8-search-ci-completed`.

Gen8 remains the network baseline. The current search uses `cPuct=1.20`, the exact zero-wall endgame leaf fallback, and the exact zero-wall lazy policy-row optimization.

## Why this experiment exists

Gen11 isolated the defect more precisely:

- scratch retraining on 22k games collapsed to **-441.2 ±22.3 Elo**;
- Gen8 warm-start improved local wall quality dramatically but still lost **-55.8 ±16.1 Elo**;
- warm-start non-positive wall efficiency improved from 13.32% to 1.18%, yet it depleted walls much earlier.

Therefore the missing concept is not merely *where to place a good wall*. It is *when spending a wall is worth the future option value that is being surrendered*.

Gen12 also failed to provide a shortcut: its old-data lambda-zero control looked +21 Elo in 800 games but long-confirmed at **1832 W / 1812 L / 356 D = +1.7 ±10.3 Elo** over 4,000 games. The monotonic regularizer itself did not improve strength.

## Hypothesis

A stronger search can provide better targets specifically at real, legal low-resource states where wall-vs-pawn timing matters. Distilling those targets into only the direct policy head may improve search priors without destroying the proven Gen8 value representation.

## Reanalysis teacher

`tools/dev/wall_reanalysis.cpp`:

- reads complete V3 self-play games;
- reconstructs the exact legal `State` from mover-canonical records;
- validates pawn locations, wall accounting, BFS distances and non-terminal status;
- selects only real states with roughly 1-4 own walls and high resource-timing priority;
- favors ambiguous wall-vs-pawn roots, severe wall imbalance and wall-poor race states;
- reruns MCAB with Gen8 at substantially higher time/node budget;
- replaces the top-8 visit distribution only at selected positions;
- keeps complete games in output so dataset game boundaries remain valid;
- clears old policy visits at non-selected positions so shallow policy labels are not duplicated.

Smoke run `34046770249`, job `101523098110`:

- 1,250 input games;
- 2 selected critical positions;
- 2/2 successfully reanalysed;
- 0 failed positions;
- 0 reconstruction errors;
- 150 output records preserving complete selected games;
- `reanalysis smoke PASS`.

## Training constraint

`training/finetune_policy_reanalysis.py` starts from the Gen8 float network and freezes **all parameters except `policy.weight` and `policy.bias`**.

Hard invariant: every trunk and value tensor must remain bit-identical to the initial Gen8 float state. The script aborts if any non-policy tensor changes.

The fitted policy is not deployed wholesale. Gen13 exports conservative interpolation candidates:

- `alpha=0.10`: 10% of the fitted policy delta;
- `alpha=0.25`: 25% of the fitted policy delta;
- `alpha=0.50`: 50% of the fitted policy delta.

For each candidate:

`policy = Gen8 + alpha * (deep_reanalysis_fit - Gen8)`

This gives a dose-response curve and limits catastrophic policy drift.

## Data split

Reuse the completed Gen11 teacher campaign from run `34043150939` (22,003 audited games / 1,380,324 positions).

The four independently generated teacher partitions are reanalysed separately. Partitions 0, 1 and 2 are training data. Partition 3 is a complete held-out reanalysis validation partition, so no game can appear in both train and validation.

## Full-run protocol

Per teacher partition:

- restore Gen8 weights;
- restore the corresponding Gen11 teacher artifact;
- reanalyse the highest-priority low-wall positions at 250 ms/root with a 100k MCAB node cap;
- broad shards: up to 160 selected positions per shard;
- existing 100 ms deep shards: up to 80 selected positions per shard;
- require zero reconstruction errors and non-empty deep visit targets;
- upload the partition as a separate artifact.

Training:

- policy head only;
- 30 epochs;
- AdamW;
- LR 2e-4 -> 1e-5 cosine;
- held-out partition 3 selects the best epoch;
- export alpha 0.10 / 0.25 / 0.50 candidates.

Correctness gates for every candidate:

1. trunk/value freeze invariant during export;
2. 5/5 wandering race PASS;
3. identical search code and Gen8 value network;
4. only policy-head weights differ.

Strength screen:

- 800 games per alpha vs Gen8;
- 20 ms/move;
- 4 threads;
- random plies 4;
- policy ordering enabled, min-depth 3;
- same production search on both sides.

The best alpha is eligible for a longer confirmation only if it is not behaviorally regressive and its Elo center is positive enough to justify more games.

## Decision rule

No promotion from validation loss, policy accuracy, wall-efficiency diagnostics or an 800-game point estimate alone.

A candidate must preserve correctness/wandering, show a credible positive Elo signal, and then survive an independently seeded long confirmation before replacing Gen8.

If all three alpha candidates are neutral/negative, reject policy-only reanalysis as insufficient and move the deep supervision one level earlier in the representation rather than globally fine-tuning the network.
