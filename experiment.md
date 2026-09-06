# Experiment: Wall Economy Gen11

## Status

**In progress. Scratch training is decisively rejected; Gen8 warm-start has completed its arena and is finishing behavioral sampling.**

Primary workflow run: `34043150939`.

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

## Teacher data

Four parallel teacher jobs completed successfully:

- teacher-0 job `101513392361`;
- teacher-1 job `101513392393`;
- teacher-2 job `101513392332`;
- teacher-3 job `101513392414`.

Intended generation was 22,000 games: 20,000 broad games at 20 ms/move and 2,000 deeper games at 100 ms/move. The V3 audit detects **22,003 complete games / 1,380,324 positions** in the 24 shards. One complete broad shard (79,954 positions) is held out for validation.

The new teacher distribution is much cleaner than the older 5k teacher set. On the training population after the held-out shard is removed:

- 1,300,370 positions / 20,753 games;
- walls are 27.25% of moves;
- non-positive local wall efficiency is **4.60%**, versus about 13.5% on the old teacher data;
- mean wall efficiency is 1.76 BFS plies, median 2;
- 2,273 critical games are selected into the focused replay;
- focused replay contributes 312,048 positions after two copies.

The 100 ms subset also shows a strong resource-conservation pattern: the side that actually spends its final wall subsequently wins only about 1.65% of those events. This is observational rather than causal, but it is useful teacher signal and motivated upweighting deeper data.

## Training population

The training source totals **1,856,580 positions** after:

- broad and deep teacher shards (minus held-out broad shard);
- two additional copies of every 100 ms deep shard, making deep positions effectively 3x represented;
- two-copy wall-critical focused replay.

All sources use `k=1.0`; no old-network evaluation is blended into the outcome target. `wl-gamma=0.9975` discounts long-delayed outcomes.

## Candidates

Both candidates use the exact Gen8 architecture.

### Scratch — REJECTED

Job `101520018937`, artifact `wall-economy-gen11-scratch-results` ID `9993343558`.

Training:

- random initialization;
- 50 epochs;
- LR 5e-4 -> 2e-5 cosine;
- best exported checkpoint: epoch 48;
- held-out `val_outcome=0.2437`;
- held-out policy accuracy about 0.899.

Correctness/behavior gates:

- wandering race suite: **5/5 PASS**;
- counterfactual diagnostic became worse: own-wall violations 37.85%, opponent-wall violations 36.98% (Gen8 diagnostic on this sample: 20.40% / 17.90%).

Arena vs Gen8, 1,600 games, 20 ms/move, 4 threads, random plies 4:

- wins: **1**;
- losses: **1,367**;
- draws: **232**;
- Elo: **-441.2 ±22.3**;
- scratch NPS: **24,367**;
- Gen8 NPS: **28,342**.

Matched 500-game behavior sampling is catastrophic in exactly the failure family being targeted:

- scratch wall-move fraction: **46.68%** versus Gen8 **29.07%**;
- scratch non-positive wall efficiency: **60.24%** versus Gen8 **13.72%**;
- scratch mean wall efficiency: **0.66** versus Gen8 **1.63**;
- scratch reaches zero walls in 986 depletion events around ply 32, while Gen8 reaches zero walls only 89 times around ply 53 in the same-sized sample.

**Decision: scratch is decisively rejected.** The much larger, cleaner dataset is still not sufficient for the current training objective to relearn strong global strategy from random initialization. High offline policy accuracy is not evidence of playing strength.

### Warm start — ARENA COMPLETE, BEHAVIOR PENDING

Job `101520018956`.

- initialized from Gen8 float weights;
- new optimizer state;
- 40 epochs;
- LR 2e-5 -> 1e-6 cosine;
- same data and held-out shard as scratch.

At the latest check training, counterfactual benchmark, wandering gate and the 1,600-game arena are complete. Matched 500-game behavioral sampling is still running; final logs/artifact are therefore not yet available from GitHub.

## Interpretation so far

The scratch result is a key negative result. Scaling from roughly 5k to 22k games and upweighting cleaner/deeper data does **not** solve the learning problem by itself. The current supervised/distillation objective can obtain excellent validation metrics while learning a policy that places far too many locally useless walls and loses over 400 Elo.

Therefore the next training direction should not be “more of the same from random initialization.” The strongest alternatives are:

1. preserve the strong Gen8 representation via warm-start and make only controlled updates;
2. reanalyse a small number of real wall-critical states with a much more expensive legal teacher search, replacing shallow policy targets only where wall-vs-pawn resource decisions matter;
3. continue to gate every network by Elo and direct wall-economy behavior, not offline loss.

A targeted legal reanalysis prototype is being validated separately. It reconstructs real V3 positions and reruns deeper MCAB only on low-wall/ambiguous positions, avoiding the off-manifold assumption in the wall-count counterfactual probe.

## Decision rule

No candidate is promoted from offline metrics. A candidate must show convincing Elo evidence and no behavioral regression. Promising 1,600-game results require a longer, preferably independently seeded confirmation before changing production weights.

## Next action

1. Finish and extract the warm-start result.
2. Finish the 4,000-game confirmation of the older hard-replay lambda-0 control.
3. Validate targeted legal reanalysis and use it only on the strongest surviving network.
4. Promote nothing until a candidate clears the playing-strength gate and retains the behavioral gates.
