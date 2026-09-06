# Experiment: Wall Economy Gen11

## Status

**Complete. Both scratch and Gen8 warm-start candidates are rejected. The experiment nevertheless isolates the main behavioral defect: wall timing/resource conservation, not merely local wall quality.**

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

Production includes `cPuct=1.20`, the exact zero-wall endgame leaf fallback, and the exact lazy zero-wall policy-row optimization. Production commit at experiment start includes `12ea0136e0c1d73ac63847b4d72d0171b4ff9473`.

Network baseline: Gen8 from run `33973129847`, artifact `gen8-search-ci-completed`.

## Teacher data

Four parallel teacher jobs completed successfully:

- teacher-0 job `101513392361`;
- teacher-1 `101513392393`;
- teacher-2 `101513392332`;
- teacher-3 `101513392414`.

The V3 audit found **22,003 complete games / 1,380,324 positions** in 24 shards, approximately 20k broad games at 20 ms/move plus 2k deeper games at 100 ms/move. One complete broad shard (79,954 positions) was held out for validation.

The new teacher distribution is much cleaner than the older 5k teacher set. On the training population after holdout:

- 1,300,370 positions / 20,753 games;
- wall moves: 27.25%;
- non-positive local wall efficiency: **4.60%**, versus about 13.5% on the old teacher data;
- mean wall efficiency: 1.76 BFS plies, median 2;
- 2,273 critical games selected into focused replay;
- 312,048 focused replay positions after two copies.

The 100 ms subset gives a strong resource-conservation signal: the side that actually spends its final wall subsequently wins only about 1.65% of those depletion events. This is observational, not causal, but motivated upweighting deeper data.

## Training population

Total training population: **1,856,580 positions** after adding two extra copies of each deep shard and a two-copy wall-critical replay. All sources use `k=1.0`. `wl-gamma=0.9975`.

## Scratch candidate — REJECTED

Job `101520018937`; artifact `wall-economy-gen11-scratch-results`, ID `9993343558`.

Training:

- random initialization;
- 50 epochs;
- LR 5e-4 -> 2e-5 cosine;
- best epoch 48;
- held-out `val_outcome≈0.2437`;
- held-out policy accuracy ≈0.899.

Wandering: **5/5 PASS**.

Arena vs Gen8, 1,600 games:

- **1 W / 1,367 L / 232 D**;
- **-441.2 ±22.3 Elo**;
- NPS: 24,367 vs 28,342.

Matched 500-game behavior:

- wall-move fraction: **46.68%** vs Gen8 ~29.07%;
- non-positive wall efficiency: **60.24%** vs Gen8 ~13.72%;
- mean wall efficiency: 0.66 vs 1.63;
- zero-wall depletion events: 986 around ply 32 vs Gen8 89 around ply 53.

Scratch is decisively rejected. Excellent offline policy accuracy did not translate to strategy or strength.

## Warm-start candidate — REJECTED

Job **`101520018956`**; artifact `wall-economy-gen11-warm-results`, ID **`9993396139`**.

Training:

- initialized from Gen8 float weights;
- new optimizer state;
- 40 epochs;
- LR 2e-5 -> 1e-6 cosine;
- best exported epoch 39;
- held-out `val_outcome=0.2954`;
- final held-out policy accuracy ≈0.901.

Counterfactual diagnostic did not materially improve:

- own-wall violations: **21.448%**;
- opponent-wall violations: **17.308%**.

Wandering: **5/5 PASS**.

Arena vs Gen8, 1,600 games, 20 ms/move, 4 threads, random plies 4:

- candidate wins: **573**;
- Gen8 wins: **828**;
- draws: **199**;
- **-55.8 ±16.1 Elo**;
- candidate NPS: **26,556**;
- Gen8 NPS: **27,411**.

The warm-start network is statistically weaker and is not promotable.

### Matched 500-game behavior: the key diagnosis

Warm-start:

- wall-move fraction: **28.75%**;
- non-positive wall efficiency: **1.18%**;
- mean wall efficiency: **1.66**;
- wall-poor race positions: **1**;
- reaches zero walls: **527 events**, mean ply 54.72;
- reaches one wall: **586 events**, mean ply 31.18.

Gen8 under the same behavior protocol/seed:

- wall-move fraction: **29.09%**;
- non-positive wall efficiency: **13.32%**;
- mean wall efficiency: **1.63**;
- wall-poor race positions: **122**;
- reaches zero walls: **72 events**, mean ply 51.72;
- reaches one wall: **504 events**, mean ply 47.66.

The warm-start candidate therefore learned to place **far better individual walls** without increasing the overall wall-move fraction, yet it reaches the last one or two walls far earlier and reaches zero walls in far more games. It loses 55.8 Elo despite eliminating most locally useless walls.

This isolates the strategic defect more precisely:

> **The important missing concept is not local wall efficiency. It is the long-horizon opportunity cost and timing of spending a finite wall inventory.**

A wall can be locally excellent and still be strategically premature because keeping the option has future value.

## Decision

- Reject scratch.
- Reject warm-start.
- Do not use local wall-efficiency improvement as a promotion criterion by itself.
- Do not continue uniform scratch training merely by adding more games/epochs.
- Preserve Gen8 as the network baseline until another candidate wins an arena gate.

## Next action

Target **real legal low-wall decisions** with expensive search reanalysis rather than broad self-imitation. Reanalyse states where the mover has roughly 1-4 walls remaining and the root must choose between spending a wall now or retaining the resource. Distill those deeper root visit distributions into Gen8 with a small controlled update, then gate by Elo, depletion timing, wall efficiency, wall-poor races and wandering.
