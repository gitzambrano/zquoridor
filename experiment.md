# Experiment: Wall Economy Gen11

## Status

**Complete. Both scratch and Gen8 warm-start candidates are rejected. The experiment isolates the main behavioral defect: wall timing/resource conservation, not merely local wall quality.**

Primary workflow run: `34043150939`.

## Baseline

Production/control branch: `fix/strong-selfplay-wandering`, including commit `12ea0136e0c1d73ac63847b4d72d0171b4ff9473`. Network baseline: Gen8 from run `33973129847`.

## Teacher data

The V3 audit found **22,003 games / 1,380,324 positions**. After holdout, teacher wall moves were 27.25% of positions and only **4.60%** had non-positive local BFS efficiency, versus about 13.5% in the old teacher. A 2-copy critical replay and 3x effective weighting of the 100ms subset produced **1,856,580 training positions**.

## Scratch — REJECTED

Job `101520018937`; artifact ID `9993343558`.

- 1 W / 1,367 L / 232 D vs Gen8;
- **-441.2 ±22.3 Elo**;
- NPS 24,367 vs 28,342;
- wandering 5/5 PASS;
- wall-move fraction 46.68%;
- non-positive wall efficiency 60.24%.

Offline policy accuracy near 0.899 did not predict playing strength.

## Warm start — REJECTED

Job `101520018956`; artifact ID `9993396139`.

- 573 W / 828 L / 199 D vs Gen8;
- **-55.8 ±16.1 Elo**;
- NPS 26,556 vs 27,411;
- wandering 5/5 PASS;
- counterfactual own/opp wall violations 21.448% / 17.308%.

Matched 500-game behavior is the key result:

| Metric | Warm | Gen8 |
|---|---:|---:|
| Wall move fraction | 28.75% | 29.09% |
| Non-positive wall efficiency | **1.18%** | 13.32% |
| Mean wall efficiency | 1.66 | 1.63 |
| Wall-poor race positions | **1** | 122 |
| Reaches 1 wall | 586 events, mean ply **31.18** | 504 events, mean ply **47.66** |
| Reaches 0 walls | **527** events, mean ply 54.72 | **72** events, mean ply 51.72 |

The warm network learned much better individual wall placements but depleted its inventory much earlier and lost 55.8 Elo. Therefore **local wall quality and strategic wall conservation are distinct learning problems**.

## Conclusion

The missing concept is long-horizon opportunity cost: a locally strong wall may still be premature because preserving the option has future value. Uniformly adding more self-play or epochs is not enough, and full-network fine-tuning is too destructive.

## Next experiment

Gen13 uses **targeted legal reanalysis** of real low-wall positions. A much more expensive MCAB search replaces shallow visit targets only where the mover has roughly 1–4 walls and the root faces a wall-vs-pawn/resource-timing decision.

The first Gen13 candidate will start from Gen8 and train **only the direct policy head**. The 354->256 trunk and full value head remain bit-identical to Gen8. This intentionally minimizes regression risk while testing whether deeper policy supervision can improve when the engine spends its last few walls.
