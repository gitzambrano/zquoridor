# Experiment: Wall-Resource Monotonicity / Hard Replay

## Status

**Complete and rejected as a source of a new baseline. The monotonicity regularizer did not help, and the apparent lambda-zero gain disappeared in long confirmation.**

## Baseline

Network baseline: Gen8 (`gen8-search-ci-completed`, run `33973129847`). Search baseline: `fix/strong-selfplay-wandering` with production `cPuct=1.20` and the current endgame/search fixes.

## Hypothesis

The experiment tested whether enforcing a counterfactual wall-resource ordering with a hinge loss would improve the network's understanding of walls as conserved options. For a fixed board geometry, one extra own wall should not reduce optimal value and one extra opponent wall should not improve our value.

## Sweep protocol

Training data:

- 4,500 teacher games after holdout;
- 268,241 teacher positions;
- 481 wall-critical games;
- 64,362 focused replay positions after two copies;
- 332,603 total training positions;
- 30,400 held-out validation positions.

All candidates used the Gen8 architecture and warm-start weights, 24 epochs, LR 1e-5 -> 1e-6 cosine, `wl-gamma=0.9975`, and the same hard replay. Only `wall-mono-lambda` changed: 0, 0.05, 0.15.

Initial arena: 800 games, 20 ms/move, 4 threads, random plies 4, weights-only comparison against Gen8.

Workflow run: `34043824388`.

## Initial results

| Candidate | W-L-D vs Gen8 | Elo ±95% | NPS candidate / Gen8 | Own-wall violations | Opp-wall violations | Wandering |
|---|---:|---:|---:|---:|---:|---:|
| lambda 0 control | 386-337-77 | **+21.3 ±22.9** | 27,565 / 27,727 | 21.764% | 16.952% | 5/5 PASS |
| lambda 0.05 | 375-352-73 | +10.0 ±23.0 | 26,822 / 26,792 | 21.648% | 16.896% | 5/5 PASS |
| lambda 0.15 | 379-342-79 | +16.1 ±22.9 | 27,715 / 27,671 | 21.594% | 16.830% | 5/5 PASS |

The regularizer changed violation frequency by only about 0.17 percentage point for own walls and 0.12 point for opponent walls across the tested range. The regularized arms did not outperform the lambda-zero control.

Artifacts:

- control: `wall-monotonic-olddata-control`, ID `9992574021`;
- lambda 0.05: `wall-monotonic-olddata-l005`, ID `9992579697`;
- lambda 0.15: `wall-monotonic-olddata-l015`, ID `9992576930`.

## Long confirmation of lambda-zero control

Run `34046123235`, job `101521361511`.

Correctness:

- wandering race suite: **5/5 PASS**.

4,000-game arena vs Gen8, identical production code:

- wins: **1,832**;
- losses: **1,812**;
- draws: **356**;
- Elo: **+1.7 ±10.3**;
- candidate NPS: **27,167**;
- Gen8 NPS: **27,749**.

Artifact: `wall-monotonic-control-long-results`, ID `9993469713`.

## Interpretation

The +21.3 Elo 800-game point estimate was sampling noise or at least not reproducible at useful confidence. The 4,000-game result is centered essentially at zero and its interval comfortably includes no effect.

This also means the shared old-data hard-replay recipe is not sufficient evidence for a stronger network. The monotonicity regularizer itself remains unsupported because it barely moved the conceptual diagnostic and did not improve Elo.

## Decision

- **Reject lambda 0.05 and lambda 0.15.**
- **Do not promote lambda 0 control.**
- Keep Gen8 as the network baseline.
- Treat wall-count counterfactuals as diagnostics, not as the primary training constraint, because changing wall inventory while freezing board geometry creates off-manifold states.
- Move the next experiment to targeted legal reanalysis of real low-wall positions with a stronger search teacher.

## Next experiment

Gen13: `exp/wall-reanalysis-gen13`.

Gen13 reconstructs actual V3 states, reruns much deeper MCAB only at wall-timing decisions, freezes the Gen8 trunk and value head, and tests conservative policy-head interpolation doses before any long confirmation.
