# Experiment: Wall-Resource Monotonicity / Hard Replay

## Status

**Monotonicity regularization is not supported by the first sweep. The no-regularizer control is promising and requires long confirmation.**

## Hypothesis

The Gen8 NNUE does not consistently understand walls as a conserved optional resource. For a fixed board position, giving the mover one additional available wall cannot reduce the optimal value because the player may simply choose not to use it. Likewise, giving the opponent an additional wall cannot improve our optimal value.

The experiment tests whether a counterfactual hinge loss enforcing these inequalities improves both this conceptual benchmark and playing strength.

## Baseline diagnosis

On the held-out/teacher family, wall-resource counterfactuals show frequent violations of the expected ordering. The wall-economy diagnostics also show a substantial population of low-value wall placements and poor outcomes after resource depletion. This motivated both hard-example replay and the monotonicity regularizer.

## Protocol

Source network: Gen8 (`gen8-search-ci-completed`, run `33973129847`).

Training data for this sweep:

- 4,500 teacher games for training after holding out one complete shard;
- 268,241 teacher positions;
- 481 wall-critical games mined into a focused replay;
- 64,362 focused replay positions after two copies;
- 332,603 total training positions;
- 30,400 held-out validation positions.

All candidates use the same Gen8 architecture and warm-start weights. Common training settings:

- 24 epochs;
- learning rate 1e-5 with cosine decay to 1e-6;
- `wl-gamma=0.9975`;
- monitor `val_outcome`;
- policy opening mask disabled for this controlled comparison;
- 25% of each training batch eligible for the counterfactual hinge term.

Only `wall-mono-lambda` changes: 0, 0.05, 0.15.

Arena protocol: 800 games, 20 ms/move, 4 threads, random opening plies 4, identical production search code, candidate NNUE vs Gen8 NNUE.

Workflow run: `34043824388`.

## Results

| Candidate | W-L-D vs Gen8 | Elo ±95% | NPS candidate / Gen8 | Own-wall violations | Opp-wall violations | Wandering |
|---|---:|---:|---:|---:|---:|---:|
| lambda 0 control | 386-337-77 | **+21.3 ±22.9** | 27,565 / 27,727 | 21.764% | 16.952% | 5/5 PASS |
| lambda 0.05 | 375-352-73 | +10.0 ±23.0 | 26,822 / 26,792 | 21.648% | 16.896% | 5/5 PASS |
| lambda 0.15 | 379-342-79 | +16.1 ±22.9 | 27,715 / 27,671 | 21.594% | 16.830% | 5/5 PASS |

Artifacts:

- control: `wall-monotonic-olddata-control`, ID `9992574021`;
- lambda 0.05: `wall-monotonic-olddata-l005`, ID `9992579697`;
- lambda 0.15: `wall-monotonic-olddata-l015`, ID `9992576930`.

## Interpretation

Increasing the monotonicity weight from 0 to 0.15 changes violation frequency by only about 0.17 percentage point for own walls and 0.12 percentage point for opponent walls. It also does not improve the Elo center relative to the lambda-0 control.

Therefore the regularizer itself is **not** the supported explanation for the positive playing-strength signal. The promising component is the retraining recipe shared by all three arms: Gen8 warm start plus the wall-critical hard replay and adjusted targets.

The lambda-0 control has the best Elo center (+21.3) but its lower 95% bound is still negative, so it is not promotable from this 800-game result.

## Decision

- Do not promote any monotonicity-regularized candidate.
- Deprioritize lambda 0.05/0.15.
- Long-confirm the lambda-0 control over 4,000 games.
- In parallel, compare against the larger 22,000-game Gen11 scratch/warm candidates when those finish.
- Promote a network only if it clears a sufficiently large Elo gate and retains the behavioral gates.

## Next action

Run a 4,000-game confirmation of the lambda-0 control against Gen8 with identical production code and weights-only variation. If the initial +21 Elo signal survives, compare it directly with the best Gen11 candidate before changing the production baseline.
