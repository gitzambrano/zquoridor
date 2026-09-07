# Structural strength experiments

This document ranks architecture changes that can produce a larger strength gain than local search tuning. Each experiment must isolate one structural variable and pass both the Elo gate and the wandering regression before promotion.

## 1. Full bilateral accumulator

Current inference maintains an `AccPair`, but the value and policy heads consume only the accumulator for the side to move. The highest-priority architecture experiment is to concatenate both activated accumulators before the heads:

`[SCReLU(acc[mover]), SCReLU(acc[opponent])] -> 512 features`

Candidate heads:

- value: `512 -> 64 -> 1`
- policy: `512 -> 209`

The sparse incremental layer remains unchanged. Both accumulators are already updated during search, so the main added cost is in the heads rather than feature maintenance.

Hypothesis: bilateral context makes race and wall asymmetry easier to represent, especially states where one player has no walls but a shorter path. This is directly relevant to the wandering failure mode.

Gate: train from fresh weights on modern V3 visit-policy data, then compare against the strongest current network with identical search. Require `Elo - 95% margin > 0` and a passing wandering regression.

## 2. Search-value teacher instead of old-network bootstrap

The current historical training recipe can blend the final game result with the static NNUE evaluation recorded during self-play. That reuses the old network's value bias as a target. A stronger structural target is the root MCAB value after search.

Add a self-play field for the searched root value, normalized to a probability. Train the value head with:

`target = alpha * final_result + (1 - alpha) * searched_root_value`

Do not use the pre-search static NNUE value as a teacher in new generations.

Hypothesis: the actor learns from the search improvement operator rather than copying its previous value estimate. This is the value-side analogue of the V3 root-visit policy target.

Initial experiment: `alpha` in `{0.7, 0.85, 1.0}`. Use whole-game result only as the control.

## 3. Strong-teacher self-play

Separate the data-generation search budget from the production search budget. Generate training targets with a stronger teacher, for example 80-150 ms per move or a higher simulation cap, while evaluating the trained network at the production budget.

Hypothesis: self-play at the same weak budget can converge to self-consistent but non-improving targets. A deeper teacher creates policy and value targets that contain information unavailable to the actor at generation time.

The first test should keep the network architecture fixed and compare data generated at 20 ms against data generated at 80 ms. This isolates teacher quality from model capacity.

## 4. Relational race features

The current sparse input exposes own and opponent distances and wall counts independently. Add a small set of explicit relational buckets:

- `ownDist - oppDist`
- `ownWalls - oppWalls`
- race class: ahead / equal / behind
- wall-poor class: own walls in `{0,1,2}` with opponent wall advantage

These features are cheap to update and directly encode interactions that currently require the first hidden layer to construct from independent one-hot inputs.

This experiment is lower priority than the full accumulator because the bilateral architecture may make most of these features redundant.

## 5. Capacity increase only after target quality is fixed

The current sparse trunk is `354 -> 256`. Test `354 -> 384` and `354 -> 512` only after visit-policy targets and search-value targets are available. Increasing capacity while training on self-referential targets risks fitting the same bias more accurately, as seen in earlier generations where validation improved without Elo improvement.

A capacity candidate must report both NPS and Elo. Reject a larger network if its strength gain disappears after accounting for the production time budget.

## Experimental order

1. Finish the current `cPuct` confirmation and Gen11 gates.
2. Implement full bilateral accumulator as the first architecture candidate.
3. Add searched-root value to the V3 self-play record and test value-target blends.
4. Test stronger-teacher self-play.
5. Add relational features only if wall-poor behavior still fails.
6. Increase hidden width last.

No architecture candidate becomes production because of validation loss alone. Promotion requires a positive lower 95% Elo bound at the production search configuration and a passing wandering regression.
