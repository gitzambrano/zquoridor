# Experiment: Antisymmetric Bilateral Value v3

## Status

**Training complete. Architecture invariants and wandering suite passed. Elo arena pending rerun because the first arena failed before game 1 due to a missing local Git ref in CI.**

## Hypothesis

Quoridor is zero-sum. Swapping own/opponent perspectives should negate the value logit. The unconstrained bilateral v2 must learn this property from data. Enforcing it by construction should improve sample efficiency and calibration consistency, especially in asymmetric wall-poor race states.

## Change

The shared transformer remains 354 -> 256 and the policy head remains unilateral.

The WL head is changed from unconstrained bilateral concat to an antisymmetric construction:

`V(A,B) = F(A,B) - F(B,A)`

with one shared `F`. Therefore `V(A,B) = -V(B,A)` exactly.

To keep first-stage value compute close to v2, `F` uses a 32-unit hidden layer instead of v2's 64-unit single bilateral layer.

## Controlled comparison

This branch deliberately reuses exactly the same teacher population as bilateral v2:

- same 10 files and hashes
- same 5,000 teacher games
- same 298,641 positions
- same 9/1 train-validation split
- same seed
- same `cPuct = 1.20`
- same LR, batch, gamma, wall-poor weighting and policy targets

The intended comparison is architectural, not a data-generation comparison.

## Training protocol

- fresh initialization
- 60 epoch maximum, early stopping
- batch 8,192
- LR 3e-4
- WL gamma 0.995
- wall-poor WL weight 3.0
- visit-policy targets from ply 0
- monitor `val_outcome`

## Results so far

Architecture tests:

- incremental accumulator parity: **PASS**, 10,398 plies
- exact quantized value antisimmetry: **PASS**, 4,618 positions
- measured `|z(s)+z(swapped)|`: mean 0, max 0

Training:

- best validation outcome loss: **0.2909**, epoch 33
- early stop at epoch 41
- this validation result is worse than v2's 0.2775, so antisimmetry alone did not improve held-out BCE under this capacity/training setup

Wandering/calibration:

- wall-poor progress signal: P(win) 0.2780 at own distance 7 -> 0.8351 at distance 1, delta +0.5570
- wandering race suite: **5/5 PASS**
- remaining monotonicity violations exist in isolated distance/wall probes, so the head is not globally monotonic despite exact antisimmetry

Artifacts:

- workflow run `33993412820`
- candidate artifact `antisym-v3-completed`, artifact `9977367287`

Arena:

- **not measured yet**. The initial 1,200-game invocation failed before game 1 because the CI checkout had not fetched `fix/strong-selfplay-wandering` as a usable worktree ref.

## Decision rule

Do not promote based on the elegant invariant alone. Require actual Elo and wandering evidence. Because v3 has worse validation BCE than v2 but cleaner mathematical structure, the direct v3-v2 arena is important.

## Next action

1. rerun v3 vs Gen8 with remote refs fetched explicitly;
2. run direct v3 vs v2 with their candidate artifacts;
3. if v3 loses, test whether the 32-unit shared `F` is capacity-limited before rejecting antisimmetry itself.
