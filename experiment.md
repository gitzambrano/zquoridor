# Experiment: Policy Low-Rank Rank 48

## Status

**Rejected. Rank 48 fails the wandering hard gate and preserves too little of the trained policy matrix.**

## Hypothesis

The production policy head is a single dense `256 -> 209` projection. It costs 53,504 multiply-accumulates per policy evaluation. A truncated SVD can replace it with a linear `256 -> 48 -> 209` factorization, nominally reducing policy-head MACs to 22,320 (-58.3%) while leaving the Gen8 trunk and value head unchanged.

## Controlled change

Only the policy projection was changed. Gen8 trunk/value weights, `cPuct = 1.20`, policy-ordering settings, MCAB parameters, and the Gen8 int8 weight file were otherwise preserved.

## Spectral result

Measured directly from the trained Gen8 quantized policy matrix:

| Rank | Energy retained | Relative Frobenius error |
|---:|---:|---:|
| 16 | 34.58% | 0.8088 |
| 32 | 51.76% | 0.6946 |
| 48 | 64.23% | 0.5981 |
| 64 | 73.97% | 0.5102 |
| 96 | 87.44% | 0.3544 |
| 128 | 95.10% | 0.2214 |

Rank 48 therefore discards about 35.8% of the policy-matrix spectral energy.

## Regression result

Core MCAB tests: **PASS**.

Wandering race suite: **4/5 PASS**.

The failing case was:

- `race_6v8_w10`: did not reach the goal, entered a two-cycle, `noProgress=4`, final own distance 5.

The other four race cases passed.

Relevant runs:

- initial infrastructure run: `34001668873` (failed because the wandering test file was absent from the branch);
- corrected regression run: `34004606082`;
- corrected-run artifact: `9980544265`.

## Decision

**Reject rank 48 without arena.** Wandering is a hard regression gate, so running the planned 1,600-game arena would spend compute on a candidate already unsuitable for promotion.

Rank 32 is also rejected without testing because it retains only 51.8% of spectral energy, materially worse than rank 48. Rank 64 still has substantial reconstruction error (0.510) and only 74.0% energy retention, so the low-rank family is not the preferred next direction.

## Next action

Prefer exact-policy compute reductions over approximate compression: compute fewer policy logits / compute policy only when needed, preserving the trained Gen8 logits exactly. The active `exp/policy-lazy-legal-v7` branch is the next candidate to evaluate.
