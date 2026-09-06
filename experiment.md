# Experiment: Policy Low-Rank Rank 48

## Status

**Prepared. Spectral preservation and inference-cost experiment starting from the Gen8 policy head.**

## Hypothesis

The production policy head is a single dense `256 -> 209` projection. It costs 53,504 multiply-accumulates per policy evaluation. Because the head is linear, its trained weight matrix can be approximated directly with a truncated SVD instead of retraining the network from scratch.

A rank-48 factorization replaces the projection with:

`256 -> 48 -> 209`

with no nonlinearity between factors. This reduces the nominal policy-head MAC count from 53,504 to 22,320, a reduction of about 58.3%, while preserving the Gen8 trunk and value head exactly.

## Controlled change

Only the policy projection changes. The experiment must preserve:

- Gen8 transformer/trunk weights;
- Gen8 value head;
- `cPuct = 1.20` search baseline;
- policy ordering enablement and minimum depth;
- all other MCAB parameters.

The low-rank factors are initialized from the trained Gen8 policy matrix with truncated SVD. The original policy bias is retained.

## Evaluation order

1. Measure singular-value energy retained at ranks 16/32/48/64/96.
2. Measure policy-logit reconstruction error on representative positions.
3. Measure KL divergence and legal-move top-1/top-3 agreement against the original Gen8 policy.
4. Benchmark isolated policy forward and whole-engine NPS.
5. Run wandering regression.
6. If policy fidelity is acceptable, run a fixed-time arena against the untouched Gen8 search baseline.

## Decision rule

Rank 48 survives only if it produces a meaningful NPS gain without a statistically meaningful Elo loss. If fidelity is substantially better than necessary, test rank 32. If fidelity is insufficient, test rank 64.
