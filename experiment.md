# Experiment: Bilateral v2 with WL Gamma 0.9975

## Status

**Prepared. Training and evaluation workflow pending in this branch.**

## Hypothesis

The 0.995 WL duration discount creates a useful progress gradient and fixed wandering, but it may be stronger than necessary and distort calibration away from pure win probability. Increasing gamma to 0.9975 weakens only the time preference while keeping a nonzero incentive to finish races.

## Controlled change

Only one training hyperparameter changes relative to `exp/full-accumulator-v2`:

- `wl-gamma`: **0.995 -> 0.9975**

Everything else stays fixed:

- bilateral concat value architecture
- same 5,000 teacher games / 10 V3 shards
- same train/validation split and seed
- `cPuct = 1.20`
- batch 8,192
- LR 3e-4
- wall-poor WL weight 3.0
- visit-policy targets from ply 0

## Evaluation

- validation outcome loss
- wall-poor progress curve
- 5-case wandering suite
- 1,200-game arena vs Gen8 search baseline
- direct comparison against gamma 0.995 if needed

## Decision rule

Keep 0.9975 only if it preserves zero wandering failures and improves Elo or calibration. A flatter progress curve without strength gain is a rejection.
