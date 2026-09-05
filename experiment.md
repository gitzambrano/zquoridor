# Experiment: Bilateral v2 with Policy Loss Weight 0.5

## Status

**Prepared. Controlled training/evaluation workflow is being added.**

## Hypothesis

The bilateral v2 shares the 354 -> 256 transformer between value and policy and trains both losses at equal weight. The visit-policy teacher is already strong and dense, while the main unresolved weakness is value quality in race states. Reducing policy loss weight may let the shared representation allocate more capacity to WL without discarding policy supervision.

## Controlled change

Only one training hyperparameter changes relative to `exp/full-accumulator-v2`:

- `w-policy`: **1.0 -> 0.5**

Everything else stays fixed:

- bilateral concat value architecture
- same 5,000 teacher games / 10 V3 shards
- same train/validation split and seed
- `cPuct = 1.20`
- batch 8,192
- LR 3e-4
- WL gamma 0.995
- wall-poor WL weight 3.0
- visit-policy targets from ply 0

## Evaluation

- validation outcome and policy losses
- policy accuracy to ensure the reduction does not collapse move priors
- value/race probes
- 5-case wandering suite
- 1,200-game arena vs Gen8 search baseline

## Decision rule

Keep the lower policy weight only if playing strength improves without a meaningful policy-quality collapse or wandering regression. Lower validation outcome loss by itself is not sufficient.
