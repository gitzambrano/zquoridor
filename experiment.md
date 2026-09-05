# Experiment: Bilateral v2 with Wall-Poor WL Weight 1.5

## Status

**Prepared. Training and evaluation workflow pending in this branch.**

## Hypothesis

Bilateral v2 fixed the reported wandering reproduction, but it trained with a 3.0x WL loss multiplier on wall-poor race states. That may over-bias a relatively small subset of positions and hurt global Elo. Lowering only this multiplier to 1.5 may retain race-progress sensitivity while improving overall calibration and strength.

## Controlled change

Only one training hyperparameter changes relative to `exp/full-accumulator-v2`:

- `wall-poor-wl-weight`: **3.0 -> 1.5**

Everything else stays fixed:

- bilateral concat value architecture
- same 5,000 teacher games / 10 V3 shards
- same train/validation split and seed
- `cPuct = 1.20`
- batch 8,192
- LR 3e-4
- WL gamma 0.995
- visit-policy targets from ply 0

## Evaluation

- validation outcome loss
- race/value probes
- 5-case wandering suite
- 1,200-game arena vs Gen8 search baseline
- direct comparison against bilateral v2 if the signal is close

## Decision rule

Prefer this branch only if wandering remains clean and Elo improves. Validation loss alone is not sufficient.
