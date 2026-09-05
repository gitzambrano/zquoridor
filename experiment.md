# Experiment: Antisymmetric Bilateral Value v3, Width 64

## Status

**Prepared. Architecture patch and training/evaluation workflow pending in this branch.**

## Hypothesis

The exact-antisymmetric v3 is mathematically cleaner and passes all wandering tests, but its best validation outcome loss (0.2909) is worse than bilateral v2 (0.2775). The v3 control reduced each shared `F` hidden layer to 32 units to keep compute near v2. The loss gap may therefore be capacity-limited rather than caused by the antisymmetry constraint itself.

## Controlled change

Only value-head width changes relative to `exp/antisym-value-v3`:

- shared `F` hidden width: **32 -> 64**

The exact form remains:

`V(A,B) = F(A,B) - F(B,A)`

so quantized antisimmetry must remain exact.

Everything else stays fixed:

- shared transformer 354 -> 256
- same policy head
- same 5,000 teacher games / 10 V3 shards
- same split and seed
- `cPuct = 1.20`
- batch 8,192
- LR 3e-4
- WL gamma 0.995
- wall-poor WL weight 3.0

## Expected tradeoff

This approximately doubles the value-head first-stage MACs relative to v3 width-32. It should only survive if the extra capacity buys clear Elo or calibration improvement.

## Evaluation

- compile and quantization checks
- exact antisymmetry test
- validation outcome loss
- value/race probes
- 5-case wandering suite
- microbenchmark / whole-engine NPS
- 1,200-game arena vs Gen8
- direct comparison vs v3 width-32

## Decision rule

Reject if Elo does not materially improve enough to justify the extra inference cost.
