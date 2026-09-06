# Experiment: Exact Lazy Policy v7

## Status

**Promising; long confirmation required.** The exact zero-wall policy-row pruning passes all correctness gates and shows a small positive fixed-time signal with a measurable NPS gain.

## Motivation

The rejected rank-48 SVD experiment showed that approximating the policy matrix can reintroduce wandering. The production policy head is already a single quantized linear map (`256 -> 209`), so the safer optimization is to avoid computing outputs that cannot be consumed rather than approximate the matrix.

Production search is MCAB. At each expanded node it computes the complete 209-logit policy vector even when the side to move has no walls. In that state every wall move is impossible, so only the 81 pawn logits can be read by `legalMoves`/PUCT.

## Controlled change

When `AccumulatorQuant::ownWallsLeftBucket == 0`:

- compute the same 256-element SCReLU activation as baseline;
- compute the same integer dot products for policy outputs `0..80` (pawn destinations);
- skip the 128 wall-output dot products;
- initialize skipped wall outputs to zero defensively; they are unreachable because the mover has zero walls.

When the mover has at least one wall, `forwardPolicyQuant` is exactly the baseline implementation.

The useful policy work in zero-wall nodes falls from 209 rows to 81 rows, a **61.2% reduction in policy-row dot products in that regime**.

## Invariants

- same Gen8 int8 weights;
- same value head and accumulator;
- same `cPuct = 1.20`;
- same MCAB parameters;
- same policy logits bit-for-bit for every legal move;
- no approximation or retraining.

## Correctness gates

Workflow `34004852374`:

- exact legal-logit parity on zero-wall test positions: **PASS**;
- core MCAB tests: **PASS**;
- wandering suite: **5/5 PASS**.

Generated source commit:

- `1cad3d9e0eefe30241de88a300a3dcbafe1d804c` — `nnue: skip unreachable wall policy rows at zero walls`.

## 1,600-game arena

Protocol:

- candidate: `exp/policy-lazy-legal-v7`;
- baseline: `fix/strong-selfplay-wandering`;
- same Gen8 int8 weights;
- 20 ms/move;
- 4 threads;
- 4 random opening plies;
- policy ordering enabled, min depth 3;
- `cPuct = 1.20`.

Result:

- candidate wins: **728**;
- baseline wins: **715**;
- draws: **157**;
- Elo: **+2.8 ±16.2**;
- candidate NPS: **27,019**;
- baseline NPS: **26,828**;
- NPS delta: **+0.71%**.

Artifact:

- `policy-lazy-legal-v7-results`, artifact `9980716114`.

## Interpretation

The Elo result is statistically inconclusive but centered slightly positive. More importantly, the candidate is mathematically exact for every legal policy output, passes the wandering hard gate, and produces a repeatable whole-engine throughput gain. The remaining question is whether the timing/search-volume benefit is large enough to produce a stable Elo gain over a longer sample.

## Decision rule

Run a 4,000-game fixed-time confirmation against the exact baseline. Promote only if correctness remains clean and the longer result is non-negative with a persistent NPS gain; prefer a positive lower confidence bound for a definitive strength promotion.

## Next action

Run `experiment-policy-lazy-legal-v7-long` for 4,000 games under the same 20 ms / 4-thread / 4-random-ply protocol.
