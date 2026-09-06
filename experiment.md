# Experiment: Exact Lazy Policy v7

## Status

**RUNNING.** This experiment starts from `fix/strong-selfplay-wandering` and tests an exact policy-head optimization. No trained weight, policy logit used by the search, value evaluation, or search parameter is changed.

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
- no approximation or retraining;
- wandering must remain 5/5.

## Evaluation

1. Exact legal-logit parity test on zero-wall positions against a dense reference dot product.
2. Core MCAB tests.
3. Five-case wandering suite.
4. 1,600-game fixed-time arena vs `fix/strong-selfplay-wandering`, same Gen8 weights, 20 ms/move, 4 threads, 4 random opening plies.
5. Compare W/L/D, Elo ±95%, and whole-engine NPS.

## Decision rule

Because the change is mathematically exact on all reachable policy outputs, any fixed-time Elo movement should come from timing/search-volume effects. Retain only if tests pass and NPS/Elo are non-negative enough to justify a longer confirmation. If the gain is real, extend the same exact idea to legal-row evaluation for wall-available nodes.
