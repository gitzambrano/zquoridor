# Closure: Apex-policy 4.10 versus 4.01 search at fixed 200 ms

Status: **closed** (15 November 2026). The production 4.10 NNUE
and native search remain unchanged.

## Controlled 400-game H2H

[GitHub Actions run](https://github.com/gitzambrano/zquoridor/actions/runs/43639981507)

The two searches were compiled from frozen C++ snapshots under identical
`-O3 -march=native -mavx2 -mfma` flags. Search 4.01 used source commit
`6e8efb08f7a9442f795f4c3110db3135d70eadf9`; search 4.10 was the
current release search. **Both loaded the same Apex INT8 network** with
SHA-256 `fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`.
The uploaded Apex policy is byte-identical to the active 4.10 policy.

Protocol: fixed real `go movetime 200`, no pondering, 200 distinct starting
positions, two reversed-color games per opening, legal referee,
100 Normal confirmation and 100 Center Rush pairs, ten isolated workflow
shards. GitHub Actions aggregation verified 400 complete games, all 200
complete color-reversed pairs, zero failed games, zero duplicate pairs,
and fixed-time protocol in every record.

| Opening | Games | Search 4.10 score | Point Elo estimate | Paired bootstrap 95% score |
| --- | ---: | ---: | ---: | ---: |
| Normal | 200 | 53.25% | +22.62 | 48.00–58.75% |
| Center Rush | 200 | 50.25% | +1.74 | 46.00–54.50% |
| **Total** | **400** | **51.75%** | **+12.17** | **48.25–55.25%** |

Bootstrap uses 10,000 resamples of paired openings. The overall interval
includes parity. There is no established competitive improvement or
regression in search 4.10 against 4.01 under this protocol.

## Context: uploaded network

The uploaded `zQuoridor401_apex_ordering.zip` is not a new architecture
or training product. Its Apex policy SHA-256 is identical to production
4.10. In a separate identical-search 64-game test, this policy achieved
59.375% versus original 4.01 policy; the paired bootstrap interval included
50%. See [policy-only trial](https://github.com/gitzambrano/zquoridor/actions/runs/43637497864).
A prior 32-game search-only pilot was inconclusive and superseded by the
400-game confirmation: [pilot](https://github.com/gitzambrano/zquoridor/actions/runs/43637756382).

## Final decision and subsequent comparison

- Do **not** revert search or replace the production network.
- Do **not** open additional 4.01-versus-4.10 searches absent a specific
  new hypothesis. All existing search-only H2H jobs are complete.
- The next useful strength comparison is a **fresh, controlled production
  4.10 versus pinned Claustrophobia and pinned Titanium**, separating
  200 ms fixed movetime and honest native 3+2, in Normal and Center Rush
  books, color reversed, with correctly enforced external CPU time.
  Use pinned opponents and book-disjoint validation where applicable.
  This measures actual competitive headroom rather than cycling old code.
- For any subsequent architecture or policy candidate, compare to
  **production 4.10** as frozen baseline, using both opening families and
  external opponents; only promote on reproducible strength evidence.

The 400-game raw records, per-shard summaries and aggregate report are
stored as artifacts of the run linked above. This document preserves
their measured conclusion on `main`, while experimental scripts remain
on their experimental ref until explicit branch deletion.
