# Training data contract

This document defines source names, target semantics, and mixture weights.
`plan.md` owns measurements and collection status. The ignored `datasets.md`
owns the local file inventory. Experiment README files own model specifications.

## Canonical state and targets

Every prepared row uses the current mover frame. Player 1 positions mirror
rows before encoding. Actions contain 81 pawn destinations, 64 horizontal wall
slots, and 64 vertical wall slots. Values are signed targets in [-1, 1].

Required arrays are `own_pawn`, `opp_pawn`, `walls_h`, `walls_v`, `own_dist`,
`opp_dist`, `walls_left_own`, `walls_left_opp`, `policy`, `value`, `weight`,
and `is_val`. Each policy has 209 finite, nonnegative entries with unit sum.
NPZ inputs require `group_id` or `opening_index`. Directory inputs can instead
carry a verified group-separation certificate in `dataset.manifest.json`.

Optional arrays include `aux_policy_target`, `policy_sample_weight`,
`value_sample_weight`, `aux_sample_weight`, `search_depth`, `search_nodes`,
and `search_budget_ms`. A zero head weight means that the row has no target
for that head. A normalized placeholder policy then contributes zero loss.

## Source names and provenance

Use `family_version_worker_snapshot` for a new materialized source. Keep the
worker number in its download ledger. Do not place account identities or Drive
resource IDs in tracked files. Preparation adds a stable path digest when
several accounts use the same family name.

The catalog assigns one status to each discovered corpus:

- `available`: The input contains the required canonical fields and groups.
- `covered`: A selected descendant already contains these records, or this
  input changes only weights on an existing corpus.
- `excluded`: The input lacks a required field, group provenance, or completion.

The production master retains its historical nominal sample weights. Its
background anchor and teacher curriculum remain part of the mixture. The
stored replay used for the central fine-tune remains a separate source.
The catalog does not count the master ancestors a second time.

The standard catalog scans canonical `dataset.npz` files and completed
directory datasets. A raw binary, legacy heuristic record, inference cache,
or unaligned target archive is not an eligible training corpus. Use the
existing replay preparers before adding those inputs. All eligible canonical
sources enter the mixture without a position cap. `--sources-file` can supply
an explicit list of sources, provenance settings, and fractions.

## Default source budgets

These fractions control base gradient mass independently in each split.
Members of a pool share its budget in proportion to eligible capped mass.

| Pool | Base fraction | Policy quality | Value quality |
| --- | ---: | ---: | ---: |
| Historical curriculum | 0.25 | 0.8 | 1.0 |
| Replay | 0.15 | 0.8 | 1.0 |
| Search teachers | 0.15 | 1.5 to 2.0 | 1.5 to 2.0 |
| Wall Drops root visits | 0.10 | 2.0 | 1.5 |
| Gumbel completed policies | 0.15 | 4.0 | 2.0 |
| Temporal search values | 0.20 | 1.0 | 3.0 |

These are initial experiment settings, not measured optimal weights. Missing
pools redistribute their base budget proportionally among available pools.
The source manifest records effective fractions. The five-account check
remains active even if an entire pool is absent.

The mixer preserves existing per-head multipliers. It applies the source
quality multiplier separately to policy, value, and auxiliary policy. A
recorded search effort can add a square-root multiplier limited to 0.5 to 2.
Effort references belong to a single search family. Alpha-beta depth and MCTS
simulations are not interchangeable. Missing effort metadata adds no bonus.
Head multipliers have a default cap of 12. Source sample weights have a cap
of 30 before pool normalization. The campaign disables further effective
clipping in the trainer to preserve the selected mixture.

For a state, the mixer accumulates each head target with that head's evidence
mass. It divides by the head mass to obtain the target. It stores head mass
divided by base mass as the corresponding sample multiplier. Thus a dense
policy can dominate policy supervision without changing the value target.
The auxiliary policy retains its own target and mass.

The shared training recipe can add a frozen-reference surprise multiplier.
Its raw signal is KL from the current-position search policy to the frozen
production prior. The cache identity binds the dataset, reference, and code.
Separate training-only scales preserve policy and auxiliary evidence mass.
The recipe never scales value weights or the source mass array. Source
confidence and search effort remain independent of surprise. Recipe ablations
disable only this additional multiplier.

Entropy does not determine confidence. A forced tactical move can correctly
have a concentrated policy. An array of 209 entries can still contain a
truncated top-eight distribution. Policy provenance, search effort, and
temporal coverage determine quality. Gumbel policy completion includes
estimates for actions outside the searched set. Dense support does not mean
that search examined every action equally.

## Rich snapshots and temporal values

Prefer `positions.jsonl` and `targets.jsonl` over a cached `targets.npz`.
Join labels by position ID. Teacher labels store values and policies inside
the `target` object. Preserve the full current-position policy. Never build
a policy target from future moves.

Repeated IDs retain each corresponding annotation in occurrence order. The
mixer then averages their evidence by canonical state. It does not replace
all repeated searches with the final annotation. Repeated rows do not count
as distinct states in the dataset report.

Gumbel prefix chains form trajectory groups. Available values at exactly
0, 4, 8, and 16 plies use coefficients 0.5, 0.25, 0.125, and 0.0625.
Convert each value to the original mover perspective before averaging.
An isolated position has no future target. Do not invent a terminal result.
The rich preparer records actual future coverage and doubles the value
multiplier only for rows with an available future evaluation.

Complete games and aligned V3 shard metadata use the existing temporal
preparer. That preparer also permits a 10% known terminal-outcome component.
It uses the actual ply index, not the index of a stored record. Current root
policies remain at the original position. External teacher roots retain
their existing quality adjustments.

The campaign accepts frozen local manifests through `raw_sources`. The
adapter verifies accepted binary and metadata hashes and aligned record
counts. It creates the same ledger used for downloaded trajectories. It
does not treat an active campaign manifest as a frozen dataset.

## Splits, identity, and artifacts

Preserve trajectory groups within each source. Remove entire source groups
that cross existing splits. Remove canonical states that occur in both
splits across sources. This verifies recorded groups and exact states. It
does not certify unseen opening families when the original corpus lacks
that provenance.

A pooled source with no safe mass contributes zero weight. The manifest
records that exclusion. Other eligible sources absorb its budget. An empty
pool redistributes its budget among the remaining pools in each split.
An explicit source fraction still requires positive mass in both splits.

Hash source files and output arrays. The manifest binds source settings,
input identities, and preparation code. A changed snapshot or preparation
recipe requires a new output directory. A training checkpoint binds the
dataset, model code, initialization, configuration, optimizer, and random
states. Preserve those identities when resuming.

Generated fields, SQLite files, checkpoints, inventories, and download ledgers
remain in ignored artifact directories. A completed manifest is the dataset
completion marker. It is not a game-strength result.

The mixer sorts mapped state records in place and reduces split conflicts
in chunks. It does not allocate a full inverse index in RAM. The mapped
policy accumulators become the final target arrays. This avoids a second
full copy of each dense policy matrix during preparation.
