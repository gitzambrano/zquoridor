# Local teaching and training

This guide explains the data contract. For which command to run, its input, and
its output, start with [scripts.md](scripts.md). For the completed campaigns and
measured network results, see [plan.md](plan.md).

## Data contract

All new self-play, replay, teaching, and training uses canonical V3 data in the
mover-relative frame:

- policy: a normalized distribution over 209 legal-action slots;
- value: signed mover-relative value in `[-1, 1]`;
- state: pawn positions, wall bitboards, distances, and wall stocks;
- split: whole trajectory/opening groups, never random rows.

Raw V3 shards are not a training dataset. Replay or teaching creates one
`dataset.npz`, a manifest, source weights, and a group-safe validation split.
Never mix two datasets merely because both files are named `dataset.npz`.

## Teaching modes

| Mode | Teacher | Best use | Risk to control |
| --- | --- | --- | --- |
| Direct | old NNUE and direct Claustrophobia output | broad coverage | do not let raw direct labels dominate a targeted search set |
| Search | Zquoridor search and Claustrophobia MCTS | corridors, wall tactics, disagreement, unstable positions | expensive; use selected positions rather than a tiny arbitrary sample |
| Mixed | direct plus selected search | normal candidate training | record source shares and sample weights in the manifest |

Claustrophobia's search result is a useful teacher; its raw prior network alone
is not a replacement for search supervision. Search labels must preserve the
side-to-move perspective and target provenance.

## Standard flow

1. Generate V3 self-play with `tools/selfplay/run_selfplay.py` or the balanced
   controller when category coverage matters.
2. Audit or migrate shards with `training/audit_selfplay.py` and
   `training/migrate_selfplay_v3.py` when needed.
3. Create replay or teaching data with `training/prepare_replay.py` or
   `training/run_teaching.py`.
4. Train one architecture with `training/run_experiment.py`; its output records
   config, architecture, metrics, float weights, int8 weights, and matching
   executable.
5. Verify Python/C++ parity before any arena.
6. Benchmark under the fixed 200 ms paired protocol. Training loss never
   promotes a network.

## Local storage

External bots, datasets, checkpoints, logs, self-play shards, and raw arena
results are local and ignored by Git. The local-only `docs/datasets.md` can
record machine-specific paths and inventories, but it must not be committed.
