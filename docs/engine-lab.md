# Engine lab

`main` carries one permanent experimental entry point: `.github/workflows/engine-lab.yml`.
It is infrastructure only; it must not change engine strength or silently promote a candidate.
Generation-specific workflows remain historical references, while new work should prefer the lab.

## Stable identities

The current external reference setup is intentionally pinned:

- Gen8 production checkpoint: `checkpoints/gen8/nnue_weights.bin` and
  `checkpoints/gen8/nnue_weights_int8.bin`; provenance is in
  `checkpoints/gen8/PRODUCTION_MANIFEST.txt` (source run `33973129847`).
- Titanium: `titaniummachine1/titanium-engine` v19.8.10, commit
  `1ac94755f69799f01b2e06869403e701bbf0bd51`.
- Frozen Titanium harness source: commit
  `0d28b7024ec0a7ac83a7f1a9282d09ec12e483a0`.
- Frozen comparison openings: `tools/external/openings_titanium.jsonl`.
- Reference control: 200 ms/move, paired colors, `TITANIUM_PONDERING=0`.

The four benchmark files under `tools/external/` are copied byte-for-byte from the frozen harness commit.
Do not casually edit them: a harness change creates a new benchmark protocol and must be identified
separately in reports.

`tools/external/build_pinned_titanium.sh` rebuilds Titanium from the exact external SHA. Historical
Actions artifacts remain useful caches, but the permanent lab does not depend on their retention period.
The Gen8 fallback is likewise the versioned production checkpoint, not an expiring artifact.

`tools/external/analyze_paired_arena.py` is deliberately separate from the frozen harness. It groups the
two color-swapped games of each opening and bootstraps those pairs as the independent units. Keep the
historical game-level interval for continuity, but prefer the pair-aware interval for promotion decisions.

## Lab modes

The workflow is manually dispatchable in six modes.

- `smoke`: syntax/build/core regression checks for permanent infrastructure.
- `selfplay`: generate self-play with `tools/selfplay/run_selfplay.py`.
- `train`: run an arbitrary Python trainer path plus arbitrary arguments. This makes the orchestration
  independent of a particular neural architecture; the trainer owns its model and output format.
- `arena`: invoke the internal arena runner with caller-supplied arguments.
- `titanium`: run a paired direct benchmark against the pinned Titanium, including a two-game smoke first,
  and save both historical game-level statistics and a paired-opening bootstrap report.
- `teacher`: collect active, architecture-neutral teaching data from the pinned Titanium. The same position
  is probed by Titanium and the pinned Gen8 student, so the corpus records agreement/disagreement as well as
  the teacher result. The job also emits high-confidence disagreement samples, unstable positions for deeper
  relabeling, and a canonical `teacher_policy.npz` ready for policy distillation.

A model with a new runtime format still needs a corresponding evaluator/weight loader in the C++ engine.
Pipeline agnosticism does not make the production engine understand an unsupported format automatically.

## Active teaching pipeline

`training/teachers/collect.py` is the architecture-neutral collector. It supports repeated probes and
multiple search budgets, preserves the raw engine `info` stream, and stores a normalized deepest rich
snapshot when the teacher exposes structured root information. `training/teachers/titanium_collect.py`
keeps the historical Titanium-specific command line as a thin wrapper around that collector.

The important diagnostic is not just whether the teacher chose move X. For each position the corpus can
record:

- teacher stability across repeated searches and search budgets;
- the highest-budget vote fraction and whether budget winners agree;
- the student move on exactly the same position;
- whether teacher and student agree;
- the rich Titanium root snapshot (`rootMoves`, root score, depth/nodes when available).

`training/teachers/select_samples.py` turns those diagnostics into two useful queues. `disagreement` keeps
stable/high-confidence positions where the student differs from the teacher; `unstable` keeps positions
where the teacher itself is not stable and therefore deserves more search before being trusted as a label.
This prevents expensive teacher time from being spent uniformly on positions the student already handles.

`tools/teacher/encode_position.cpp` replays the move history with the production ZQuoridor rules and emits
the compact state in the same canonical mover perspective used by NNUE training. The textual protocol is
explicitly framed (`ZQTEACH`) and byte-sized state fields are serialized numerically, so wall counts cannot
turn into control characters. `training/build_teacher_policy.py` consumes that protocol and writes the
architecture-specific `.npz`; the raw teacher JSONL remains architecture-neutral and reusable.

The first controlled distillation experiment is `training/train_teacher_policy.py`. By default it updates
only the 256->209 policy head from an existing Gen8 float checkpoint and verifies that the accumulator and
value head remain bit-for-bit unchanged. This intentionally isolates policy teaching from feature/value
changes. A later experiment may opt into trunk training or richer soft/ranking targets, but those are
separate gates rather than hidden changes to the first baseline.

## Benchmark data is not teacher data

The frozen 40-opening Titanium comparison set is an external benchmark. It must not become training data.
The teacher collector rejects both the benchmark file itself and any independently named teacher corpus
containing an exact opening from that set. Use an independent opening corpus, split by whole openings, and
keep a separate unseen external test set.

The raw v2 teacher schema stores move history, side to move, selected teacher move, all configured probes,
raw `info` lines, normalized rich snapshots, stability diagnostics, and optional student diagnostics. This
lets later converters build one-hot, soft/ranking, value, or architecture-specific targets without coupling
collection to the current 354/256/209 NNUE layout.

## Promotion discipline

Keep `main` immutable during strength exploration. A candidate should normally pass code regressions,
fixed-200-ms internal H2H, NPS/behavior checks, Titanium smoke, then a paired external screen and a larger
confirmation before any strength promotion. Eighty external games are a directional screen, not proof of
parity or superiority.
