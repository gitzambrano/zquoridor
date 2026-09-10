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

## Lab modes

The workflow is manually dispatchable in six modes.

- `smoke`: syntax/build/core regression checks for permanent infrastructure.
- `selfplay`: generate self-play with `tools/selfplay/run_selfplay.py`.
- `train`: run an arbitrary Python trainer path plus arbitrary arguments. This makes the orchestration
  independent of a particular neural architecture; the trainer owns its model and output format.
- `arena`: invoke the internal arena runner with caller-supplied arguments.
- `titanium`: run a paired direct benchmark against the pinned Titanium, including a two-game smoke first.
- `teacher`: collect raw, architecture-neutral Titanium best-move data and raw info lines.

A model with a new runtime format still needs a corresponding evaluator/weight loader in the C++ engine.
Pipeline agnosticism does not make the production engine understand an unsupported format automatically.

## Benchmark data is not teacher data

The frozen 40-opening Titanium comparison set is an external benchmark. It must not become training data.
`training/teachers/titanium_collect.py` refuses that file as teacher input. Use an independent opening
corpus, split by whole openings, and keep a separate unseen external test set.

The raw teacher schema stores move history, side to move, Titanium best move, elapsed think time, and raw
`info` lines. A later converter can build one-hot, soft/ranking, value, or architecture-specific targets
without coupling collection to the current 354/256/209 NNUE layout.

## Promotion discipline

Keep `main` immutable during strength exploration. A candidate should normally pass code regressions,
fixed-200-ms internal H2H, NPS/behavior checks, Titanium smoke, then a paired external screen and a larger
confirmation before any strength promotion. Eighty external games are a directional screen, not proof of
parity or superiority.
