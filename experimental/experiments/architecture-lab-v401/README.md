# Architecture lab for version 4.01

This directory defines eight distinct experimental networks, one control,
and seven matched recipe experiments.
`experiment.json` specifies one model and its training overrides. `suite.json`
selects the default sequence. Generic implementations remain in `training/`.

| ID | Model | Initialization | Extra representation |
| --- | --- | --- | --- |
| 00 | Production control | Production copy | Existing 858 inputs |
| 01 | Joint race NNUE | Exact transfer | 121 distance-pair buckets |
| 02 | Corridor NNUE | Exact transfer | 64 buckets for goal-directed exits and forced prefixes |
| 03 | Wall-pair NNUE | Exact transfer | 512 stable hash buckets for pairs and boundary motifs |
| 04 | Auxiliary topology NNUE | Exact transfer | Eight geometry labels for a training-only head |
| 05 | Context gate NNUE | Identity adapter | 16 channel groups and a 16-unit context MLP |
| 06 | Shared action decoder | Zero residual adapter | Action embeddings of width 32 and a shared MLP |
| 07 | Graph network | Distillation | 81 nodes, width 32, four residual graph blocks |
| 08 | Token Transformer | Distillation | Up to 24 tokens, width 32, two layers, two heads |

All variants retain six value heads and a 209-action policy. Graph and
Transformer trunks replace the sparse accumulator. Their first five epochs
add distillation from the frozen production reference. True dataset targets
remain active during distillation. The NNUE extensions start with identical
value and policy outputs to the reference. Added feature columns start at zero.

The pair encoder uses binary hash features. Hash collisions are intentional.
The topology cache stores complete goal-distance maps for a wall layout.
Those maps ignore pawn occupancy. Existing contact inputs retain jump geometry.
The auxiliary topology head supervises branching, forced-prefix length,
physical exits, and distance. It does not prove a wall combination under
adversarial replies. The action decoder scores all actions before legal
filtering. It does not depend on the old policy's top candidates.

## Annealing and recipe comparisons

Warm-started models use 40 epochs with cosine decay from `1e-4` to `1e-7`.
The accumulator uses one tenth of the head rate. Thus its initial rate is
`1e-5`. The graph model uses 100 epochs from `1e-3` to `1e-5`. The Transformer
uses 120 epochs from `3e-4` to `3e-6`. These are experimental schedules.
The models do not stop early, so each schedule reaches its final rate.

The batch setting starts at 8192. CUDA training automatically measures full
forward, backward, and Adam memory demand. The calibration tests powers of
two from 256 through 65536 and selects the largest successful batch. The
memory budget uses 85% of the actual GPU capacity and accounts for other
processes. On a 6 GiB GPU, this permits approximately 5.1 GiB before that
adjustment. The calibration restores weights and random state after probes.
The report records the selected batch and measured peaks.

Matched recipes share the calibration cache for their architecture. The
probe includes auxiliary-head memory even when a recipe disables that loss.
Thus an auxiliary-loss comparison retains the same batch. Graph and
Transformer models receive separate calibrations. Their activations usually
require smaller batches. A CPU run uses the fixed batch setting.
Set `--no-auto-batch-size --batch-size 8192` to select a fixed batch.
Use `--batch-size-max` and `--gpu-memory-fraction` to change the limits.

The shared recipe retains the existing auxiliary policy at temperature 2
and loss coefficient 0.15. It also adds frozen-reference surprise weighting
with alpha 0.5 and KL cap 4. The reference remains fixed for all architectures.
The weight cache preserves the training evidence mass of each policy head.
It does not change value confidence or derive a scale from validation rows.

Each recipe experiment changes one factor relative to its named control:

| ID | Reference | Single changed factor |
| --- | --- | --- |
| 09 | 00 | Initial head rate `3e-5` instead of `1e-4` |
| 10 | 00 | Disable the additional frozen-reference surprise multiplier |
| 11 | 00 | Auxiliary policy coefficient 0.25 instead of 0.15 |
| 12 | 00 | 80 annealing epochs instead of 40 |
| 13 | 02 | Initial head rate `3e-5` on the corridor model |
| 14 | 03 | Auxiliary policy coefficient 0.25 on the wall-pair model |
| 15 | 00 | Disable the training-only auxiliary policy |

Source quality and any multipliers already present in the corpus remain
constant in these comparisons. Experiment 10 does not erase that provenance.
The seven recipes use the same seed, source mixture, and initialization as
their references. Tests verify the single changed configuration field.
Compare common policy KL and value metrics across recipes. Their combined
weighted loss is not the same objective. Game strength still requires matches.

## Entry point

Run commands from the repository root:

```powershell
python training/train_nnue.py
python training/train_nnue.py --cli-command initialize
python training/train_nnue.py --cli-command plan --cloud inventory
python training/train_nnue.py --cli-command plan --cloud download
python training/train_nnue.py --cli-command prepare
python training/train_nnue.py --cli-command train
```

The default command writes a plan. `initialize` creates all initial models
and freezes the reference weights. Neither command starts a training campaign.
`prepare` creates the common dataset. `train` prepares or verifies that dataset
and trains the selected models sequentially. Checkpoints permit interruption.
The runner never starts a benchmark, tournament, self-play job, or promotion.

Edit `CONFIG["campaign"]` in `training/train_nnue.py` for normal use. Every
campaign setting also accepts a CLI override. Lists and dictionaries use JSON.
The default `epochs=0` selects each recipe's duration. A positive `--epochs`
overrides all selected durations. `--training-overrides` applies other shared
optimizer or loss overrides after the model recipe.
For example, run only the corridor model:

```powershell
python training/train_nnue.py --cli-command train --experiments '["02_corridor"]' --device cuda --epochs 40
```

Use a new `--output` after a change to source contents, preparation code, or
mixture settings. The same output and configuration resume an interrupted run.
Use `--workflow legacy` or an explicit `--mode` for the retained soup workflow.

## Data and runtime boundaries

The catalog includes the previous production curriculum, stored replay,
eligible local teacher corpora, and all supplied download ledgers. Composite
ancestors remain visible with a coverage reason. The default preparation
also derives temporal values from the frozen central-weakness and V3.00
shard manifests. `raw_sources` selects these hash-verified local inputs.
It preserves the current policy and uses actual game plies for future values.
The default preparation
requires paired recent version 4.01 files from workers 1 through 5. Add
materialized ledgers through `--downloads`. Read [the data contract](../../../docs/training-data.md)
for target quality, source budgets, grouping, and file names.

The CLI snapshot helper reads existing runtimes only. It neither executes a
cell nor allocates a VM. A missing assignment or Drive permission does not
count as a completed snapshot. Materialized connector downloads can use the
same ledger contract when a runtime is unavailable.

Models 00 and 04 can export the existing native weight layout. All other
models export a reloadable PyTorch checkpoint and an explicit architecture
manifest. They have no C++ or WASM evaluator yet. New adapters use floating
point. Inherited heads and NNUE accumulators use the existing QAT contract.
Graph and Transformer activations also use simulated quantization before
the inherited heads. No manifest declares native compatibility for an
unsupported model.

Generated output resides under `artifacts/` in this directory. It contains
the plan, source catalog, prepared inputs, common dataset, frozen reference,
initial models, per-model checkpoints, and reports. The cloud download ledger
resides in root `artifacts/architecture_campaign/cloud/`. Actual collection
status, measurements, and next actions belong only to `docs/plan.md`.
