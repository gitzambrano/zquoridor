# Architecture lab for version 4.01

This directory defines eight distinct experimental networks and one control.
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
