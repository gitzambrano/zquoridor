# Script catalog

This file is the entry point for operating Zquoridor. Run commands from the
repository root. Large data, checkpoints, opponent clones, logs, and arena
reports are local artifacts and are ignored by Git.

## Configuration convention

A public runner must have a documented configuration block at the top of its
file. Running it with no arguments uses those values; a CLI flag overrides one
value for a single invocation. A script that is a library, test, bridge, or
one-off migration is not a public runner and is not the normal way to operate
the project.

The public runners currently following this convention are `tools/setup_bots.py`,
`tools/run_benchmark.py`, `tools/run_clock_smoke.py`,
`tools/run_full_candidate_suite.py`, `tools/benchmark_candidate.py`,
`training/run_teaching.py`, `training/run_experiment.py`,
`training/run_experimental_anneal.py`,
`training/run_production_finetune.py`,
`training/run_campaign.py`, `tools/teacher/run_four_million_selfplay.py`,
`tools/teacher/run_weakness_selfplay.py`, and
`tools/teacher/run_central_weakness_rollouts.py`. `tools/selfplay/run_selfplay.py`
uses its documented uppercase settings at the top of the file. Other scripts
are libraries, tests, or reusable internal stages and are not standalone
campaigns.

## Normal workflows

| Goal | Canonical runner | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| Install external opponents | `tools/setup_bots.py` | pinned revisions | `external_bots/` | Run once or after a clean clone. |
| Benchmark production or one candidate | `tools/run_benchmark.py` | executable, NNUE, opening book | `results/benchmarks/` | Fixed move time, paired colours and shared openings are mandatory. |
| Full candidate gate | `tools/run_full_candidate_suite.py` | a built candidate | `results/benchmarks/` | Uses one generic suite configuration and optional phase skips. |
| Clock safety smoke | `tools/run_clock_smoke.py` | two UCI engines and openings | `results/benchmarks/` | Defaults to a safe dry run; set paths in `CONFIG` to execute. |
| Ordinary self-play | `tools/selfplay/run_selfplay.py` | architecture-matched self-play executable | `data/selfplay/` | The simple, configurable shard generator. |
| Balanced, resumable self-play | `tools/teacher/run_four_million_selfplay.py` | central/broad seed schedules and matching executable/NNUE | a V3 corpus with metadata sidecars | Active controller. It deduplicates states, resumes safely, and exposes its configuration block at the top of the file. |
| Build a seed schedule | `tools/teacher/build_selfplay_seed_schedule.py` | JSONL snapshots/openings | central and broad JSONL schedules | Internal stage used by the balanced controller. |
| Weakness self-play | `tools/teacher/run_weakness_selfplay.py` | any compatible position snapshot | weakness V3 shards | Resumable generic runner; configure paths and target counts at the top. |
| Central weakness rollouts | `tools/teacher/run_central_weakness_rollouts.py` | Center-Rush catalog plus production and contact-bucketed self-play executables/NNUEs | source-separated V3 + metadata corpus and optional stored-search replay | Shared network-independent 8-10-ply bank; identical wide/balanced/sharp profile cycle per source. |
| Convert and audit old self-play | `training/migrate_selfplay_v3.py`, then `training/audit_selfplay.py` | old shards | canonical V3 shards and manifest | Migration preserves original files. |
| Build replay/teaching data | `training/prepare_replay.py` or `training/run_teaching.py` | V3 shards or JSONL trajectories | one `dataset.npz` plus manifest | The dataset contains mover-relative policy and value targets. |
| Train one network | `training/run_experiment.py` | a canonical NPZ or a manifest-verified memory-map directory | float/int8 weights, checkpoints, manifest, matching executable | QAT, source metrics, and resume are configured here. |
| Prepare and anneal the experimental matrix | `training/run_experimental_anneal.py` | frozen champion, complete local and Colab replays | four initialized candidates, shared mixture, sequential training runs | Dry run by default; 120 epochs; low LR; 75% new and 25% historical effective mass. |
| Production-only fine-tune | `training/run_production_finetune.py` | frozen partial local snapshot, complete Colab manifests, historical dataset | one `production_bucketed512` candidate | 120 epochs; 75% new and 25% historical effective mass. |
| Mix frozen experimental data | `training/mix_experimental_datasets.py` | canonical datasets and explicit source or pool shares | memory-mapped arrays and a hashed manifest | Internal stage; weighted duplicate averages, group-safe splits, and exact effective shares. |
| Prepare all stored-search records | `training/prepare_stored_replay_all.py` | frozen accepted-shard manifests | memory-mapped 50/50 replay and source audit | Internal stage; disk aggregation, no sampling cap, and whole-game splits. |
| Train an architecture matrix | `training/run_architecture_matrix.py` | one dataset and matrix settings | one experiment directory per candidate | Auxiliary batch wrapper; use only when its matrix matches the experiment. |
| Complete campaign | `training/run_campaign.py` | configuration, data and optional teaching | data, candidates and arenas | Generic orchestration for a reproducible experiment. |
| Resilient remote/Colab self-play | `tools/selfplay/run_colab_worker.py` | seed openings, executable, NNUE | chunked V3 shards and metadata in Google Drive | Auto-resumes from existing Drive shards, unique worker seeds. |
| Inspect Colab workers | `scripts/colab/inspect_workers.py` | local Playwright profiles | status, hardware type, live output stream | Inspects Colab worker nodes headlessly without interrupting runs. |
| Launch Colab workers | `scripts/colab/launch_workers.py` | local Playwright profiles, bootloader | updated notebook cells, running self-play | Bootstraps resilient auto-clone cells and triggers execution. |
| Historical all-in-one cycle | `training/strong_cycle.py` | generation settings | self-play, replay, training and arena output | Legacy orchestration. Do not use for a new campaign until it is migrated to the same `CONFIG` contract. |

## Self-play transitions and provenance

The balanced controller accepts any positive `time_ms`. Its historical filename
does not restrict corpus size. Set targets and thread reservations in `CONFIG`.
Each new accepted shard records its command and SHA-256 hashes of the generator
and weights. Historical shards without these fields retain their original records.

Set `visit_temperature_after` and `visit_temperature_exe` to switch generators
at the first shard boundary after the unique-state threshold. Both executables
must match the same weights. `visit_temperature_args` controls the phase schedule.
The default schedule uses root visits and disables residual epsilon moves.

Create `stop.request` in the corpus directory to stop an updated controller
after its current shard transaction. Remove the request before resuming.
An older process does not acquire this behavior from a source edit.
Do not terminate a controller during shard admission or launch a second controller.

## Central weakness rollout campaign

Use `tools/teacher/run_central_weakness_rollouts.py` for the dedicated
Center-Rush recovery corpus. The default campaign builds 50,000 unique legal
8-10-ply roots around `e2 e8 e3 e7 e4 e6`, explicitly oversamples the
empirically weak catalog families, adds left/right mirrors, and does not use the
current NNUE to accept or reject opening roots. Self-play then cycles
wide/balanced/sharp MCAB visit-temperature profiles with Dirichlet root noise.
Temperature changes the played trajectory; the
stored policy supervision remains the untempered top-eight root-visit
distribution.

The 16 million design target assigns 8 million positions to the 504-feature
production bucketed champion (`data/nnue/nnue_weights_int8.bin`) and 4 million
to the 858-feature contact-bucketed student
(`results/experiments/multipath_contact_bucketed_unified/student_int8.bin`).
The remaining 4 million are explicitly unassigned. The contact executable is
built from its versioned `student.architecture.json` flags, separately from
`bin/selfplay`. The runner validates weight SHA and architecture before running.
Each source uses the same opening bank and local profile cycle.

The first 16 plies after each 8-10-ply seed are always searched at 200 ms.
Later plies use playout-cap randomization: full searches receive 200 ms and are
recorded; cheap 20 ms searches only steer the trajectory and are never recorded.
Every recorded target therefore has a 200 ms MCAB search. Every shard has an
aligned metadata sidecar and globally distinct game IDs across sources. The
optional replay stage caps the deduplicated dataset at 12 million states and
reports source contribution, overlap, root-value disagreement, opening family,
rollout ply and exploration profile. Source provenance is retained in the
campaign manifest and per-shard JSON. Training should mix about 70-80% of this
central replay with 20-30% frozen broad/general champion replay.

For this campaign the documented value target is a literal 50/50 blend:

`0.5 * signed_mcab_root_value + 0.5 * terminal_game_result`

It is implemented with `prepare_replay.py --mode stored_search`,
`stored_outcome_weight=0.5`, and `stored_gamma=1.0`. The raw NNUE forward
evaluation stored in V3 is deliberately not used as the value teacher. Before
promotion, mix a frozen broad/general replay slice into network training and
gate the candidate on both normal-book and Center-Rush paired external-opponent
matches.

## Stored-search replay

Use `training/prepare_replay.py --mode stored_search` to consume V3 visits and
aligned 20-byte metadata. Configure `source`, `out_dir`, and `max_positions`
in `CONFIG`, or override these fields through CLI options.
Managed corpora use only accepted manifest entries. This mode filters missing
root values and zero-visit rows without modifying the source files.

The value target is `(1-alpha)*(2*root-1) + alpha*gamma**plies*result`.
Set `stored_outcome_weight` for alpha and `stored_gamma` for gamma.
The policy target is the stored, renormalized top-eight visit distribution.
This format does not retain the full root distribution or Action-Q targets.

The sampler deduplicates states and separates game IDs between train and
validation. It shuffles shards with a reproducible seed and limits the sample
count. This is a bounded sample, not a globally uniform sample or an export of
the strongest reference in `states.sqlite`. Audit source proportions before
training and use a frozen corpus for reproducible datasets.
Use a new output directory when the input or configuration changes.

## Experimental annealing matrix

Use `training/run_experimental_anneal.py` for the four-candidate campaign.
The default command reports the recipe and missing inputs without starting
training. The matrix contains production bucketed 512 and contact bucketed
512, 768, and 1024. All candidates start from the frozen production champion.

Use `--no-dry-run --initialize-only` to export initial float and int8 weights,
architecture manifests, and `initial.pt` files. These files stay under the
ignored experiment directory. The wider candidates require the experimental
width flag in their architecture manifests.

Use `--no-dry-run --prepare-only --prepare-replays` after the raw sources
finish. Configure the download paths in `data_sources`. Each Colab source
needs a frozen `manifest.json` with `complete: true` and `accepted_shards`.
Each accepted entry contains `v3`, `bin_sha256`, and `meta_sha256`.
The production fine-tune uses the frozen partial manifest `data/selfplay/central-weakness-rollouts-16m/frozen_partial_manifest.json`. It lists only the 85 accepted local shards. The live campaign manifest remains incomplete.
Preparation uses `training/prepare_stored_replay_all.py`. It consumes all
accepted records through a disk-backed aggregation store, filters invalid search
targets, and averages duplicate states. It assigns whole games to a split and
removes states that occur in both splits. Flat Colab shards receive distinct
game namespaces. Replay output, the mixer, and the trainer use memory maps.

The mixer assigns all new local and Colab data to one pool with 75% effective
sample-weight mass. It assigns the frozen historical recipe to the remaining
25%. It preserves historical ratios after the previous weight cap of 30.
Global normalization prevents high historical weights from exceeding 25%.
New source shares follow eligible counts. No source receives an equal-account
quota, and no new population has a sampling cap. Invalid records and split
conflicts remain explicit exclusions in manifests.

The mixer writes `fields/*.npy`, per-source mass, and `dataset.manifest.json`.
It validates hashes, filters cross-split states, and averages duplicate targets
by normalized weight. It preserves the train and validation assignments.
Changed inputs require a new output directory.

Use `--no-dry-run` to train prepared candidates sequentially. Training uses
120 epochs, cosine head LR `1e-5` to `1e-7`, trunk scale 0.05, QAT, horizontal
mirror augmentation, and policy and value loss coefficients of 1.0.
The new replay value target mixes searched MCAB root and terminal result
equally, with gamma 1. Historical targets remain frozen.

Each run writes `initial.pt`, `resume.pt` each epoch, `best.pt`, and retained
`epoch_NNNN.pt` files every ten epochs. Resume restores the optimizer, RNGs,
history, and schedule position. Exports use the best validation state,
including epoch zero when later states do not improve validation.
Reports include source and bucket metrics. Native builds use each architecture
manifest. The orchestrator also compares initial and best networks against
unchanged historical validation targets in `frozen_retention.json`.
Promotion requires the documented paired arena gates.

### Current production-only run

#### New-data snapshot

The 2026-10-02 freeze contains 10,092,765 raw records. Colab contributes 8,704,079 records from 671 accepted shards. The partial local campaign contributes 1,388,686 records from 85 accepted shards. Exclude the unaccepted `c4_shard_0229` pair. Every accepted binary and metadata hash matches.

The Colab manifests mark 943,318 records with `META_ROOT_MISSING`. These records also have zero policy visits. Replay excludes them. The remaining 7,760,761 Colab records have valid roots and nonzero policy visits. The combined snapshot has 9,149,447 valid search records before duplicate aggregation and split-conflict removal.

#### Historical source

The historical dataset contains 15,637,120 samples and 374,177 game groups. The training split contains 12,893,910 samples. The validation split contains 2,743,210 samples. No group crosses the split. Preserve the historical targets and weight ratios.

#### Training and evaluation

Run `training/run_production_finetune.py` to prepare the accepted sources and fine-tune one `production_bucketed512` candidate. Assign 75% of effective sample-weight mass to new data. Assign 25% to historical data. Train for 120 epochs with a cosine head learning rate from `1e-5` to `1e-7` and a trunk scale of 0.05.

Arena evaluation is deferred. Do not start arena matches automatically after training. Resume only after explicit user authorization. When resumed, run 200 pairs (400 games) against Claustrophobia and 200 pairs (400 games) against the frozen current main, at 200 ms per move. Use one arena worker by default; increase to at most four only if memory permits, and keep total search use at or below 10 cores.

At 2026-10-02 13:53 replay preparation and mixing were complete. The CUDA training process was active in epoch one. No full training epoch had completed. Arena evaluation and self-play were stopped.

## Folder map

| Folder | Contents | Operational status |
| --- | --- | --- |
| `build/` | Portable build scripts and arena build helper | Canonical. Use the platform script, not compiler commands copied from a past run. |
| `tools/` | External-bot setup, arena wrappers, benchmark runners and diagnostics | `setup_bots.py` and `run_benchmark.py` are public. Other files are specialist tools. |
| `tools/selfplay/` | C++ self-play program and the ordinary Python launcher | Canonical for simple self-play. |
| `tools/teacher/` | Reusable pipeline stages: sampling, relabeling, blending, dataset assembly, seed scheduling | Internal stages. They are valid when called by a documented workflow; they are not separate campaigns. |
| `training/` | Canonical data contract, replay, teaching, trainer, quantizer, parity, campaign | Public entry points are listed above. `student_model.py` and `read_selfplay.py` are libraries. |
| `training/teachers/` | Teacher adapters and target construction | Library/internal workers; invoke through `run_teaching.py` or the documented search-relabel stages. |
| `tests/` | Regression, parity, and controller tests | Not production runners. |
| `benchmarks/` | Reproduction positions and diagnostic C++ benchmarks | Not training data. |

## Cleanup policy

Do not add a script for a one-time command when an existing runner has the
needed option. Keep a new script only when it is a reusable stage with a clear
input/output contract, a top-of-file configuration block, a test where it
changes data, and a row in this catalog. Delete or archive a duplicate only
after its replacement reproduces its manifest and output contract.

`docs/datasets.md` is deliberately local-only. It may describe the local data
inventory, but it must never be added to Git or used as a required public
reference.
