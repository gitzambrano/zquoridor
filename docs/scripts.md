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
`training/run_campaign.py`, `tools/teacher/run_four_million_selfplay.py`, and
`tools/teacher/run_weakness_selfplay.py`. `tools/selfplay/run_selfplay.py`
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
| Ordinary self-play | `tools/selfplay/run_selfplay.py` | architecture-matched self-play executable | `data/selfplay_canonical_v3/` | The simple, configurable shard generator. |
| Balanced, resumable self-play | `tools/teacher/run_four_million_selfplay.py` | central/broad seed schedules and matching executable/NNUE | a V3 corpus with metadata sidecars | Active controller. It deduplicates states, resumes safely, and exposes its configuration block at the top of the file. |
| Build a seed schedule | `tools/teacher/build_selfplay_seed_schedule.py` | JSONL snapshots/openings | central and broad JSONL schedules | Internal stage used by the balanced controller. |
| Weakness self-play | `tools/teacher/run_weakness_selfplay.py` | any compatible position snapshot | weakness V3 shards | Resumable generic runner; configure paths and target counts at the top. |
| Convert and audit old self-play | `training/migrate_selfplay_v3.py`, then `training/audit_selfplay.py` | old shards | canonical V3 shards and manifest | Migration preserves original files. |
| Build replay/teaching data | `training/prepare_replay.py` or `training/run_teaching.py` | V3 shards or JSONL trajectories | one `dataset.npz` plus manifest | The dataset contains mover-relative policy and value targets. |
| Train one network | `training/run_experiment.py` | one `dataset.npz` | float/int8 weights, manifest, matching executable | QAT and resume are configured here. |
| Train an architecture matrix | `training/run_architecture_matrix.py` | one dataset and matrix settings | one experiment directory per candidate | Auxiliary batch wrapper; use only when its matrix matches the experiment. |
| Complete campaign | `training/run_campaign.py` | configuration, data and optional teaching | data, candidates and arenas | Generic orchestration for a reproducible experiment. |
| Resilient remote/Colab self-play | `tools/selfplay/run_colab_worker.py` | seed openings, executable, NNUE | chunked V3 shards and metadata in Google Drive | Auto-resumes from existing Drive shards, unique worker seeds. |
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
