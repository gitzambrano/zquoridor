# Script Catalog and Execution Contracts

This document is the reference catalog for operational scripts in Zquoridor.
Run all commands from the repository root. Datasets, checkpoints, logs, and benchmark results are local artifacts outside version control.

Do not place roadmap targets or training plans in this document. Record roadmap targets and measurements in `docs/plan.md`.

## Configuration convention

Every public runner adheres to the following contract:
1. The script defines a top-level dictionary named `CONFIG` containing default settings.
2. Executing the script without arguments runs the default workflow.
3. Command-line flags override specific `CONFIG` keys for a single execution.
4. Utility libraries, internal stages, and test suites are not public runners.

## Public training entry point

`training/train_nnue.py` owns architecture training and retains the legacy
model-soup workflow. The default architecture command writes an inventory and
a resolved plan. It does not start training.

| Command | Effect |
| --- | --- |
| `--cli-command plan` | Inspect canonical corpora, recent worker coverage, and all resolved model configurations. |
| `--cli-command initialize` | Freeze reference weights and instantiate all selected models without training. |
| `--cli-command prepare` | Validate recent worker coverage, prepare JSONL targets and trajectories, then mix the common dataset. |
| `--cli-command train` | Prepare or verify the mixture, then train selected models sequentially. |
| `--workflow legacy` | Execute the retained three-arm and soup coordinator. An explicit `--mode` also selects this workflow. |

Defaults reside in the top-level `CONFIG["campaign"]` dictionary. CLI flags
override individual settings. Lists and dictionaries accept JSON. Useful
flags include `--suite`, `--experiments`, `--downloads`, `--data-roots`,
`--sources-file`, `--pool-fractions`, `--output`, `--epochs`, `--batch-size`,
`--device`, `--cpu-threads`, and `--resume` or `--no-resume`.
`--training-overrides` accepts a JSON dictionary of other optimizer and loss
settings from `run_experiment.CONFIG`. These overrides apply after each model
recipe. They cannot change dataset paths, model identity, or runtime builds.

`--cloud inventory` lists training artifacts through existing Colab runtime
connections. `--cloud download` also copies stable file versions. Both use
the configured CLI interpreter. They do not execute notebook cells, allocate
VMs, or recover jobs. The snapshot helper resides in `colab/cli/data_snapshot.py`.
`--worker-ids` defaults to `[1,2,3,4,5]`. `--require-five-workers` requires paired
recent position and label files from each selected worker. The explicit
`--no-require-five-workers` override permits a diagnostic partial preparation.
It does not certify complete fleet coverage.

Internal modules have separate roles:

- `architecture_models.py` defines models, state-only encoders, and portable exports.
- `architecture_data.py` inventories canonical corpora and prepares rich labels.
- `architecture_campaign.py` resolves recipes and invokes shared preparation and training.
- `architecture_recipes.py` caches reference surprise and preserves independent head mass.
- `batch_capacity.py` measures CUDA training memory and selects a shared batch per architecture.
- `mix_experimental_datasets.py` deduplicates states and preserves separate evidence per head.
- `run_experiment.py` owns optimization, QAT, checkpoint recovery, and reports.

Outputs include the resolved plan, source inventory, common dataset manifest,
frozen reference, initial models, best and resume checkpoints, model manifests,
and training reports. Unsupported native architectures export PyTorch only.
The command never starts benchmarks, tournaments, or promotion. Model details
belong to the experiment README. Data semantics belong to `training-data.md`.
Collection status and next actions belong to `plan.md`.

---

## Script index and campaign scripts

The script index at the end of this document lists every tracked script.
`python tools/script_catalog.py` regenerates the index from the module docstrings.
`tests/test_script_catalog.py` fails if the index is out of date.

`tools/campaigns/` holds the scripts that built one specific dataset.
`tools/campaigns/README.md` lists each script and its output.

## 1. Evaluation and Benchmark Runners

### `tools/measure_temporal_strength.py`
- **Purpose**: Consolidates the requested H2H match and runs a matched external strength screen for the temporal candidate and production baseline.
- **Inputs**: The H2H game records and summary, the selected opening books, production executable, candidate weights, and baseline weights.
- **Outputs**: Per-participant external games, paired score-change intervals, category summaries, a consolidated strength report, status, and a report ZIP.
- **Default settings**: 100 H2H pairs with 50 Normal and 50 Center Rush pairs. The external screen uses 50 pairs per participant and opponent, with 25 pairs per book. All participants receive 200 ms per move. Pondering is disabled. The external adapters use corrected deadline clocks and pinned source revisions.
- **Compute**: Three CPU workers. External builds and games start after the H2H completes.
- **Comparison**: The bootstrap resamples identical opening pairs jointly for candidate and baseline. Reported changes exclude unmatched pairs. The runner preserves production weights.
- **Key CLI flag**: `--output`.

### `training/prepare_temporal_search.py`
- **Purpose**: Builds value, policy, and auxiliary policy targets from downloaded self-play shards and complete search games.
- **Inputs**: A JSON or JSONL download ledger with original paths, source kinds, and materialized file paths. Stored shards require aligned V3 binary and metadata files. Complete game archives require root visits, values, histories, and outcomes. Isolated synthetic positions require their aligned target NPZ file.
- **Outputs**: An identity-checked SQLite cache and a manifest-verified memory-map dataset.
- **Default settings**: Value horizons of 0, 4, 8, and 16 plies with geometric weights. The terminal outcome contributes 10%. Future values use the current player's perspective. Unavailable horizons are omitted and available weights are normalized. Positions beyond the game endpoint use the known terminal outcome.
- **Teacher mixture**: Claustrophobia has three times the root contribution of Zquoridor at the same position before search-budget and visit-count adjustments. The auxiliary policy uses equal teacher factors before those quality adjustments. Policies remain tied to the current position.
- **Sample weights**: Source confidence supplies the base weight. Bounded policy KL surprise and bounded value disagreement supply separate head multipliers. Normalization uses training rows only.
- **Validation**: Complete game groups determine the split. Canonical states that occur in both splits are excluded. The native encoder checks resources, wall topology, paths, and terminal status.
- **Key CLI flags**: `--downloads`, `--output`.

### Per-head targets in `training/run_experiment.py`
- **Purpose**: Trains the main policy, value heads, and optional auxiliary policy with independent supervision.
- **Optional dataset arrays**: `aux_policy_target`, `value_sample_weight`, `policy_sample_weight`, and `aux_sample_weight`. Existing datasets preserve their original behavior.
- **Auxiliary policy**: Horizontal augmentation mirrors both policy targets. Temperature softening applies to the auxiliary target. Export omits the auxiliary head.
- **Schedule**: `--epochs 100 --lr 1e-5 --min-lr 1e-7 --warmup-epochs 0 --schedule cosine --patience 0` executes all 100 epochs with the requested endpoint rates.

### `training/run_temporal_campaign.py`
- **Purpose**: Waits for the selected download ledger, prepares the temporal dataset, trains 100 epochs, and verifies native exports.
- **Inputs**: The Drive inventory, completed download ledger, and production 4.0 weights.
- **Outputs**: Campaign status, logs, epoch checkpoints, training report, float and int8 weights, native candidate, parity report, and a ZIP package.
- **Default settings**: Production 858-input network, 512 hidden units, full training scope, QAT, auxiliary policy temperature 2, auxiliary loss weight 0.15, cosine learning-rate decay from `1e-5` to `1e-7`, and 100 epochs without early stopping. The accumulator learning rate is one tenth of the base rate. The runner uses three CPU threads and batch size 2048.
- **Recovery**: `resume.pt` preserves the model, optimizer, schedule position, and random states. The preparer rejects an incompatible cache identity. The runner does not promote production weights.
- **Key CLI flags**: `--output`, `--dataset`.

### `training/finetune_external_value.py`
- **Purpose**: Runs one short value-only fine-tune from an external teacher archive. The runner preserves the production weights.
- **Inputs**: One ZIP file with a gzip JSONL member. Each row supplies a position state, side to move, teacher value, source, and optional game ID. The native `bin/teacher_encode_state` encoder must exist.
- **Outputs**: A canonical dataset, initial predictions, fixed disagreement weights, epoch exports, a selected checkpoint, and a report in `experimental/experiments/external-value-short-20261013/`.
- **Default settings**: Three epochs, learning rate `1e-5`, batch size 2048, three CPU threads, and quantization-aware value heads. The accumulator and policy remain fixed. The sample weight is `1 + 3 * abs(initial_student_probability - teacher_probability)`, normalized on training rows.
- **Key CLI flags**: `--archive`, `--output`, `--init`, `--epochs`, `--lr`, `--min-lr`, `--weight-reference`.
- **Continuation**: Set `--init` to a previous float candidate. Set `--min-lr` to enable linear decay from `--lr` on epoch 1 to the final rate on the last epoch. Set `--weight-reference` to preserve disagreement weights from an earlier network.
- **Validation**: The runner reserves approximately 5% by connected game groups and repeated states. Identical state rows receive their mean teacher value. Rows without game IDs use state groups. The runner selects the epoch with the lowest validation BCE. No game-strength claim follows from this metric.

### `tools/run_complete_promotion_battery.py`
- **Purpose**: Runs the full evaluation battery for candidate models against the frozen production champion, Claustrophobia, and Titanium. Measures win rates, Elo with 95% bootstrap confidence intervals, average search depth, and nodes per second (NPS).
- **Inputs**:
  - `--candidate-exe`: Path to candidate executable.
  - `--candidate-nnue`: Path to quantized int8 candidate weights.
  - `--baseline-exe`: Path to baseline executable.
  - `--baseline-nnue`: Path to baseline int8 weights.
  - `--normal-openings`: Path to normal opening book (`tools/external/openings_normal_screen_100.jsonl`).
  - `--centerrush-openings`: Path to Center Rush book (`tools/external/openings_center_rush_sound_5k.jsonl`).
- **Outputs**: Game transcripts in `results/benchmarks/<suite>/` and aggregate summary in `promotion_summary.json`.
- **Default settings**: Fixed 200 ms per move, 50 pairs (100 games) per sub-suite, 4 workers.

### `tools/run_benchmark.py`
- **Purpose**: Runs reproducible local strength benchmarks against external bots (`titanium`, `claustrophobia`).
- **Inputs**: Executable path, int8 NNUE weights, opening book JSONL file.
- **Outputs**: Match records in `results/benchmarks/` with paired Elo estimates.
- **Key CLI flags**: `--opponents`, `--pairs`, `--workers`, `--zq-move-time-ms`, `--zq-executable`, `--nnue`.

### `tools/match_finalists.py`
- **Purpose**: Runs a direct head-to-head match between two Zquoridor NNUE candidates.
- **Inputs**: Executable and weight paths for engine 1 and engine 2, opening book JSONL.
- **Outputs**: Game records in `results/benchmarks/` and summary JSON file.
- **Key CLI flags**: `--engine1-executable`, `--engine1-nnue`, `--engine2-executable`, `--engine2-nnue`, `--pairs`, `--move-time-ms`.

### `tools/run_full_candidate_suite.py`
- **Purpose**: Runs four-phase external battery versus Claustrophobia and Titanium across normal and Center Rush opening books.
- **Inputs**: Candidate executable and int8 weights.
- **Outputs**: Sub-suite directories and `battery_summary.json`.

### `tools/run_clock_smoke.py`
- **Purpose**: Verifies time management, time increments, and pondering safety between two UCI engines.
- **Inputs**: Two UCI executables and opening book.
- **Outputs**: Clock stability report and game termination checks.

### `tools/mine_weaknesses.py`
- **Purpose**: Mines empirical opening weaknesses for Candidate (`Soup_Tri_Contact`) and Main 3.01 (`Soup_Tri_Equal`) networks from benchmark transcripts against external opponents (`claustrophobia`, `titanium`).
- **Inputs**: Benchmark transcripts under `results/benchmarks/`, opening catalogs (`tools/external/openings_normal_screen_100.jsonl`, `tools/external/openings_center_rush_sound_5k.jsonl`).
- **Outputs**: Curated weak opening banks in `tools/external/weak_openings_mined.json` and `tools/external/weak_openings_mined.jsonl`.
- **Key CLI flags**: `--output`, `--score-threshold` (default 0.40), `--losses-threshold` (default 2).

### `tools/setup_bots.py`
- **Purpose**: Clones, compiles, and configures external benchmark opponents (`titanium`, `claustrophobia`).
- **Inputs**: Git repositories and pre-trained checkpoints.
- **Outputs**: Binaries under `external_bots/`.

### `tools/build_claustrophobia_dynamic_bridge.py`
- **Purpose**: Builds an isolated Claustrophobia bridge that searches until a monotonic deadline and reports the completed root visits.
- **Inputs**: The pinned Claustrophobia checkout and local IPC evaluator source. The checkout must match the pinned revision.
- **Outputs**: A content-addressed bridge build under `results/benchmarks/claustrophobia_dynamic_bridge/`.
- **Default settings**: `shared_root` is the repository root. `output` is `results/benchmarks/claustrophobia_dynamic_bridge/`.
- **Key CLI flags**: `--shared-root`, `--output`.

### `tools/build_titanium_clock_bridge.py`
- **Purpose**: Builds an isolated Titanium executable that accepts remaining time, increment, and opponent time through its UCI clock protocol.
- **Inputs**: The pinned Titanium checkout. The checkout must match the pinned revision.
- **Outputs**: A content-addressed executable under `results/benchmarks/titanium_clock_bridge/`.
- **Default settings**: `shared_root` is the repository root. `output` is `results/benchmarks/titanium_clock_bridge/`.
- **Key CLI flags**: `--shared-root`, `--output`, `--build-jobs` (default `1`).
- **Notes**: The builder uses an operating system process lock in the output cache.

### Clock mode in `tools/run_benchmark.py`
- **Purpose**: Runs the existing paired benchmark with a native game clock instead of a fixed move time.
- **Inputs**: The selected opening book, Zquoridor executable and weights, and the pinned external bot checkouts.
- **Outputs**: The usual benchmark game records and summary under the selected output directory.
- **Default settings**: `clock_initial_ms` is `0`, which selects fixed move time. `clock_increment_ms` is `2000`.
- **Key CLI flags**: `--clock-initial-ms`, `--clock-increment-ms`. Set a positive initial clock to enable clock mode. For example, `--clock-initial-ms 180000 --clock-increment-ms 2000` selects a 3+2 clock.
- **Notes**: Clock mode uses the native deadline-aware Claustrophobia bridge and increment-aware Titanium bridge. The benchmark setup builds these isolated tools when required.

### `tools/run_clock_protocol_validation.py`
- **Purpose**: Checks the deadline-aware Claustrophobia bridge and native clock requests for the pinned external engines.
- **Inputs**: The pinned bot checkouts and Claustrophobia checkpoint. A Zquoridor executable and weights are optional.
- **Outputs**: A JSON validation report at `results/benchmarks/clock_protocol_validation.json`.
- **Default settings**: CPU device, one worker, 80, 200, and 333 ms move budgets, and a 180000 ms initial clock with a 2000 ms increment.
- **Key CLI flags**: `--shared-root`, `--output`, `--checkpoint`, `--device`, `--workers`, `--movetimes-ms`, `--white-ms`, `--black-ms`, `--increment-ms`, `--zq-executable`, `--weights`.

### `tools/run_search_games.py`
- **Purpose**: Collects complete games and the actual Claustrophobia root visit policy and value from each Claustrophobia move.
- **Inputs**: Opening books in JSON or JSONL format, the Zquoridor NNUE weights, and the pinned Claustrophobia checkpoint. Match mode also uses the local Zquoridor engine.
- **Outputs**: A resumable per-game ledger under `games/`, plus `games.jsonl`, `positions.jsonl`, `labels.jsonl`, `teacher_targets.npz`, a summary, and manifests in the output directory.
- **Default settings**: Match mode, 1500 opening pairs, one worker, 400 ms per move through searched ply 14 after the opening, and a linear taper to 50 ms at searched ply 80. The default books are `normal` (`openings_irregular_bank.jsonl`), `center_rush` (`openings_center_rush_sound_5k.jsonl`), and `weakness` (`weak_openings_mined.jsonl`). The output directory is `data/teaching/search_games/`. The runner resumes by default and provisions the pinned Claustrophobia checkout and checkpoint.
- **Key CLI flags**: `--mode` (`match`, `zquoridor-selfplay`, or `claustrophobia-selfplay`), `--pairs`, repeatable `--opening-book NAME=PATH` and `--opening-weight NAME=WEIGHT`, `--opening-temperature`, `--temperature-plies`, `--record-both-searches`, `--unique-openings-first`, `--batch-games`, `--compress-game-ledger`, `--export-final-run`, `--workers`, `--start-move-time-ms`, `--end-move-time-ms`, `--decay-start-ply`, `--decay-end-ply`, `--schedule-origin` (`opening` or `game`), `--output`, `--claustrophobia-device` (`cpu` or `gpu`), `--resume`, `--export-targets`, and `--dry-run`.
- **Paired search data**: `--record-both-searches` records both engines on each identical history and budget, including dense root visits, values, and raw engine responses. It preserves the best move separately from the move sampled by temperature. These extra searches increase collection time. The default temperature is zero, the default books have equal weight, and paired searches are disabled by default.
- **Notes**: The default schedule uses the opening as its origin and rounds to the nearest millisecond. Match mode records Claustrophobia targets on its turns. The `zquoridor-selfplay` mode does not export Claustrophobia targets. Only complete goal or repetition games contribute training targets. Proven solver roots and roots without child visits remain in the raw games but do not supply policy targets. A run marks interrupted games and does not retry them. The runner does not create missing opening books. Weakness rows can repeat when the requested pair count exceeds the book size. The dry run reports unique openings and repeated pairs. See `colab/SEARCH_GAMES.md` for a Google Colab recipe.

---

### `tools/arena/run_arena.py`
- **Purpose**: Runs parallel tournament games between two engine configurations or Git revisions, with optional color inversion and binary self-play dataset recording.
- **Inputs**: Engine executables, NNUE weight directories, and random or prescribed opening schedules.
- **Outputs**: Match summaries to stdout and optional training shards in `data/arena/`.
- **Default settings**: `GAMES=1000`, `TIME_MS=150`, `THREADS=14`, `RANDOM_OPENING_PLIES=4`, `INVERT_COLORS=True`, `CREATE_BIN=True`.
- **Key CLI flags**: `--games`, `--time-ms`, `--threads`, `--e1-heuristic`, `--e2-heuristic`, `--policy-order`, `--no-invert`, `--no-bin`.

### `tools/run_v3_search_screening.py`
- **Purpose**: Screens aggressive V3 search candidate configurations (LMR, LMP, and cpuct variations) in parallel against V3-main and Claustrophobia.
- **Inputs**: Production V3 NNUE weights (`data/nnue/nnue_weights_int8.bin`), compiled Zquoridor engine, and opening books.
- **Outputs**: Screening match summaries and win-rate statistics under `results/benchmarks/`.
- **Key CLI flags**: `--candidate`, `--pairs`, `--openings`, `--workers`, `--move-time-ms`, `--output`.

### `tools/run_v3_search_confirmation.py`
- **Purpose**: Executes deep confirmation gates (`main-fixed`, `main-clock`, and `external`) for finalist search candidates against Main, Claustrophobia, and Titanium.
- **Inputs**: Production V3 NNUE weights, candidate search parameters, opening books, and opponent binaries.
- **Outputs**: Gate match transcripts and statistical summary JSON files under specified output path.
- **Key CLI flags**: `--candidate`, `--gate`, `--pairs`, `--openings`, `--output`, `--workers`, `--move-time-ms`, `--base-ms`, `--increment-ms`, `--opponents`.

### `tools/run_central_finetune_evaluation.py`
- **Purpose**: Evaluates candidate fine-tuned NNUE weights against frozen production and Claustrophobia on Center Rush opening lines.
- **Inputs**: Candidate NNUE weights and architecture JSON, frozen production directory, and `openings_center_rush_sound_5k.jsonl`.
- **Outputs**: Paired match transcripts and evaluation reports in `results/benchmarks/central-finetune/`.
- **Default settings**: 200 pairs, 200 ms move clock, one worker, bootstrap iterations 20000.
- **Key CLI flags**: `--candidate-nnue`, `--candidate-architecture`, `--candidate-name`, `--pairs`, `--workers`, `--move-time-ms`, `--openings`, `--output`.

### `tools/release_410.py`
- **Purpose**: Releases and updates the 4.10 search-distilled quantized policy over the 4.03 base engine.
- **Inputs**: Release payload, base `data/nnue/nnue_weights_int8.bin`, architecture metadata, and GitHub release blobs.
- **Outputs**: Updated production `data/nnue/nnue_weights_int8.bin`, `nnue_weights.architecture.json`, and `release.json`.
- **Key environment variables**: `GH_TOKEN`, `GITHUB_REPOSITORY`.

---

## 2. Remote and Cloud Self-Play (Google Colab)

### `colab/launch.py`
- **Purpose**: Unified launcher for Google Colab workers supporting official CLI (default) and Playwright transports.
- **Inputs**: Account registry in `colab/config.py` and local credentials in `~/.config/colab-cli/`.
- **Defaults**: `CONFIG = {"backend": "cli", "worker_ids": [1, 2, 3, 4, 5], "gpu": "T4", "force_restart": False, "timeout": 300}`.
- **Key flags**: `--backend` (`cli` or `playwright`), `--worker-ids`, `--gpu`, `--force-restart`, `--headed`, and `--show-config`.
- **Outputs**: Remote VM runtime sessions executing configured self-play data generation writing to Google Drive.

### `colab/watchdog.py`
- **Purpose**: Unified continuous watchdog and keep-alive monitor for Google Colab workers across all accounts.
- **Inputs**: Registered worker definitions in `colab/config.py`.
- **Defaults**: `CONFIG = {"backend": "cli", "worker_ids": [1, 2, 3, 4, 5, 6, 7], "interval": 60.0, "stall_seconds": 900.0, "recover": True, "output": artifacts/colab/cli/monitor.json}`.
- **Key flags**: `--backend` (`cli` or `playwright`), `--worker-ids`, `--interval`, `--stall-seconds`, `--max-cycles`, `--once`, `--no-recover`, and `--show-config`.
- **Outputs**: Continuous monitoring logs, stall detection, automated session reattachment, and `monitor.json` telemetry.

### `colab/cli/`
- **Purpose**: Provides the official CLI transport for all seven registered Google accounts.
- **Setup**: Run `colab/cli/setup.ps1` to install the pinned Windows PR and Jupyter client in `artifacts/colab/cli/venv`.
- **Entry points**: Run `python -m colab.cli.accounts --attach`, `python -m colab.cli.run --worker-id N -- COMMAND`, or `colab/cli/colab.ps1 N COMMAND` with the installed interpreter.
- **Inputs**: The account registry, isolated ADC credentials outside Git, and explicit command arguments.
- **Outputs**: CLI command output and separate session metadata for each account and project.
- **Inspection**: `accounts`, `run`, `login`, `monitor`, and `recover` support `--show-config` without remote actions.

### `colab/cli/monitor.py`
- **Purpose**: Reads existing runtime state and persisted artifact metadata without browser control or kernel execution.
- **Defaults**: Seven accounts, a 60-second interval, a 900-second stall threshold, and continuous observation.
- **Outputs**: `artifacts/colab/cli/monitor.json` and account status lines.
- **Key flags**: `--worker-ids`, `--once`, `--max-cycles`, `--interval`, `--stall-seconds`, `--output`, `--recover`, and `--recovery-spec`.
- **Recovery**: An explicit policy can resume a job after process and persistent checkpoint checks. Recovery permits three attempts and a 900-second cooldown. It never terminates an active process or allocates a VM.
- **Limits**: Checkpoint metadata does not prove file integrity. Authentication and runtime loss require operator attention or a switch of transport.

### `colab/playwright/monitor.py`
- **Purpose**: Reads connection state and progress from notebook pages without reconnect or execution actions.
- **Inputs**: Seven account profiles and fixed or dynamic browser debugging ports.
- **Outputs**: `artifacts/colab/playwright/monitor.json` and account status lines.
- **Key flags**: `--worker-ids`, `--once`, `--max-cycles`, `--interval`, `--stall-seconds`, `--open-profiles`, and `--output`.
- **Notes**: Both read-only transports can observe the same accounts. Managed execution controllers use account locks shared across projects. See `colab/README.md` for transport selection.

### `colab/playwright/switch_search_games.py`
- **Purpose**: Completes the current self-play shard on each account, then launches paired search collection through its notebook. One locked controller monitors all selected accounts.
- **Inputs**: Authenticated account profiles in `colab/config.py`, a pinned Main revision, and the collection settings in `colab/search_games_profile.py`.
- **Outputs**: Controller status in `artifacts/colab/search_games_handover/status.json` and search game ledgers in each account's Drive directory under `zquoridor_data/claustro_search_games_v1/worker_X`.
- **Default settings**: Five accounts, one game per account, 50000 pairs per account, 250 games per batch, 70% Center Rush, 20% Normal, 10% weakness, initial temperature 1.0 for 14 plies, and both engines' searches recorded. Each account uses a distinct seed and a frozen artifact cache.
- **Key CLI flags**: `--worker-ids`, `--revision`, `--check-interval-seconds`, `--headless`, and `--output`.
- **Notes**: Stop the old Zquoridor watcher before this controller. The controller requires the Colab runtime terminal. It preserves browser profile locks and does not bypass authentication or runtime limits. It attempts browser recovery three times per account and reuses an active handover helper. A persistent failure blocks collection on that account.

### `colab/playwright/watch_search_games.py`
- **Purpose**: Provides the automatic Python watcher entry point for paired search collection. It completes the initial shard handover and reconnects or resumes the frozen workflow after a notebook disconnect.
- **Inputs and defaults**: Uses `switch_search_games.CONFIG` and the collection profile. The native runner manages every batch and writes it to Drive.
- **Outputs**: The shared controller status file and remote Drive batch artifacts.
- **Driver recovery**: A closed browser driver permits three retries with a 60-second delay. A separate watcher lock prevents duplicate controllers during recovery.
- **Key CLI flags**: The switch flags, plus `--auto-resume`, `--retry-cooldown-seconds`, and `--max-resume-attempts`. Automatic resume permits three attempts with a 900-second cooldown. Identity mismatches and authentication requirements block automatic re-execution.

### `colab/remote/finish_current_shard.py`
- **Purpose**: Completes one native self-play shard before a workflow change. The Linux helper pauses only its Python launcher. It leaves the native child active, verifies aligned binary and metadata files after exit, then terminates the launcher.
- **Inputs**: Linux process identities and the worker ID.
- **Outputs**: An atomic JSON status, artifact hashes, and terminal status markers. Errors resume the launcher.
- **Default settings**: Worker 1, a two-second check interval, and `/content/zquoridor_search_games/handover_worker_1.json`.
- **Key CLI flags**: `--worker-id`, `--status-path`, and `--poll-seconds`.

### `colab/playwright/launch_workers.py`
- **Purpose**: Bootstraps Google Colab self-play workers headlessly via Playwright. Injects the resilient bootloader cell, connects runtimes, and triggers execution.
- **Inputs**: Authenticated user profiles under `C:\Projetos\TikTok\profiles\` with stored cookies.
- **Outputs**: Active self-play sessions writing to Google Drive and confirmation screenshots in `artifacts/colab/`.
- **Key CLI flags**: `--worker-ids` (e.g. `1 2 3 4 5`), `--headless`, `--force-restart`.

### `colab/playwright/watchdog_workers.py`
- **Purpose**: Continuous keep-alive monitor for active Google Colab workers. Simulates human idle interactions to prevent disconnects and auto-reconnects dropped runtimes.
- **Inputs**: Worker registry in `colab/config.py`.
- **Outputs**: Health checks, periodic status screenshots, and automatic cell re-execution.
- **Key CLI flags**: `--worker-ids`, `--check-interval-seconds`, `--headless`, `--target-delta-positions`.

### `colab/playwright/report_workers.py`
- **Purpose**: Audits live Colab notebook DOM state, current shard indices, and games completed across all workers.
- **Inputs**: Active browser contexts or profile sessions.
- **Outputs**: Markdown summary table and per-worker status screenshots.

### `colab/playwright/login_worker.py`
- **Purpose**: Interactive login helper to authenticate Google accounts and export permanent session cookies to `cookies.json`.
- **Inputs**: `--worker-id` (single worker) or `--worker-ids` (sequential list).
- **Outputs**: Updated `cookies.json` in worker profile directory.

### `colab/playwright/dispatch_rich_targets.py`
- **Purpose**: Autonomous orchestrator that deploys Wall-Drop generation and Gumbel AlphaZero Sequential Halving across all 5 remote Google Colab workers. Handles Google Drive mounts, injects native C++ compile steps, and monitors execution health.
- **Inputs**: Registered worker definitions in `colab/config.py`.
- **Outputs**: Remote execution jobs producing `positions.jsonl`, `targets.jsonl`, and `targets.npz` in Google Drive.

### `colab/playwright/orchestrate_v4.py`
- **Purpose**: Fleet controller for continuous self-play and Claustrophobia search collection. Detects GPU quota limits, switches to CPU self-play, and keeps browser connections alive.
- **Inputs**: Worker configuration dictionary and runtime profiles.
- **Outputs**: Execution monitor log and active game generation on Google Drive.

### `colab/playwright/diagnostics/solve_audio.py`
- **Purpose**: Solves Google reCAPTCHA v2 audio challenges using the OpenAI Whisper tiny model. Downloads challenge MP3 payloads, resamples to 16 kHz, transcribes spoken text, and submits verification.
- **Inputs**: CDP port of the target browser instance (defaults to 9005).
- **Outputs**: Verified reCAPTCHA session.

### `colab/playwright/diagnostics/fix_fleet_popups.py`
- **Purpose**: Scans all 5 Colab workers for pending Google Drive OAuth consent tabs and reCAPTCHA challenge dialogs, approves permissions, and ensures Cell 0 execution.
- **Inputs**: Running CDP worker endpoints on ports 9001 to 9005.
- **Outputs**: Approved permissions and triggered notebook cells.

### `colab/remote/run_colab_worker.py`
- **Purpose**: Standalone chunked self-play generator designed to execute on Linux/Colab virtual machines.
- **Inputs**: Opening bank JSONL, self-play binary (`bin/selfplay`), int8 weights.
- **Outputs**: Aligned V3 shards (`c{id}_shard_XXXX.bin`) and metadata sidecars in Google Drive.
- **Key CLI flags**: `--worker-id`, `--drive-dir`, `--total-games`, `--chunk-games`, `--time-ms`, `--positions`, `--mc-temp-obvious`, `--playout-cap`, `--cheap-time-ms`, `--cheap-time-end-ms`, `--cheap-time-decay-plies`, `--full-search-opening-plies`, `--full-search-prob`.

---

## 3. Local Self-Play and Rollouts

### `tools/selfplay/run_selfplay.py`
- **Purpose**: Basic local self-play shard generator.
- **Inputs**: Compiled self-play binary, opening positions, NNUE weights.
- **Outputs**: V3 binary shards in `data/selfplay/`.

### `tools/teacher/run_four_million_selfplay.py`
- **Purpose**: Resumable multi-threaded self-play controller. Manages state deduplication and opening quotas across central and broad families.
- **Inputs**: Opening schedules, self-play binary, NNUE weights.
- **Outputs**: Durable SQLite state store (`states.sqlite`) and deduplicated shards.

### `tools/teacher/run_weakness_selfplay.py`
- **Purpose**: Self-play generator focused on tactical weakness recovery positions.
- **Inputs**: Weakness position bank JSONL, matching engine.
- **Outputs**: Dedicated weakness V3 shards.

### `tools/teacher/run_central_weakness_rollouts.py`
- **Purpose**: Executes 8 to 10 ply Center Rush rollouts using wide, balanced, and sharp exploration profiles.
- **Inputs**: 50,000 root opening catalog, production and candidate self-play executables.
- **Outputs**: Source-separated V3 shards and aligned metadata records.

### `tools/teacher/generate_weakness_variations.cpp`
- **Purpose**: High-speed C++ generator that traverses mined weak openings ply-by-ply, evaluates all legal moves with quantized NNUE policy and win probability heads, and branches variations on the 2nd, 3rd, and 4th best moves to unbias engine training.
- **Inputs**: Quantized NNUE weights (`data/nnue/nnue_weights_int8.bin`), mined weakness seeds (`tools/external/weak_openings_mined.jsonl`).
- **Outputs**: Formatted JSON array (`tools/external/openings_weakness_variations.json`) and line-delimited JSONL (`tools/external/openings_weakness_variations.jsonl`) compatible with Colab self-play workers.
- **Key CLI flags**: `--weights`, `--seeds`, `--out-json`, `--out-jsonl`.

### `tools/generate_wall_drops.py`
- **Purpose**: Generates synthetically perturbed maze positions by placing random legal walls, computes breadth-first shortest path distances, and extracts dense 209-action policy distributions and root values via engine search.
- **Inputs**: Optional Zquoridor UCI executable path (`--zq-executable`), wall counts (`--min-walls`, `--max-walls`), pawn walk plies (`--pawn-walk-plies`), search move time (`--move-time-ms`), random seed (`--seed`), and split name (`--split`).
- **Outputs**: Architecture-neutral JSONL positions (`positions.jsonl`), teacher targets (`targets.jsonl`), compressed NumPy arrays (`targets.npz`), and execution manifest (`manifest.json`) under the output directory.
- **Key CLI flags**: `--count`, `--min-walls`, `--max-walls`, `--pawn-walk-plies`, `--move-time-ms`, `--zq-executable`, `--output`, `--teacher`, `--seed`, `--split`, `--no-export-npz`.

### `tools/generate_gumbel_sims.py`
- **Purpose**: Executes Gumbel AlphaZero planning and Sequential Halving to generate high-density improved policy targets and expected values with focused simulation budgets.
- **Inputs**: Optional source position corpus (`--source-positions`), Zquoridor UCI executable (`--zq-executable`), candidate pool size (`--top-k`), value scaling factor (`--c-scale`), search time (`--move-time-ms`), random seed (`--seed`), and split name (`--split`).
- **Outputs**: Refined JSONL positions (`positions.jsonl`), Gumbel teacher targets (`targets.jsonl`), compressed NumPy arrays (`targets.npz`), and execution manifest (`manifest.json`) under the output directory.
- **Key CLI flags**: `--count`, `--top-k`, `--c-scale`, `--move-time-ms`, `--zq-executable`, `--source-positions`, `--output`, `--teacher`, `--seed`, `--split`, `--no-export-npz`.

### `tools/rebuild_npz_from_jsonl.py`
- **Purpose**: Synchronizes, validates, and rebuilds dense NumPy target archives (`targets.npz`) directly from canonical `positions.jsonl` and `targets.jsonl` files. Preserves existing archives with timestamped backups.
- **Inputs**: Dataset directory containing `positions.jsonl` and `targets.jsonl`.
- **Outputs**: Validated, non-zero `targets.npz`, updated `manifest.json`, and timestamped backup `targets.npz.bak_<timestamp>`.
- **Key CLI flags**: `--dir`, `--no-backup`, `--verify-only`.

### `tools/clean_repository.py`
- **Purpose**: Safely removes ephemeral Python bytecode caches, temporary dataset memmaps (`.input_cache`), aborted self-play staging files, and transient recovery dumps without touching tracked files, canonical datasets, model weights, or private credentials.
- **Inputs**: Repository directory tree (`--root`).
- **Outputs**: Reclaimed disk space summary reported to stdout.
- **Key CLI flags**: `--dry-run`, `--verbose`, `--skip-pycache`, `--skip-input-cache`, `--skip-staging`, `--skip-recovery`, `--show-config`.

---

## 4. Dataset Preparation and Mixing

### `tools/dataset_inventory.py`
- **Purpose**: Builds the dataset inventory from the provenance records under `data/selfplay/` and `data/teaching/`.
- **Inputs**: Each dataset directory with `provenance.json`, and each dataset file `<file>` with `<file>.provenance.json`. `data/selfplay/manifest.json` supplies the historical generations.
- **Outputs**: The generated section of `docs/datasets.md` (ignored by Git) and `data/dataset_index.json` (tracked).
- **Checks**: The script reports a dataset without a valid record, a source that is absent from disk, and a historical generation that is absent from disk.
- **Key CLI flags**: `--check` (exit status 1 if a dataset has no valid record), `--no-write`, `--roots`.

### `training/dataset_provenance.py`
- **Purpose**: Defines the provenance record format `zquoridor.dataset_provenance.v1`. Provides `write`, `read`, and `validate`.
- **Required fields**: `name`, `kind`, `status`, `description`, `generator`, `sources`, and `contents`.
- **Use**: Call `dataset_provenance.write(...)` at the end of each dataset generator. `training/prepare_replay.py` calls it after each finished run.

### `training/prepare_replay.py`
- **Purpose**: Converts raw V3 self-play shards into training datasets with policy distributions and value targets.
- **Inputs**: Directory of V3 shards and metadata sidecars.
- **Outputs**: `dataset.npz`, `replay_manifest.json`, and `provenance.json` in `out_dir`.
- **Key CLI flags**: `--mode` (`direct` or `stored_search`), `--source`, `--out-dir`, `--stored-outcome-weight`, `--stored-gamma`.

### `training/prepare_stored_replay_all.py`
- **Purpose**: Consumes accepted shard manifests, aggregates duplicate states on disk, and enforces whole-game train and validation splits.
- **Inputs**: Manifests of accepted shards from local and remote campaigns.
- **Outputs**: Memory-mapped binary array directory.

### `training/mix_experimental_datasets.py`
- **Purpose**: Blends multiple prepared datasets according to specified mass ratios while preserving group separation.
- **Inputs**: Source dataset directories and target mixture configuration.
- **Outputs**: Combined memory-mapped dataset directory with SHA-256 manifest.

### `training/migrate_selfplay_v3.py`
- **Purpose**: Upgrades legacy self-play formats to the canonical V3 specification.
- **Inputs**: Legacy shard files.
- **Outputs**: V3 shards and verification manifests.

### `training/audit_selfplay.py`
- **Purpose**: Validates integrity, header structures, and metadata alignment across self-play shards.
- **Inputs**: Shard directory or manifest.
- **Outputs**: Validation report and corruption warnings.

### `training/run_teaching.py`
- **Purpose**: Generates multi-teacher training datasets combining evaluations from Zquoridor search, direct NNUE, and Claustrophobia teachers.
- **Inputs**: Optional architecture-neutral JSONL positions (`--positions`), game count, discount, and teacher weights.
- **Outputs**: Generated positions, cached teacher targets, combined `dataset.npz`, and dataset manifest.
- **Key CLI flags**: `--mode` (`direct`, `search`, `mixed`), `--games`, `--max-positions`, `--gamma`, `--outcome-weight`, `--bootstrap-weight`.

---

### `tools/campaigns/assemble_unified_clean_dataset.py`
- **Purpose**: Assembles the master production dataset (`multipath_unified_clean_15m`) by combining weakness-boosted positions with clean stored-search self-play.
- **Inputs**: Weakness master archive (`multipath-weakness-boosted-11m/dataset.npz`) and clean replay archive (`replay_clean_stored/dataset.npz`).
- **Outputs**: Final consolidated master dataset archive (`data/teaching/multipath_unified_clean_15m/dataset.npz`) and dataset manifest.
- **Default settings**: `weakness_scale=1.0`, `replay_scale=1.0`. Streams uncompressed arrays in chunks of 50,000 samples.
- **Key CLI flags**: `--weakness-master`, `--clean-replay`, `--out`, `--weakness-scale`, `--replay-scale`.

### `tools/campaigns/assemble_5tier_dataset.py`
- **Purpose**: Assembles multi-tier hierarchical training datasets from background replay, past search cases, generic search, dual crisis, and deep search tiers.
- **Inputs**: Tier directories containing state arrays and supervision targets.
- **Outputs**: Combined `dataset.npz` and build manifest using chunked array streaming.
- **Key CLI flags**: `--output`, `--tier1`, `--tier2`, `--tier3`, `--tier4`, `--tier5`.

### `tools/teacher/mine_weakness_positions.py`
- **Purpose**: Extracts critical positions from benchmark game losses, draws, swept openings, and wall-depletion crises against external opponents.
- **Inputs**: Benchmark transcripts directory containing `games.jsonl`.
- **Outputs**: Architecture-neutral `positions.jsonl` formatted in canonical `zquoridor.position.v1`.
- **Key CLI flags**: `--benchmarks-dir`, `--out`, `--max-positions`, `--val-fraction`, `--seed`.

### `tools/campaigns/calibrate_claustrophobia_weakness_weights.py`
- **Purpose**: Calibrates sample loss weights to emphasize Center Rush clashes, bottleneck corridor states, and pawn race disadvantages while keeping background anchor weights at 1.0.
- **Inputs**: Input master dataset archive (`dataset.npz`).
- **Outputs**: Calibrated dataset archive with adjusted sample weights.
- **Key CLI flags**: `--input`, `--output`, `--max-weight`.

---

## 5. Training, Quantization, and Parity

### `training/run_experiment.py`
- **Purpose**: Trains, exports, and builds one compact NNUE student. Supports cosine annealing, QAT, horizontal reflection, and custom value heads.
- **Inputs**: A teaching NPZ or a manifest-verified memory-map directory, and the initialization weights.
- **Outputs**: `student.bin` (float), `student_int8.bin` (quantized), the architecture manifest, `config.json` (training identity), the training report, and a native executable in `out_dir`.
- **Settings order**: The script applies `CONFIG`, then the `--config` recipe, then the other CLI flags.
- **Key CLI flags**: `--config`, `--architecture`, `--hidden`, `--epochs`, `--batch-size`, `--lr`, `--qat`, `--mirror-h`, `--resume`.
- **Recipe format**: `--config` accepts a flat JSON object of `CONFIG` keys, or an object with a `config` key. An unknown key stops the run.

### `training/run_contact_finetune.py`
- **Purpose**: Resilient supervisor runner for experimental contact-bucketed candidate training. Handles automatic restarts across Windows memory limits while preserving checkpoint state.
- **Inputs**: Prepared mixed dataset directory.
- **Outputs**: Exported weights, quantized int8 model, and compiled candidate executable.

### `training/run_production_finetune.py`
- **Purpose**: Executes fine-tuning for production bucketed candidates on combined historical and new rollout data.
- **Inputs**: Manifest-verified mixed dataset.
- **Outputs**: Production candidate checkpoints and native binary.

### `training/quantize_nnue.py`
- **Purpose**: Quantizes float32 weights into int8 and int16 representations matching C++ `NNUEWeightsQuant` layout.
- **Inputs**: Float32 weight file (`.bin`).
- **Outputs**: Quantized weight file (`_int8.bin`).

### `training/run_stage_a.py`
- **Purpose**: Executes Stage A of the auxiliary policy experiment. Trains A0 (control) and A1 (auxiliary softened policy) from frozen production weights, compiles native candidate binaries, and screens candidates against frozen V3.
- **Inputs**: Dataset directory (`--data`), production weights (`--init-from`), opening books (`--normal-openings`, `--centerrush-openings`).
- **Outputs**: Candidate models under `results/experiments/aux_policy_stage_a/`, screening transcripts, and aggregate comparison table.
- **Key CLI flags**: `--epochs`, `--batch-size`, `--lr`, `--aux-policy-temperature`, `--aux-policy-weight`, `--screening-pairs`, `--screening-move-time-ms`, `--run-a0`, `--run-a1`, `--run-screening`.

### `training/compute_surprise_weights.py`
- **Purpose**: Evaluates Kullback-Leibler divergence between stored search policies and predictions from a frozen baseline network. Computes bounded sample weight multipliers that prioritize tactical surprise positions while conserving total sample mass.
- **Inputs**: Dataset directory (`--data`), baseline float32 weights (`--model`), architecture and hidden width settings.
- **Outputs**: Optional modified sample weights array (`--output-weights`) and raw divergence array (`--output-kl`).
- **Key CLI flags**: `--data`, `--model`, `--architecture`, `--hidden`, `--alpha`, `--s-max`, `--batch-size`, `--device`, `--output-weights`, `--output-kl`.

### `training/model_soup.py`
- **Purpose**: Blends multiple compatible NNUE student checkpoints via convex combination of weights, verifies bounds, exports int8 weights, and compiles native candidates.
- **Inputs**: Checkpoints or float weight files (`--models <path1> <w1> <path2> <w2> ...`).
- **Outputs**: Blended float and int8 weights, architecture manifest, soup metadata, and optional native candidate executable.
- **Key CLI flags**: `--models`, `--out-dir`, `--architecture`, `--hidden`, `--no-build`.

### `training/parity_check.py`
- **Purpose**: Validates mathematical parity between Python forward passes and C++ engine evaluation across thousands of test positions.
- **Inputs**: Trained student weights and compiled C++ test harness.
- **Outputs**: Discrepancy report and maximum absolute difference metrics.


### `training/new_experiment.py`
- **Purpose**: Creates a new experiment workspace with a training recipe (`config.json`) and a `README.md` template.
- **Inputs**: The experiment name, the hypothesis, and the training overrides. The script reads the production version from `data/nnue/release.json` and the production architecture from `data/nnue/nnue_weights.architecture.json`.
- **Outputs**: `experimental/experiments/<name>/config.json` and `README.md`. Every `config.json` key must exist in `run_experiment.CONFIG`.
- **Default settings**: `architecture="multipath_phase_contact_bucketed"`, `hidden=512`, `epochs=60`, `batch_size=1024`, `lr=0.0001`.
- **Key CLI flags**: Every `CONFIG` key is a flag, for example `--name`, `--hypothesis`, `--data`, `--epochs`, `--lr`, and `--force`.
- **Next step**: `python training/run_experiment.py --config experimental/experiments/<name>/config.json`.

---

## Width comparison runner

### `training/run_local_training.py`

This portable entry point selects hidden widths 512, 768, or 1024.
`--target-mode current` uses current search values and the selected outcome
weight. `--target-mode temporal` combines current and future search values.
`--future-horizons` and `--future-weights` accept JSON lists. Their defaults
are `[0,4,8,16]` and `[0.5,0.25,0.125,0.0625]`. `--result-weight` defaults
to 0.1. `--policy-value-fraction` independently selects the policy share of value
mass; zero excludes policy. Policy is a data-source parameter, not a training mode.
All choices retain QAT, augmentation, auxiliary policy, and source confidence.
The default schedule uses 100 epochs and cosine rates from 1e-5 to 1e-7.
`--data`, `--init-from`, `--out-dir`, `--epochs`, `--lr`, `--min-lr`,
`--device`, and `--batch-size` override defaults. `--dry-run` prints the
resolved configuration. `--no-build` skips native compilation and parity.
Target changes require trajectories or aligned metadata. The runner prepares
a separate cache when existing targets do not match the requested settings.
Future value targets preserve the original mover perspective. Policy targets
remain attached to the current position. Policy-only samples retain their
provided value because the archive does not supply future trajectories.
Outputs include weights, architecture metadata, checkpoints, training
reports, and native verification when enabled. The runner resolves default
paths relative to its project directory and does not modify production.

### `training/run_policy_mix_campaign.py`

This runner waits for a completed width campaign and its paired match report.
It prepares a canonical mixed dataset, trains matched replay-only and policy
arms, verifies native exports, and compares the candidate with both controls.
`CONFIG` defines the dependency, archive, replay dataset, output paths, value
fraction, epochs, rates, and game settings. CLI flags override these values.
`status.json` records the dependency wait and subsequent phases. Outputs
include the mixed manifest, selected artifact hashes, weights, checkpoints,
original replay validation, parity reports, games, and `report.json`.
Policy availability masks exclude policy-only states from both policy losses.
The original width trainer remains unchanged. Production promotion requires
a separate operator decision.

### `training/run_width_pilot.py`

This runner trains a wider candidate and a matched control from the same
reference weights. It verifies native parity and incremental accumulators,
then runs paired games between the participants. `CONFIG` defines the
reference directory, dataset, output directory, hidden width, epochs,
learning rates, openings, and game settings. CLI flags override these values.
The output contains participant weights, checkpoints, training reports,
native executables, game transcripts, and `report.json`. `status.json`
records the active phase. The runner does not change production weights.

## Archived profile builder

`experimental/build_profile.py` compiles one registered network and search
configuration without changing production. `CONFIG` selects the profile,
registry, output directory, and compiler. The default profile is
`858_delta_dense_only`. `--profile` overrides one invocation. `--list` lists
the available configurations without a build. The script checks the weight
checksum before compilation. It writes executables to
`experimental/artifacts/bin/`. It does not run games.

<!-- script-index:begin -->

## Script index

`python tools/script_catalog.py` generates this section. Do not edit it by hand.
Edit the module docstring of the script instead.

The index lists 182 scripts: 10 campaign, 19 diagnostic, 74 module, 79 runner.

### `colab/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `compat.py` | module |  | Forward legacy entry points to one canonical implementation. |
| `config.py` | module |  | Central configuration and worker registry for Colab automation in Zquoridor. |
| `control.py` | module |  | Serialize local actions across projects for each Google account. |
| `dispatch_rich_targets.py` | module |  | Compatibility entry point. |
| `finish_current_shard.py` | module |  | Compatibility entry point. |
| `launch.py` | runner | [section](#colablaunchpy) | Launch and bootstrap Google Colab workers. |
| `monitoring.py` | module |  | Evaluate progress without changes to remote tasks. |
| `privacy_check.py` | runner |  | Check versionable files without disclosure of private values. |
| `private.py` | module |  | Load account identities from private local configuration. |
| `search_games_profile.py` | module |  | Define isolated Colab profiles for Claustrophobia search games. |
| `watchdog.py` | runner | [section](#colabwatchdogpy) | Continuous watchdog and keep-alive monitor for Google Colab workers. |

### `colab/cli/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `accounts.py` | module |  | List account access or attach existing runtimes. |
| `backend.py` | module |  | Read runtime metadata through the official Colab client. |
| `data_snapshot.py` | module |  | Internal read-only snapshots of training files on existing Colab runtimes. |
| `login.py` | module |  | Authorize isolated CLI credentials for registered Google accounts. |
| `monitor.py` | runner | [section](#colabclimonitorpy) | Monitor runtime access and persisted artifacts without browser control. |
| `recover.py` | runner |  | Resume explicitly configured jobs after a checkpoint and process check. |
| `run.py` | module |  | Run an official Colab command for one account. |
| `settings.py` | module |  | Locate the CLI environment and isolate each account. |

### `colab/playwright/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `authorize_cli.py` | module |  | Authorize CLI accounts through their existing browser profiles. |
| `browser_utils.py` | module |  | Browser session utilities and safe Playwright helpers for Colab automation. |
| `deploy_search_cell.py` | module |  | Deploy the Claustrophobia search games cell to Colab workers and launch execution. |
| `dispatch_rich_targets.py` | module | [section](#colabplaywrightdispatch_rich_targetspy) | Dispatch Wall-Drop and Gumbel simulation jobs across all 5 Google Colab workers using version 4.10. |
| `dispatch_v2_fleet.py` | module |  | Dispatch and monitor the complete zQuoridor 4.0 (858) collection fleet on Google Colab. |
| `human_actions.py` | module |  | Human motion and stealth interaction engine based on Fitts' Law and Movement Physics. |
| `launch_workers.py` | runner | [section](#colabplaywrightlaunch_workerspy) | Launch and bootstrap Google Colab self-play workers headlessly via Playwright. |
| `login_worker.py` | runner | [section](#colabplaywrightlogin_workerpy) | Interactive login helper for Google Colab worker profiles in Zquoridor. |
| `manage_runtime.py` | runner |  | Administrative runtime management for Google Colab notebooks via Playwright. |
| `monitor.py` | runner | [section](#colabplaywrightmonitorpy) | Observe notebook progress without reconnect or execution actions. |
| `orchestrate_v4.py` | module | [section](#colabplaywrightorchestrate_v4py) | Robust orchestrator and daemon monitor for zQuoridor 4.0 (Network 858) collection across all 5 workers. |
| `report_workers.py` | runner | [section](#colabplaywrightreport_workerspy) | Comprehensive status inspection, metric parser, and reporting tool for Colab workers. |
| `sessions.py` | module |  | Locate existing profile sessions without browser termination. |
| `switch_search_games.py` | runner | [section](#colabplaywrightswitch_search_gamespy) | Finish current self-play shards and launch paired search collection in Colab. |
| `watch_search_games.py` | runner | [section](#colabplaywrightwatch_search_gamespy) | Watch Colab search collection and resume the frozen workflow after disconnects. |
| `watchdog_workers.py` | runner | [section](#colabplaywrightwatchdog_workerspy) | Continuous watchdog and keep-alive monitor for Google Colab self-play workers in Zquoridor. |

### `colab/playwright/diagnostics/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `audit_fleet.py` | diagnostic |  | No module docstring. |
| `check_cdp_fleet.py` | diagnostic |  | No module docstring. |
| `check_fleet_live.py` | diagnostic |  | Check live status of all 5 Colabs and ensure execution is active. |
| `check_fleet_status.py` | diagnostic |  | No module docstring. |
| `dismiss_and_run.py` | diagnostic |  | No module docstring. |
| `fix_fleet_popups.py` | diagnostic | [section](#colabplaywrightdiagnosticsfix_fleet_popupspy) | No module docstring. |
| `fix_workers.py` | diagnostic |  | No module docstring. |
| `inspect_output_clean.py` | diagnostic |  | No module docstring. |
| `inspect_state.py` | diagnostic |  | No module docstring. |
| `inspect_workers.py` | diagnostic |  | No module docstring. |
| `poll_status.py` | diagnostic |  | Poll live progress across all 5 workers. |
| `query_progress.py` | diagnostic |  | No module docstring. |
| `restart_worker1.py` | diagnostic |  | No module docstring. |
| `run_w4.py` | diagnostic |  | No module docstring. |
| `run_worker2_cpu.py` | diagnostic |  | No module docstring. |
| `scratch_recaptcha.py` | diagnostic |  | No module docstring. |
| `solve_audio.py` | diagnostic | [section](#colabplaywrightdiagnosticssolve_audiopy) | No module docstring. |
| `solve_w4.py` | diagnostic |  | No module docstring. |
| `start_cdp_daemons.py` | diagnostic |  | Launch all 5 Colab Chrome instances headlessly with remote debugging ports 9001-9005. |

### `colab/remote/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `finish_current_shard.py` | runner | [section](#colabremotefinish_current_shardpy) | Finish one active self-play shard before the Colab handover. |
| `run_colab_anneal_and_arena.py` | runner |  | Execute 60-epoch annealing runs for Zquoridor candidates and benchmark vs baseline. |
| `run_colab_arena.py` | runner |  | Run candidate arena matches against main, Titanium, or Claustrophobia. |
| `run_colab_worker.py` | runner | [section](#colabremoterun_colab_workerpy) | Run a resilient, chunked self-play worker on Google Colab or Linux. |

### `tools/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `analyze_center_openings.py` | module |  | Analyze benchmark games breaking down central vs flank/wall openings. |
| `audit_repository.py` | runner |  | Report source organization without changes to repository files. |
| `benchmark_candidate.py` | runner |  | Compare one NNUE candidate against main and the external reference bots. |
| `build_claustrophobia_dynamic_bridge.py` | runner | [section](#toolsbuild_claustrophobia_dynamic_bridgepy) | Build an isolated deadline-aware bridge and pinned MCTS library. |
| `build_titanium_clock_bridge.py` | runner | [section](#toolsbuild_titanium_clock_bridgepy) | Build an isolated, increment-aware Titanium UCI protocol bridge. |
| `clean_repository.py` | runner | [section](#toolsclean_repositorypy) | Clean ephemeral caches, bytecode, and aborted staging files from the repository. |
| `dataset_inventory.py` | runner | [section](#toolsdataset_inventorypy) | Build the dataset inventory from the provenance records on disk. |
| `generate_gumbel_sims.py` | runner | [section](#toolsgenerate_gumbel_simspy) | Run Gumbel AlphaZero planning and Sequential Halving to generate high-density policy targets. |
| `generate_wall_drops.py` | runner | [section](#toolsgenerate_wall_dropspy) | Generate diverse Quoridor positions and dense search targets via random wall placements. |
| `match_finalists.py` | runner | [section](#toolsmatch_finalistspy) | Run a direct head-to-head match between two Zquoridor NNUE candidates. |
| `measure_temporal_strength.py` | runner | [section](#toolsmeasure_temporal_strengthpy) | Complete a paired H2H and matched external strength screen. |
| `mine_weaknesses.py` | runner | [section](#toolsmine_weaknessespy) | Mine weak openings for Main 3.01 and Candidate networks against Claustrophobia and Titanium. |
| `rebuild_npz_from_jsonl.py` | runner | [section](#toolsrebuild_npz_from_jsonlpy) | Rebuild and synchronize dense NumPy target archives from canonical JSONL records. |
| `release_410.py` | campaign | [section](#toolsrelease_410py) | Promote the 4.10 search-distilled quantized policy over the 4.03 engine. |
| `run_benchmark.py` | runner | [section](#toolsrun_benchmarkpy) | Build dependencies and run reproducible local strength benchmarks. |
| `run_central_finetune_evaluation.py` | runner | [section](#toolsrun_central_finetune_evaluationpy) | Run paired center-rush evaluation against frozen production and Claustrophobia. |
| `run_clock_protocol_validation.py` | runner | [section](#toolsrun_clock_protocol_validationpy) | Validate deadline searches and native game clocks for local engines. |
| `run_clock_smoke.py` | runner | [section](#toolsrun_clock_smokepy) | Run a small paired game-clock smoke test between two UCI engines. |
| `run_complete_promotion_battery.py` | runner | [section](#toolsrun_complete_promotion_batterypy) | Run complete promotion benchmark battery for a candidate against main and external bots. |
| `run_full_candidate_suite.py` | runner | [section](#toolsrun_full_candidate_suitepy) | Run complete evaluation battery for a candidate: 1. |
| `run_search_games.py` | runner | [section](#toolsrun_search_gamespy) | Collect complete games and the search targets that Claustrophobia used in each game. |
| `run_v3_search_confirmation.py` | campaign | [section](#toolsrun_v3_search_confirmationpy) | Confirmation gates for the strongest Zquoridor V3 search candidates. |
| `run_v3_search_screening.py` | campaign | [section](#toolsrun_v3_search_screeningpy) | Screen aggressive V3 search configurations against V3 main and Claustrophobia. |
| `script_catalog.py` | runner |  | Generate the script index of ``docs/scripts.md`` from the tracked sources. |
| `setup_bots.py` | runner | [section](#toolssetup_botspy) | Prepare pinned external bots for local jobs. |
| `validate_openings.py` | module |  | Check that every opening in a JSONL opening book is a legal move sequence. |

### `tools/arena/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `compare_arena_results.py` | runner |  | Compare candidate and baseline arena benchmark results against common opponents. |
| `run_arena.py` | module | [section](#toolsarenarun_arenapy) | Run parallel arena games between two engine configurations or Git revisions. |
| `run_colab_arena.py` | module |  | Compatibility entry point. |

### `tools/campaigns/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `assemble_200k_dataset.py` | campaign |  | Assemble massive 200k+ dataset with action-Q policy sharpening and background mix. |
| `assemble_5tier_dataset.py` | campaign | [section](#toolscampaignsassemble_5tier_datasetpy) | Assemble hierarchical multi-tier training dataset for Zquoridor. |
| `assemble_multipath_master_dataset.py` | campaign |  | Assemble the full multi-tier master dataset for the experimental multipath architecture. |
| `assemble_unified_clean_dataset.py` | campaign | [section](#toolscampaignsassemble_unified_clean_datasetpy) | Assemble the unified clean master dataset for Zquoridor NNUE training. |
| `assemble_weakness_dataset.py` | campaign |  | Assemble fine-tuning dataset for weakness recovery and center-rush tactics. |
| `build_center_rush_priority_dataset.py` | campaign |  | Build Center-Rush Focused Priority Teaching Dataset. |
| `calibrate_claustrophobia_weakness_weights.py` | campaign | [section](#toolscampaignscalibrate_claustrophobia_weakness_weightspy) | Calibrate and boost sample weights specifically targeting Claustrophobia weaknesses. |

### `tools/external/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `analyze_paired_arena.py` | module |  | Pair-aware statistics for Zquoridor/Titanium arena JSONL. |
| `bot_setup.py` | module |  | Provision pinned external bots for local teacher and benchmark jobs. |
| `build_lock.py` | module |  | Serialize builds that share one artifact cache. |
| `local_arena.py` | module |  | Run paired Quoridor games under a complete local referee. |
| `ponder_gate.py` | module |  | Paired strength gate for opponent-root pondering versus the same baseline. |
| `titanium_arena.py` | module |  | Paired external arena: Zquoridor UCI adapter vs Titanium UCI. |
| `titanium_arena_fixed.py` | module |  | Run titanium_arena with robust I/O and Titanium's native session protocol. |

### `tools/gui/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `contrast_check.py` | module |  | contrast_check.py -- accessibility gate for the GUI tokens (gui-premium.md, section 10). |

### `tools/selfplay/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `run_colab_worker.py` | module |  | Compatibility entry point. |
| `run_selfplay.py` | module | [section](#toolsselfplayrun_selfplaypy) | run_selfplay.py -- Orquestrador Python para geração de dados de self-play. |

### `tools/spsa/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `plot_spsa.py` | module |  | plot_spsa.py -- plota a convergencia de uma (ou varias) rodada(s) de tools/spsa/tune_spsa.cpp a partir do CSV de historico que o proprio tune_spsa grava a cada iteracao (--history, default spsa_history.csv; modo hybrid grava um arquivo por candidato, com sufixo "_depth{D}"). |
| `run_spsa.py` | module |  | run_spsa.py -- launcher do tuner evolutivo/genético. |

### `tools/teacher/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `blend_search_pair.py` | module |  | Blend ZQuoridor and Claustrophobia search targets into a canonical teaching dataset. |
| `blend_weakness_teachers.py` | module |  | Blend Claustrophobia and ZQuoridor search targets on mined weakness positions. |
| `build_selfplay_seed_schedule.py` | module |  | Build deterministic central and broad self-play seed schedules. |
| `extract_generic_positions.py` | module |  | Extract unique generic positions with move histories from benchmark games. |
| `extract_live_rollouts.py` | module |  | Extract completed StepRecord entries from live generate_rollouts.exe memory, validate integrity, and generate dataset.npz without stopping the process. |
| `generate_irregular_bank.py` | runner |  | Generate a validated opening bank of irregular / perimeter / non-standard openings. |
| `generate_targeted_weakness_bank.py` | runner |  | Generate a targeted opening bank focused on empirical engine weaknesses. |
| `mine_generic_critical.py` | module |  | Mine 100,000 generic critical positions from self-play for Tier 3.5 curriculum. |
| `mine_losses_and_center.py` | module |  | Mine loss positions and center pawn collision states from benchmark matches. |
| `mine_weakness_positions.py` | module | [section](#toolsteachermine_weakness_positionspy) | Mine and export weakness positions from benchmark games. |
| `prepare_massive_seeds.py` | runner |  | Prepare 3,500+ diverse weakness seeds for massive rollout generation. |
| `run_central_weakness_rollouts.py` | runner | [section](#toolsteacherrun_central_weakness_rolloutspy) | Generate a large self-play corpus focused on weak central openings. |
| `run_four_million_selfplay.py` | runner | [section](#toolsteacherrun_four_million_selfplaypy) | Generate an exact, resumable four-million-state self-play corpus. |
| `run_rollouts.py` | runner |  | Run multithreaded branching rollouts from crisis positions and build dataset.npz. |
| `run_targeted_weakness_rollouts.py` | runner |  | Run selfplay rollouts targeted on empirical engine weaknesses. |
| `run_weakness_selfplay.py` | runner | [section](#toolsteacherrun_weakness_selfplaypy) | Generate resumable 100 ms self-play shards from weakness snapshots. |
| `sample_massive_background.py` | runner |  | Sample massive background replay dataset from historical canonical V3 shards. |
| `sample_search_states.py` | module |  | Sample states from canonical V3 shards formatted as @state for MCTS and MCAB bridges. |
| `selfplay_unique_store.py` | module |  | Store exact canonical self-play states and admit validated shards. |
| `weight_weakness_seeds.py` | module |  | Build a 60/40 Claustrophobia/Titanium seed schedule with crisis weighting. |

### `training/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `architecture_campaign.py` | module |  | Internal orchestration for the shared architecture campaign. |
| `architecture_data.py` | module |  | Inventory local corpora and prepare complete rich labels for shared training. |
| `architecture_models.py` | module |  | Experimental models and state-only encoders for the architecture campaign. |
| `architecture_recipes.py` | module |  | Frozen-reference surprise caches that preserve independent head evidence. |
| `audit_selfplay.py` | module | [section](#trainingaudit_selfplaypy) | Audit local self-play shards before a NNUE training cycle. |
| `batch_capacity.py` | module |  | Select a large training batch from measured CUDA memory demand. |
| `build_search_priority_dataset.py` | runner |  | Build weighted teaching data from two expensive search teachers. |
| `build_teacher_policy.py` | module |  | Convert teacher JSONL records into a compact canonical NNUE policy dataset. |
| `build_teacher_soft.py` | module |  | Join a position corpus with soft teacher targets and canonical NNUE states. |
| `compute_surprise_weights.py` | runner | [section](#trainingcompute_surprise_weightspy) | Compute policy surprise weights using the frozen baseline network. |
| `dataset_provenance.py` | module | [section](#trainingdataset_provenancepy) | Read, write, and validate dataset provenance records. |
| `finetune_external_value.py` | runner | [section](#trainingfinetune_external_valuepy) | Run a short value-only experiment from an external JSONL teacher archive. |
| `freeze_campaign_snapshot.py` | runner |  | Freeze the currently accepted shards of a local self-play campaign. |
| `migrate_selfplay_v3.py` | runner | [section](#trainingmigrate_selfplay_v3py) | Migrate every local self-play generation to one verified V3 contract. |
| `mirror_augmentation.py` | module |  | Horizontal (left-right) board symmetry transformations for Quoridor. |
| `mix_experimental_datasets.py` | runner | [section](#trainingmix_experimental_datasetspy) | Build a disk-backed, split-safe experimental training mixture. |
| `mix_teaching_datasets.py` | runner |  | Mix a direct replay dataset and a search-teaching dataset by sample weight. |
| `model_soup.py` | runner | [section](#trainingmodel_souppy) | Create model soups by convex combination of compatible NNUE student checkpoints. |
| `new_experiment.py` | runner | [section](#trainingnew_experimentpy) | Create a new NNUE experiment workspace. |
| `parity_check.py` | module | [section](#trainingparity_checkpy) | Evaluate float and quantized NNUE weights for Python and C++ parity. |
| `prepare_replay.py` | runner | [section](#trainingprepare_replaypy) | Build a replay dataset from existing canonical self-play shards. |
| `prepare_stored_replay_all.py` | runner | [section](#trainingprepare_stored_replay_allpy) | Prepare all frozen accepted stored-search shards into a disk-backed replay. |
| `prepare_temporal_search.py` | runner | [section](#trainingprepare_temporal_searchpy) | Prepare trajectory-aware search targets from downloaded Drive artifacts. |
| `quantize_nnue.py` | module | [section](#trainingquantize_nnuepy) | quantize_nnue.py -- quantização int8/int16 (Seção 7.8 do plano), espelhando exatamente `NNUEWeightsQuant::loadFromFile` em nnue.hpp. |
| `read_selfplay.py` | module |  | read_selfplay.py -- leitura do dataset binario de self-play gerado por `selfplay` (harness C++, ver selfplay.hpp). |
| `run_architecture_matrix.py` | runner |  | Train and benchmark an NNUE architecture matrix in one resumable run. |
| `run_campaign.py` | runner |  | Train and compare several NNUE students on shared teaching data. |
| `run_colab_anneal_and_arena.py` | module |  | Compatibility entry point. |
| `run_contact_finetune.py` | runner | [section](#trainingrun_contact_finetunepy) | Train the contact-bucketed experimental candidate with high VRAM batch size and auto-resuming supervisor. |
| `run_experiment.py` | runner | [section](#trainingrun_experimentpy) | Train, export, and build one compact NNUE experiment. |
| `run_experimental_anneal.py` | runner |  | Prepare champion-initialized candidates and train a shared annealing matrix. |
| `run_local_training.py` | runner | [section](#trainingrun_local_trainingpy) | Select network capacity, teaching sources, and value targets independently. |
| `run_policy_mix_campaign.py` | runner | [section](#trainingrun_policy_mix_campaignpy) | Wait for a width screen, then compare mixed policy teaching with replay alone. |
| `run_production_finetune.py` | runner | [section](#trainingrun_production_finetunepy) | Prepare all accepted new data and fine-tune the production architecture. |
| `run_stage_a.py` | runner | [section](#trainingrun_stage_apy) | Execute Stage A evaluation of auxiliary policy head against frozen baseline. |
| `run_teaching.py` | runner | [section](#trainingrun_teachingpy) | Create a multi-teacher dataset for an NNUE student. |
| `run_temporal_campaign.py` | runner | [section](#trainingrun_temporal_campaignpy) | Prepare downloaded trajectories, train 100 epochs, and verify native exports. |
| `run_width_pilot.py` | runner | [section](#trainingrun_width_pilotpy) | Train matched width controls, verify native parity, and run paired games. |
| `select_replay_disagreement.py` | runner |  | Export the hardest direct-relabel states for expensive search teaching. |
| `strong_cycle.py` | runner |  | Run one search-driven self-play cycle and gate the candidate by Elo. |
| `student_model.py` | module |  | Compact NNUE students with explicit architecture and portable exports. |
| `target_options.py` | module |  | Resolve independent value-target settings without loading training libraries. |
| `train_nnue.py` | runner |  | Inspect, initialize, prepare, and train the shared architecture suite. |
| `train_teacher_policy.py` | module |  | Fine-tune the ZQuoridor policy head from an external teacher dataset. |

### `training/teachers/`

| Script | Role | Details | Summary |
| :--- | :--- | :--- | :--- |
| `claustrophobia_inference_worker.py` | module |  | Binary inference transport for the original Claustrophobia Rust MCTS. |
| `claustrophobia_relabel.py` | runner |  | Relabel ZQuoridor positions with a Claustrophobia TorchScript teacher. |
| `claustrophobia_search_relabel.py` | runner |  | Relabel positions with Claustrophobia MCTS visit policy and root value. |
| `collect.py` | module |  | Collect architecture-neutral data from a Quoridor teacher engine. |
| `common.py` | module |  | Common helpers for architecture-neutral teacher data collection. |
| `export_positions.py` | module |  | Export de-duplicated architecture-neutral positions from teacher/DAgger shards. |
| `select_samples.py` | module |  | Select high-value samples from architecture-neutral teacher corpora. |
| `targets.py` | module |  | Common teacher-target helpers for multi-source policy/value distillation. |
| `teaching_pipeline.py` | runner |  | Create a cached multi-teacher policy and value dataset. |
| `titanium_collect.py` | module |  | Collect architecture-neutral policy-teacher data from pinned Titanium. |
| `value_targets.py` | module |  | Build signed mover value targets for teacher distillation. |
| `zq_deep_relabel.py` | runner |  | Relabel positions with a large-budget ZQuoridor MCAB teacher. |

<!-- script-index:end -->
