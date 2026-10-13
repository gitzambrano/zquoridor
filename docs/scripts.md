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

---

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

## 2. Remote and Cloud Self-Play (Google Colab)

### `colab/switch_search_games.py`
- **Purpose**: Completes the current self-play shard on each account, then launches paired search collection through its notebook. One locked controller monitors all selected accounts.
- **Inputs**: Authenticated account profiles in `colab/config.py`, a pinned Main revision, and the collection settings in `colab/search_games_profile.py`.
- **Outputs**: Controller status in `artifacts/colab/search_games_handover/status.json` and search game ledgers in each account's Drive directory under `zquoridor_data/claustro_search_games_v1/worker_X`.
- **Default settings**: Five accounts, one game per account, 50000 pairs per account, 250 games per batch, 70% Center Rush, 20% Normal, 10% weakness, initial temperature 1.0 for 14 plies, and both engines' searches recorded. Each account uses a distinct seed and a frozen artifact cache.
- **Key CLI flags**: `--worker-ids`, `--revision`, `--check-interval-seconds`, `--headless`, and `--output`.
- **Notes**: Stop the old Zquoridor watcher before this controller. The controller requires the Colab runtime terminal. It preserves browser profile locks and does not bypass authentication or runtime limits. It attempts browser recovery three times per account and reuses an active handover helper. A persistent failure blocks collection on that account.

### `colab/watch_search_games.py`
- **Purpose**: Provides the automatic Python watcher entry point for paired search collection. It completes the initial shard handover and reconnects or resumes the frozen workflow after a notebook disconnect.
- **Inputs and defaults**: Uses `switch_search_games.CONFIG` and the collection profile. The native runner manages every batch and writes it to Drive.
- **Outputs**: The shared controller status file and remote Drive batch artifacts.
- **Driver recovery**: A closed browser driver permits three retries with a 60-second delay. A separate watcher lock prevents duplicate controllers during recovery.
- **Key CLI flags**: The switch flags, plus `--auto-resume`, `--retry-cooldown-seconds`, and `--max-resume-attempts`. Automatic resume permits three attempts with a 900-second cooldown. Identity mismatches and authentication requirements block automatic re-execution.

### `colab/finish_current_shard.py`
- **Purpose**: Completes one native self-play shard before a workflow change. The Linux helper pauses only its Python launcher. It leaves the native child active, verifies aligned binary and metadata files after exit, then terminates the launcher.
- **Inputs**: Linux process identities and the worker ID.
- **Outputs**: An atomic JSON status, artifact hashes, and terminal status markers. Errors resume the launcher.
- **Default settings**: Worker 1, a two-second check interval, and `/content/zquoridor_search_games/handover_worker_1.json`.
- **Key CLI flags**: `--worker-id`, `--status-path`, and `--poll-seconds`.

### `colab/launch_workers.py`
- **Purpose**: Bootstraps Google Colab self-play workers headlessly via Playwright. Injects the resilient bootloader cell, connects runtimes, and triggers execution.
- **Inputs**: Authenticated user profiles under `C:\Projetos\TikTok\profiles\` with stored cookies.
- **Outputs**: Active self-play sessions writing to Google Drive and confirmation screenshots in `artifacts/colab/`.
- **Key CLI flags**: `--worker-ids` (e.g. `1 2 3 4 5`), `--headless`, `--force-restart`.

### `colab/watchdog_workers.py`
- **Purpose**: Continuous keep-alive monitor for active Google Colab workers. Simulates human idle interactions to prevent disconnects and auto-reconnects dropped runtimes.
- **Inputs**: Worker registry in `colab/config.py`.
- **Outputs**: Health checks, periodic status screenshots, and automatic cell re-execution.
- **Key CLI flags**: `--worker-ids`, `--check-interval-seconds`, `--headless`, `--target-delta-positions`.

### `colab/report_workers.py`
- **Purpose**: Audits live Colab notebook DOM state, current shard indices, and games completed across all workers.
- **Inputs**: Active browser contexts or profile sessions.
- **Outputs**: Markdown summary table and per-worker status screenshots.

### `colab/login_worker.py`
- **Purpose**: Interactive login helper to authenticate Google accounts and export permanent session cookies to `cookies.json`.
- **Inputs**: `--worker-id` (single worker) or `--worker-ids` (sequential list).
- **Outputs**: Updated `cookies.json` in worker profile directory.

### `colab/dispatch_rich_targets.py`
- **Purpose**: Autonomous orchestrator that deploys Wall-Drop generation and Gumbel AlphaZero Sequential Halving across all 5 remote Google Colab workers. Handles Google Drive mounts, injects native C++ compile steps, and monitors execution health.
- **Inputs**: Registered worker definitions in `colab/config.py`.
- **Outputs**: Remote execution jobs producing `positions.jsonl`, `targets.jsonl`, and `targets.npz` in Google Drive.

### `colab/orchestrate_v4.py`
- **Purpose**: Fleet controller for continuous self-play and Claustrophobia search collection. Detects GPU quota limits, switches to CPU self-play, and keeps browser connections alive.
- **Inputs**: Worker configuration dictionary and runtime profiles.
- **Outputs**: Execution monitor log and active game generation on Google Drive.

### `colab/solve_audio.py`
- **Purpose**: Solves Google reCAPTCHA v2 audio challenges using the OpenAI Whisper tiny model. Downloads challenge MP3 payloads, resamples to 16 kHz, transcribes spoken text, and submits verification.
- **Inputs**: CDP port of the target browser instance (defaults to 9005).
- **Outputs**: Verified reCAPTCHA session.

### `colab/fix_fleet_popups.py`
- **Purpose**: Scans all 5 Colab workers for pending Google Drive OAuth consent tabs and reCAPTCHA challenge dialogs, approves permissions, and ensures Cell 0 execution.
- **Inputs**: Running CDP worker endpoints on ports 9001 to 9005.
- **Outputs**: Approved permissions and triggered notebook cells.

### `tools/selfplay/run_colab_worker.py`
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
- **Key CLI flags**: `--count`, `--min-walls`, `--max-walls`, `--pawn-walk-plies`, `--move-time-ms`, `--zq-executable`, `--output`, `--seed`, `--split`, `--no-export-npz`.

### `tools/generate_gumbel_sims.py`
- **Purpose**: Executes Gumbel AlphaZero planning and Sequential Halving to generate high-density improved policy targets and expected values with focused simulation budgets.
- **Inputs**: Optional source position corpus (`--source-positions`), Zquoridor UCI executable (`--zq-executable`), candidate pool size (`--top-k`), value scaling factor (`--c-scale`), search time (`--move-time-ms`), random seed (`--seed`), and split name (`--split`).
- **Outputs**: Refined JSONL positions (`positions.jsonl`), Gumbel teacher targets (`targets.jsonl`), compressed NumPy arrays (`targets.npz`), and execution manifest (`manifest.json`) under the output directory.
- **Key CLI flags**: `--count`, `--top-k`, `--c-scale`, `--move-time-ms`, `--zq-executable`, `--source-positions`, `--output`, `--seed`, `--split`, `--no-export-npz`.

---

## 4. Dataset Preparation and Mixing

### `training/prepare_replay.py`
- **Purpose**: Converts raw V3 self-play shards into training datasets with policy distributions and value targets.
- **Inputs**: Directory of V3 shards and metadata sidecars.
- **Outputs**: `dataset.npz` or memory-mapped array folder with `dataset.manifest.json`.
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

## 5. Training, Quantization, and Parity

### `training/run_experiment.py`
- **Purpose**: Core neural network training engine for compact NNUE students. Supports cosine annealing, QAT, horizontal reflection, and custom value heads.
- **Inputs**: Prepared dataset directory, initialization weights.
- **Outputs**: `student.bin` (float), `student_int8.bin` (quantized), architecture manifests, and training history JSON.
- **Key CLI flags**: `--architecture`, `--hidden`, `--epochs`, `--batch-size`, `--lr`, `--qat`, `--mirror-h`, `--resume`.

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


## Width comparison runner

### `training/run_local_training.py`

This portable entry point selects hidden widths 512, 768, or 1024.
`--target-mode current` uses current search values and the selected outcome
weight. `--target-mode temporal` combines current and future search values.
`--future-horizons` and `--future-weights` accept JSON lists. Their defaults
are `[0,4,8,16]` and `[0.5,0.25,0.125,0.0625]`. `--result-weight` defaults
to 0.1. `--apex-value-fraction` independently selects the apex share of value
mass; zero excludes apex. Apex is a data-source parameter, not a training mode.
All choices retain QAT, augmentation, auxiliary policy, and source confidence.
The default schedule uses 100 epochs and cosine rates from 1e-5 to 1e-7.
`--data`, `--init-from`, `--out-dir`, `--epochs`, `--lr`, `--min-lr`,
`--device`, and `--batch-size` override defaults. `--dry-run` prints the
resolved configuration. `--no-build` skips native compilation and parity.
Target changes require trajectories or aligned metadata. The runner prepares
a separate cache when existing targets do not match the requested settings.
Future value targets preserve the original mover perspective. Policy targets
remain attached to the current position. Apex-only samples retain their
provided value because the archive does not supply future trajectories.
Outputs include weights, architecture metadata, checkpoints, training
reports, and native verification when enabled. The runner resolves default
paths relative to its project directory and does not modify production.

### `training/run_apex_mix_campaign.py`

This runner waits for a completed width campaign and its paired match report.
It prepares a canonical mixed dataset, trains matched replay-only and apex
arms, verifies native exports, and compares the candidate with both controls.
`CONFIG` defines the dependency, archive, replay dataset, output paths, value
fraction, epochs, rates, and game settings. CLI flags override these values.
`status.json` records the dependency wait and subsequent phases. Outputs
include the mixed manifest, selected artifact hashes, weights, checkpoints,
original replay validation, parity reports, games, and `report.json`.
Policy availability masks exclude apex-only states from both policy losses.
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
