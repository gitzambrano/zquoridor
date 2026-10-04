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

### `tools/run_complete_promotion_battery.py`
- **Purpose**: Runs the full evaluation battery for candidate models against the frozen production champion, Claustrophobia, and Titanium. Measures win rates, Elo with 95% bootstrap confidence intervals, average search depth, and nodes per second (NPS).
- **Inputs**:
  - `--candidate-exe`: Path to candidate executable.
  - `--candidate-nnue`: Path to quantized int8 candidate weights.
  - `--baseline-exe`: Path to baseline executable.
  - `--baseline-nnue`: Path to baseline int8 weights.
  - `--normal-openings`: Path to normal opening book (`tools/external/openings_screen_v1.jsonl`).
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

### `tools/setup_bots.py`
- **Purpose**: Clones, compiles, and configures external benchmark opponents (`titanium`, `claustrophobia`).
- **Inputs**: Git repositories and pre-trained checkpoints.
- **Outputs**: Binaries under `external_bots/`.

---

## 2. Remote and Cloud Self-Play (Google Colab)

### `scripts/colab/launch_workers.py`
- **Purpose**: Bootstraps Google Colab self-play workers headlessly via Playwright. Injects the resilient bootloader cell, connects runtimes, and triggers execution.
- **Inputs**: Authenticated user profiles under `C:\Projetos\TikTok\profiles\` with stored cookies.
- **Outputs**: Active self-play sessions writing to Google Drive and confirmation screenshots in `artifacts/colab/`.
- **Key CLI flags**: `--worker-ids` (e.g. `1 2 3 4 5`), `--headless`, `--force-restart`.

### `scripts/colab/watchdog_workers.py`
- **Purpose**: Continuous keep-alive monitor for active Google Colab workers. Simulates human idle interactions to prevent disconnects and auto-reconnects dropped runtimes.
- **Inputs**: Worker registry in `scripts/colab/config.py`.
- **Outputs**: Health checks, periodic status screenshots, and automatic cell re-execution.
- **Key CLI flags**: `--worker-ids`, `--check-interval-seconds`, `--headless`, `--target-delta-positions`.

### `scripts/colab/report_workers.py`
- **Purpose**: Audits live Colab notebook DOM state, current shard indices, and games completed across all workers.
- **Inputs**: Active browser contexts or profile sessions.
- **Outputs**: Markdown summary table and per-worker status screenshots.

### `scripts/colab/login_worker.py`
- **Purpose**: Interactive login helper to authenticate Google accounts and export permanent session cookies to `cookies.json`.
- **Inputs**: `--worker-id` (single worker) or `--worker-ids` (sequential list).
- **Outputs**: Updated `cookies.json` in worker profile directory.

### `tools/selfplay/run_colab_worker.py`
- **Purpose**: Standalone chunked self-play generator designed to execute on Linux/Colab virtual machines.
- **Inputs**: Opening bank JSONL, self-play binary (`bin/selfplay`), int8 weights.
- **Outputs**: Aligned V3 shards (`c{id}_shard_XXXX.bin`) and metadata sidecars in Google Drive.
- **Key CLI flags**: `--worker-id`, `--drive-dir`, `--total-games`, `--chunk-games`, `--time-ms`, `--positions`, `--mc-temp-obvious`, `--playout-cap`, `--cheap-time-ms`, `--full-search-opening-plies`, `--full-search-prob`.

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

### `training/parity_check.py`
- **Purpose**: Validates mathematical parity between Python forward passes and C++ engine evaluation across thousands of test positions.
- **Inputs**: Trained student weights and compiled C++ test harness.
- **Outputs**: Discrepancy report and maximum absolute difference metrics.

