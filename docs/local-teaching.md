# Local teaching and architecture experiments

Each public runner has a `CONFIG` block near the top. Edit that block and run the script without arguments. CLI arguments override the supplied settings. Run these commands from the repository root:

```powershell
python tools/setup_bots.py
python tools/run_benchmark.py
python training/run_teaching.py
python training/run_experiment.py
python training/run_campaign.py
```

The first two commands install the pinned opponents and benchmark the current engine. The next two create a small teaching corpus and train one candidate. The campaign performs a larger experiment with shared data, multiple architectures, and independent arenas. Do not run several campaigns against the same output directory or GPU at once.

## Dependencies and local files

Use Python with NumPy and PyTorch, Git, a C++17 compiler, and Rust/Cargo. CUDA is optional. On Windows, MinGW-w64 and a Rust GNU toolchain work together. The setup script also finds a repo-local toolchain under `external_bots/.toolchain/cargo` and `external_bots/.toolchain/rustup`.

Opponent sources, builds, and weights stay under ignored `external_bots/titanium` and `external_bots/claustrophobia`. Datasets, checkpoints, logs, and arena results also stay in ignored folders. The setup script checks pinned Git revisions and the Claustrophobia checkpoint SHA-256. It refuses to replace an unknown modified checkout.

Titanium runs its original executable. Claustrophobia runs its upstream Rust MCTS through a local bridge, with its original TorchScript network in a persistent Python process. This avoids the Windows libtorch/MSVC dependency. The bridge adds IPC overhead. Reports record this backend, device, simulations, and elapsed time. A simulation budget for Claustrophobia is not an equal-time comparison with the C++ engine.

## Benchmark

```powershell
python tools/run_benchmark.py --opponents titanium,claustrophobia --pairs 20 --claustrophobia-device gpu
python tools/run_benchmark.py --opponents titanium --pairs 40 --zq-move-time-ms 200 --titanium-move-time-ms 200 --output benchmark_results/titanium-200ms
```

A pair uses one opening with both player assignments. The default frozen book has 40 openings. Supply `--openings` for a larger independent book. Set `--nnue`, `--zq-executable`, and repeated `--zq-arg` values to evaluate another engine. Architecture experiments build their matching executable automatically.

The referee checks legal pawn moves, jumps, walls, and paths to goal. Failures are separate from draws and are excluded from pair scores. `games.jsonl` records each result. Resume checks manifests and skips completed games. Use a new output directory after changes to weights, executable, openings, or game settings.

Summaries include paired bootstrap intervals and a conservative bounded-score interval for decisions. The latter avoids false certainty from a tiny perfect score. At least 100 complete pairs and no failures are necessary for `strength_claim_ready`; that flag alone does not establish superiority. The lower decision bound must also exceed 50%. A fixed opening suite measures performance on that suite.

## Teaching choices

`run_teaching.py` supports three modes:

| Mode | Sources |
| --- | --- |
| `direct` | Old NNUE and direct Claustrophobia network outputs |
| `search` | Zquoridor search and Claustrophobia MCTS |
| `mixed` | All four sources |

```powershell
python training/run_teaching.py --mode mixed --games 128 --max-positions 2048 --claustro-sims 128 --zq-nodes 512 --gamma 0.995 --outcome-weight 1 --bootstrap-weight 1 --out-dir data/teaching/mixed
python training/run_teaching.py --mode direct --positions data/teaching/mixed/positions.jsonl --policy-source-weight old-direct=1 --policy-source-weight claustro-direct=2 --out-dir data/teaching/direct
```

Separate policy and value source weights, policy temperature, value temperature, chunk size, batch size, device, simulation count, cpuct, own-search nodes, time budget, and leaf depth are configurable. Set `--old-nnue` and `--zq-nnue` to change the direct teacher and the search teacher. Override bridge or checkpoint paths when required. Keep a matching architecture executable when using nonstandard NNUE weights.

Values use the side-to-move frame in `[-1,1]`. Temporal supervision is `signed_result * gamma ** remaining_plies`. It discounts a signed value toward a draw, not a win probability toward a loss. It requires a known game outcome and remaining plies. Fresh unfinished games are excluded when outcome supervision is active. Supplied data with missing required metadata fails validation.

`bootstrap_weight` adds the current position's own-search value to the value target. It supports search distillation and subsequent generations by changing the teacher. It is not an implementation of TD(lambda). Search and direct targets remain in separate chunk caches. A change to teacher code, weights, data, or settings invalidates the relevant cache.

The empty-handed production solver exports a legal selected move and its solved value even when no MCTS root exists. That makes solved final positions available for teaching.

## Data and architectures

Use `data/selfplay_canonical_v3` for all replay and standard NNUE training.
`training/migrate_selfplay_v3.py` is the only migration entry point for old
27-byte, V2, or V3 shards. It preserves originals, writes 64-byte V3 shards,
and emits a manifest for every generation. `prepare_replay.py` samples only
this canonical corpus. Fresh JSONL data must contain real histories; the
pipeline never invents a history for an old binary position.

The default campaign labels 1,000,000 replay positions and requests up to 16,384 fresh positions from 1,024 games. It uses replay chunks of 8,192, GPU inference batches of 4,096, GPU training batches of 4,096, and eight CPU workers for trajectory collection. Fresh targets contribute 20% of the loss weight in each split. The fresh teacher uses 1,024 Zquoridor nodes and 256 Claustrophobia simulations per position. The final manifest gives actual counts after filtering. Splits use whole shards or trajectory groups. Identical states cannot cross train and validation. Frozen benchmark opening states are excluded. The campaign reserves separate screening, confirmation, and external opening states before training and excludes their prefixes from the combined dataset.

The first candidates are:

| Architecture | Inputs | Accumulator |
| --- | ---: | ---: |
| `base:256` | 354 | 256 |
| `base:384` | 354 | 384 |
| `race:256` | 456 | 256 |
| `race:384` | 456 | 384 |

The 102 extra inputs encode distance margin, wall-stock margin, and the interaction between race lead and wall stock. They reuse the existing cached BFS buckets. The accumulator still updates incrementally. These features expose path and resource relations cheaply; arena results must establish whether they reduce wandering or improve strength.

The current `exp/gen11-teacher-features` branch is a negative control: its 320-wide and compact race variants lost to the same `main` baseline in its recorded arena. The campaign therefore keeps `base:256` as a trained control and treats every race model as an ablation, never as a presumed upgrade. The mixed old-network, own-search, and Claustrophobia teaching sources test a different hypothesis from that branch: whether stronger targets help the compact model without assuming extra features are beneficial.

All students keep the compact value and policy heads and fixed quantization scales. The trainer supports full-network, heads-only, or policy-only optimization; QAT; widths 128, 256, 384, and 512; initialization from a compatible old model; and direct output distillation from scratch. Expanding a model preserves its initial function. Shrinking requires `--from-scratch`.

### Architecture and search compatibility

The search algorithm is shared by all NNUE variants: alpha-beta, MCAB, move
ordering, wall quiescence, and the endgame race solver do not change merely
because the network width changes. The network changes the leaf value and
policy-ordering scores used by that search. The C++ executable is therefore
compiled for the exact student shape and loads the matching quantized file:

| Student | Compile flags | Quantized layout |
| --- | --- | --- |
| `base:256` | `ZQ_NNUE_RACE_FEATURES=0`, `ZQ_NNUE_HIDDEN=256` | 354 x 256 |
| `base:384` | `ZQ_NNUE_RACE_FEATURES=0`, `ZQ_NNUE_HIDDEN=384` | 354 x 384 |
| `race:256` | `ZQ_NNUE_RACE_FEATURES=1`, `ZQ_NNUE_HIDDEN=256` | 456 x 256 |
| `race:384` | `ZQ_NNUE_RACE_FEATURES=1`, `ZQ_NNUE_HIDDEN=384` | 456 x 384 |

`run_experiment.py` builds these flags and records them in
`student.architecture.json`. Loading a `race` or wider weight file into the
production executable is rejected by the size check; it is not a runtime
architecture switch. To compare students, build one executable per student
and pass its weights to the arena. Search tuning can still differ per
executable through its normal runtime options, but it must be reported as a
separate search configuration rather than attributed to the NNUE alone.

```powershell
python training/run_experiment.py --no-teaching --data data/teaching/mixed/dataset.npz --architecture race --hidden 384 --out-dir results/race384
python training/run_experiment.py --no-teaching --data data/teaching/mixed/dataset.npz --architecture race --hidden 128 --from-scratch --out-dir results/race128
```

Training exports float and quantized weights, an architecture manifest, a restart checkpoint, and a matching native engine. Production remains the default 354-input, 256-unit build. Do not load an experimental binary weight file in a production-width executable.

## Campaign decisions and restart

The default campaign trains each architecture with two seeds. Each candidate plays 100 screening pairs at 50 ms per move against the current baseline. The selected candidate then plays 400 new confirmation pairs at 200 ms. Both the candidate and baseline play 100 additional pairs against each external opponent. Claustrophobia uses 512 simulations per move. This can take several hours on a laptop.

Read `results/campaign/status.json`, each `train_report.json`, arena `summary.json` files, and the final `decision.json`. Re-run the same command to resume completed teaching chunks, training epochs, and arena games. Configuration or input changes require a new directory. An interrupted trajectory collection can be regenerated; a completed collection is reused with its checksum.

The screening winner is a selection for confirmation, not proof that its architecture is best. External score differences are descriptive unless their own uncertainty supports a conclusion. The campaign can conclude that evidence is insufficient. It never replaces production weights automatically. Increase data diversity, teaching budget, or independent evaluation pairs in a new campaign when the result remains inconclusive.
