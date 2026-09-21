# Status & Technical Reference

Technical reference and roadmap for Zquoridor. History here is minimal:
only durable lessons that a future contributor (human or LLM) must not
relearn by experiment.

---

## 1. Production Status

### Version 2.00 release state — 2026-09-21

- **Search**: production hybrid PUCT/MCGS with alpha-beta support, graph
  transpositions, Q-corrected cross-ply sharing, persistent tree reuse,
  bounded caches, root repetition escape, and adaptive real-clock budgeting.
- **Pondering**: opponent-root pondering is a production feature. A paired
  120-game fixed-200 ms A/B against the same engine without pondering scored
  **68.33% (+133.6 Elo)** with a 95% bootstrap score interval of
  **61.25%–75.42%**. The candidate reused a pondered subtree on **86.8%** of
  its searches. A separate 100-game-per-configuration Claustrophobia check
  moved from **51.0%** without pondering to **51.5%** with pondering; that
  sample establishes no significant external-opponent gain but showed no
  measured regression. The small 3+2 experiment contained a clock timeout
  and is not a valid strength measurement.
- **Browser**: engine play and pondering run in a Web Worker. Pondering is
  split into short bounded slices so UI work stays on the main browser thread.
- **Network**: production `multipath_phase:512` at
  `data/nnue/nnue_weights_int8.bin`: **504 sparse inputs → 512 SCReLU**,
  WL head `512→32→1`, policy head `512→209`, QAT/int8 inference.
- **Protocol**: `tools/external/zquoridor_uci.cpp` exposes the UCI-style text
  protocol plus `ponder movetime <ms>` and an interactive `help` command.

- **Search**: hybrid PUCT MCTS (`src/mcab.hpp`), default in all tools.
  Leaves are direct `nnueEvalInt` (`leafDepth=0`), except in a wall-poor
  endgame: `endgameMoverWallThreshold=0` gives alpha-beta leaves of 2 plies
  once the side to move has no walls left. Backup `AvgBlend`,
  tree reuse on, node budget 20000/move (not time-binding: p99 is ~18.6k
  nodes at 150ms).
- **Network**: Production baseline `multipath_phase:512` (`data/nnue/nnue_weights_int8.bin`),
  `504 -> 512` SCReLU accumulator (`ZQ_NNUE_RACE_FEATURES = 1`,
  `ZQ_NNUE_MULTIPATH_FEATURES = 1`, `ZQ_NNUE_PHASE_FEATURES = 1`,
  `ZQ_NNUE_HIDDEN = 512`),
  WL head `512->32->1`, policy head `512->209`. QAT with fixed `QA=255`, `QB=64`.
- **Experimental Architectures**:
  - `multipath:512` (`src/nnue.hpp`, opt-in via `-DZQ_NNUE_MULTIPATH_FEATURES=1`):
    `480 -> 512` SCReLU accumulator adding 24 multi-path and pawn collision features
    (0 extra BFS: 8 directional unblocked exits, 8 exit count buckets, 8 pawn collision features).
    **600-game match vs Claustrophobia GPU: 51.08% (+7.53 Elo)** — project historical first win vs Claustrophobia GPU.
  - `margin_regime:512` (`src/nnue.hpp`, opt-in via `-DZQ_NNUE_MARGIN_REGIME_FEATURES=1`):
    `588 -> 512` SCReLU accumulator adding 132 distance margin $\times$ wall regime interaction features.
    **600-game match vs Claustrophobia GPU: 49.00% (-6.95 Elo)**.
  - `multipath_phase:512` (active experiment, started 2026-09-20):
    `504 -> 512` SCReLU accumulator combining the 24 multipath features and 24 phase features.
    The phase block adds no BFS and keeps the value and policy heads unchanged. The warm start remaps
    the existing multipath columns and initializes only the phase columns to zero. Training uses the
    11,065,000-sample weakness-boosted dataset for 100 QAT epochs. Cosine schedules reduce the learning
    rate and weight decay to `1e-7`. Source-level boosts increase the weight of bilateral search,
    critical search, crisis search, and 512-simulation deep-search samples. The campaign will run a
    paired match against `multipath:512`, then the full Claustrophobia and Titanium benchmark suite.
    The promotion target is more than 60% against Claustrophobia in every opening family at 200 ms.
    The completed candidate scored 51.25% against the synchronized multipath baseline
    (200 games), 50.83% against Claustrophobia overall (600 games), 60.0% against
    Titanium normal (100 games), and 46.0% against Titanium Center Rush (100 games).
    Its direct Claustrophobia Center Rush score was 37.5% (100 games). All five required
    families failed the gate: `front_wall` 38.46%, `pawn_jump` 50.0%,
    `vertical_channel` 41.67%, `reed_rear_wall` 22.22%, and `sidestep_flank` 33.33%.
    The 3+2 clock smoke completed four games without a failure.
  - `multipath_phase:512` reliable-search fine-tune (queued 2026-09-20):
    the follow-up dataset contains 313,344 samples and excludes direct-network labels,
    stale Gen 1 search, and duplicated standalone subsets. It contains 100k bilateral
    generic-search positions, 100k generic-critical positions at 2x weight, 100k
    dual-crisis positions at 2.5x weight, 10k Claustrophobia 512-simulation positions
    at 8x weight, and 3,344 terminal rollout steps at 4x their stored weight. Total
    gradient mass is 3,672,536.25; critical-search sources contribute most of it and
    recovered rollouts remain a small auxiliary block. The 40-epoch fine-tune starts
    from the completed phase network with a `5e-6` head learning rate and a 0.05 trunk
    multiplier. It runs only after the active benchmark releases the GPU. Promotion
    requires paired matches against both the phase network and the strongest multipath
    baseline, the five-family Claustrophobia gate, the full external suite, and a four-game
    3+2 clock smoke test.
  - `multipath_phase_contact:512` (queued 2026-09-20): `858 -> 512`
    SCReLU accumulator. The architecture keeps all 504 Stage A inputs and adds
    354 sparse contact columns. Exactly four new columns are active: one exact
    relative pawn displacement, one local edge mask for each pawn, and one
    direct-jump or diagonal-option state. These features use local wall tests
    and add no BFS. A zero-column warm start from Stage A preserves both heads
    exactly before training. The campaign uses 11,065,000 broad samples plus
    313,344 reliable search and rollout samples at 2x source weight. It uses
    160 QAT epochs, eight warmup epochs, cosine learning-rate and weight-decay
    schedules, a 1.5 policy-loss weight, a 1.0 value-loss weight, 6 GB of GPU
    memory, and 16 CPU threads. The campaign starts after the active Stage B
    controller exits. It then runs paired 200 ms tests and a four-game 3+2
    clock smoke.
- **Time management**: `src/time_manager.hpp` allocates a move budget from
  the remaining clock, increment, estimated moves to go, ply, and move overhead.
  Version 1 returns an optimum and maximum budget. The production search uses
  the optimum budget as the effective move limit. The external adapter accepts
  `wtime/btime/winc/binc/movestogo`, and the arena accepts
  `--tc-base-ms` plus `--tc-inc-ms` for game-clock tests. `tools/run_clock_smoke.py`
  runs four Zquoridor games from two color-swapped openings at 3 minutes plus a
  2-second increment. This is a clock-safety check, not a strength benchmark.
- **MCGS self-play heap fix (2026-09-21)**: `MCABSearch::runSimulation` no
  longer keeps its descent path in a dynamic `thread_local` vector. MinGW
  could report `0xC0000374` when a self-play worker destroyed that vector
  after hundreds of games. The path buffer now belongs to each search object.
  Replaying the failing shard 14 (512 games, 10 threads, 50 ms/move) changed
  the result from a crash at 509/512 to exit 0 with 512/512 games and 13,995
  positions. The accepted self-play shards were preserved.
- **Performance baseline (2026-09-20)**:
  - `race512-cr200k-champion` (Production baseline on `main`):
    - vs Titanium: **63.0% (+92.5 Elo)** in official 600-game match (378W / 0D / 222L).
    - vs `main` (Gen 5): **81.25% (+254.7 Elo)** in direct screening (32W / 1D / 7L).
    - vs Claustrophobia: **47.92% (-14.5 Elo)** in 600-game match on GPU (141W white / 140W black).
- Center-Rush Tactical Suite: **37.12% (+35 Elo)** against Claustrophobia (`pawn_jump` at 62.5%, `front_wall` at 47.2%).

### Main synchronization and baseline promotion (2026-09-21)

- Local `main` now includes the MCGS Q-correction promotion through `fb110ab`.
- The synchronized search includes the MCAB transposition DAG, adaptive real-clock budgeting, and bounded hot-node retention.
- The DSU move-generation changes were already present before this synchronization.
- The production weights now use the Stage A `multipath_phase:512` checkpoint.
- Default native and WASM builds use the 504-feature layout.
- C++ and Python parity checks agree on the fixed test position.
- The MCGS promotion is active in the production search path.
- A 400-game Claustrophobia check remains an external experiment. Its result does not change the selected weights automatically.
  - `multipath:512` (Experimental Champion):
    - vs Claustrophobia GPU (600g): **51.08% (+7.53 Elo)** (306.5 / 600, 95% CI: [47.50%, 54.58%]).
    - Claustrophobia Central Openings: **46.62% (34.5 / 74)** (up from 31.8% baseline).
    - vs Titanium Normal (100g): **64.0% (+99.95 Elo)**.
    - vs Titanium Center Rush (100g): **45.0% (-34.86 Elo)** (Black: 72.0%, White: 18.0%).
  - `margin_regime:512` (Experimental Candidate):
    - vs Claustrophobia GPU (600g): **49.00% (-6.95 Elo)** (294.0 / 600, 95% CI: [45.25%, 52.75%]).
    - Claustrophobia Central Openings: **43.92% (32.5 / 74)** (White: 35.1%, Black: 52.7%).
    - vs Titanium Normal (100g): **65.0% (+107.5 Elo)**.
    - vs Titanium Center Rush (100g): **47.0% (-20.87 Elo)** (Black: 70.0%, White: 24.0%).

---

## 2. Future Plans (priority order)

1. **Center-Rush Tactical Dominance Fine-Tuning Cycle (Active)**:
   Target full superiority over Claustrophobia in all 5 Center-Rush opening
   families (`front_wall`, `pawn_jump`, `vertical_channel`, `reed_rear_wall`,
   `sidestep_flank`) while preserving overall strength (>60% vs Titanium,
   >80% vs previous baseline). Mine losses from the completed 600-game match
   vs Claustrophobia, execute deep MCTS search relabeling (512-1024 sims) with
   Action-Q sharpening, and run modular fine-tuning.
   The active `multipath_phase:512` campaign tests whether the wall-stock phase interaction improves
   weak central openings without a measurable search-speed regression.
2. **Self-play generation, Gen 6**: regenerate datasets with the current
   engine (~3.3x more MCTS nodes per move than the data the Gen 5 net saw),
   root visit distribution as policy target. Retrain, quantize, arena-test
   vs Gen 5.
2. **SPSA tuning** (`tools/spsa/tune_spsa.cpp`): MCAB knobs (`cPuct`,
   `fpuReduction`, `scoreScale`) and search tuning set from
   `search_tuning.hpp`. Run AFTER Gen 6 data exists.
3. **Structural speed work** (profile is flat now; diminishing returns):
   fixed-size storage for per-edge vectors inside `MCABNode`
   (~4 vectors x up to 131 moves, heap-allocated per expansion), and
   eliminating the ~1KB `AccPair` copy per traversed edge.
4. **Time-control curve**: measure hybrid vs pure AB at 500ms/1000ms;
   the 40ms and 200ms points are done (inv/ab-policy round: the hybrid
   wins at both, see History Notes).
5. **GUI worker offload** (`gui_web/`) -- completed for standalone bundle (Blob URL worker); see GUI section.
6. **MCTS visit-share %** in analysis PV rows needs a
   `rootNodeForInspection` export channel.
7. **Per-action Q information in training (Experimental Network Candidate)**:
   The search relabeling pipeline already computes per-action values: `zq_deep_relabel.py`
   records `action_q` for visited root actions, but `build_search_priority_dataset.py`
   currently discards `action_q` and trains the policy head solely on visit shares ($N_a$).
   Visits reflect tree traversal frequency, but discard the evaluated quality of alternative
   actions. In Quoridor, multiple wall placements share similar visit priors, yet only
   a few produce true strategic consequence.
   Two experimental phases are planned:
   - **Phase A (Architecture-neutral policy sharpening)**: Generate enriched policy
     targets combining visit volume and action quality without changing network shape:
     $$\pi'_a \propto N_a^\alpha \cdot \exp(\beta Q_a)$$
     This preserves the 209-action policy head while teaching the network how much
     better or worse each action evaluated.
   - **Phase B (Dedicated Q-head)**: Add an auxiliary per-action Q-value head
     to the student trunk to predict action values directly, improving move ordering
     and root pruning before PUCT expansion.
8. **Distance margin across wall regimes (Experimental Feature Candidate)**:
   The existing network contains `distance_delta` and a coarse interaction
   (`ahead/equal/behind` $\times$ wall classes). This loses the exact numerical
   magnitude of the distance advantage inside that interaction. For example,
   $(ownDist=2, oppDist=3, ownWalls=0, oppWalls=8)$ and
   $(ownDist=2, oppDist=12, ownWalls=0, oppWalls=8)$ both map to `ahead × 0 × 3+`.
   An interaction feature resolves this ambiguity:
   - 33 distance margins $[-16 \dots +16]$ cross 4 wall regimes:
     1. Both players retain walls.
     2. Mover has zero walls; opponent retains walls.
     3. Opponent has zero walls; mover retains walls.
     4. Both players have zero walls.
   - Total features: $33 \times 4 = 132$ one-hot inputs. Exactly one feature is
     active per position. This is computationally cheap for NNUE accumulator updates.
9. **Shortest path structure and corridor bottleneck masks (Experimental Feature Candidate)**:
   Current features record total distance to goal (for example, distance = 7), but
   do not distinguish whether a player has a single bottleneck corridor or multiple
   independent optimal paths. Two states with identical distance have completely
   different strategic stability. The path cache already computes cell distances.
   Extracting initial step choices costs negligible CPU time:
   - First-step branching count per player: 1, 2, 3, or 4 optimal initial moves.
   - Directional mask (forward, left, right, backward) indicating which directions
     preserve a shortest path.
   This provides the network with a structural distinction between narrow corridors
   and redundant open routes, offering stronger inductive bias than expanding hidden
   layer width.
10. **Local pawn interaction and contact geometry (Experimental Feature Candidate)**:
    Currently, pawn positions are encoded as separate one-hot vectors (81 for own pawn,
    81 for opponent pawn). The network must learn spatial proximity, jump legality,
    and diagonal detours without inductive bias. A compact block of 10 to 30 features
    provides local contact geometry:
    - Relative coordinate delta $(\Delta x, \Delta y)$ between pawns.
    - Orthogonal adjacency indicator.
    - Jump contact state (direct jump available, wall-blocked straight jump with diagonal options).
    This directly improves policy head accuracy for tactical close-contact plies.

---

## 3. History Notes (durable lessons only)

- **Game-clock time manager v1 (2026-09-20)**: The engine previously accepted
  one fixed `timeBudgetMs` per move. Therefore, the search did not know the
  remaining game clock or the increment. Version 1 adds a conservative budget
  allocator and keeps the existing search deadline as the enforcement point.
  The search uses the optimum budget as its hard move limit until a separate
  root-stability rule is validated. This avoids spending a larger maximum
  budget on every move before an adaptive stop rule exists. The next step is
  to use root visit share, Q gap, best-move changes, and tree reuse to decide
  whether the search can continue from the optimum limit toward the maximum
  limit.

- **Recovered center-loss rollouts (2026-09-20)**: A live-process recovery
  preserved 3,344 terminal rollout steps from 61 completed rollouts across 15
  crisis seeds in `data/teaching/loss-center-rollouts`. A full semantic audit
  found zero illegal histories, illegal policy actions, mover mismatches,
  broken trajectory links, or discounted-value mismatches. The grouped split
  keeps complete seeds on one side. The corpus has 3,274 unique histories. Of
  33 repeated-state groups, 30 contain different stochastic actions and values,
  so the records are valid Monte Carlo samples but are not fully independent.
  At weight 8, this dataset adds only 26,752
  units of gradient mass, approximately 0.06% of the 44,002,312-unit active
  training mix. It is too small to affect the current campaign and too narrow
  to upweight without a controlled follow-up experiment. Keep it separate
  unless the opening-family benchmark confirms a matching weakness.

- **Seeded weakness self-play (2026-09-20)**: `selfplay` accepts
  `--positions <zquoridor.position.v1.jsonl>` and starts each game from a
  validated nonterminal snapshot. The loader replays the full move history,
  preserves the side to move and the repetition table, and rejects an illegal
  history before generation starts. `tools/teacher/run_weakness_selfplay.py`
  writes resumable 64-byte V3 shards and stops at a position target.
  `tools/teacher/weight_weakness_seeds.py` builds the current 60% Claustrophobia
  and 40% Titanium schedule. It increases the sampling weight for swept
  openings, ZQuoridor turns in losses, and early crisis positions. The active
  campaign uses 12 CPU threads, 100 ms per move, and the
  `multipath_phase:512` Stage A weights. Its primary target is 500,000
  positions. It can add one 100,000-position block only if Stage C has no
  training report when the primary target completes.

- **Experimental candidate models tracked in version control (2026-09-18)**: The
  experimental network weights (`student.bin` and `student_int8.bin`) and their
  matching architecture manifests under `results/experiments/`,
  `results/campaign-smoke/`, and `results/campaign-verified/` are now tracked in
  Git. This allows remote continuous integration workflows and cloud runners to
  build and benchmark candidate engines directly. All training datasets,
  Claustrophobia caches, PyTorch checkpoints, and third-party bots
  (`external_bots/`) remain untracked.

- **Empirical proof of Claustrophobia search necessity over direct network (2026-09-19)**:
  Screening benchmark of candidate `race512-weakness-ft` (trained on 2M replay
  plus 34,787 weakness positions using 75% Claustrophobia direct forward-pass
  and 25% ZQuoridor search) produced a decisive divergence:
  1. Against Titanium: scored **62.5%** (+88.7 Elo, up from 57.5% in the finalist),
     proving strong mastery of anti-Titanium wall navigation.
  2. Against `main`: scored **45.0%** (-34.9 Elo).
  3. Against Claustrophobia: dropped to **31.25%** (-137.0 Elo) with **153 repeated
     states** across 40 games.
  The empirical analysis confirms that Claustrophobia's raw policy network without
  MCTS search suffers from severe tactical wandering and cyclical loops.
  Distilling the raw network directly poisoned tactical resolution in Claustrophobia
  mirror positions. Claustrophobia teaching targets MUST be generated by its MCTS
  search (`zq_search_bridge.exe`), which resolves tactical blunders through tree
  visits.

- **Claustrophobia loss taxonomy and resource preservation (2026-09-18)**: Analysis
  of 200 confirmation games between `race512-search10-ft` (the crowned finalist,
  scoring 58.5% vs `main`, 57.5% vs Titanium, and 49.0% vs Claustrophobia) and
  Claustrophobia identified wall-depletion asymmetry as the primary failure mode:
  1. ZQuoridor exhausted all 10 walls before Claustrophobia in 50.5% of games
     (101/200), winning only 32.7% of those games. When Claustrophobia exhausted
     its walls first, ZQuoridor won 72.3%.
  2. Over 54% of all losses occurred when ZQuoridor held 0 walls while
     Claustrophobia preserved 1 to 4 walls (28.8% winrate). When both held walls,
     ZQuoridor scored 53.8%; in zero-wall endgame races, `endgame_race.hpp` gave
     ZQuoridor a 55.6% winrate.
  3. In 21 of the 23 openings swept 0-2 against ZQuoridor, all six opening moves
     were wall placements, creating early corridor density that caused premature
     wall expenditure.
  4. Search throughput measured ~53.4k nodes/s (200ms) and ~65.8k nodes/s (100ms)
     for ZQuoridor (~10,680 nodes/move to ply 18-28), compared to 10-20 MCTS
     simulations/move for Claustrophobia on CPU. Claustrophobia compensates for low
     simulation volume with superior prior intuition on wall conservation.
  5. The previous search-teaching set had only 2,000 positions (0.1% of the 2M
     corpus), hitting a plateau at 48.8%-49.0%. Overcoming Claustrophobia (>50%)
     requires scaling up teaching data to 50,000+ mined positions from loss games
     and divergence states, using fast 100ms games to double generation throughput.

- **Hierarchical 5-Tier curriculum and data scaling architecture (2026-09-19)**:
  To break through the 49.0% ceiling against Claustrophobia without regressing
  against Titanium (57.5%+) or baseline `main` (58.5%+), the training pipeline
  expands from flat replay distillation into a structured 5-tier curriculum:
  1. **Tier 1 (Massive Background Replay, 10M positions)**: 10,000,000 distinct
     positions sampled across all 499 historical canonical V3 shards, with oversampling
     on wall asymmetry ($|\text{own} - \text{opp}| \ge 2$ or $\le 3$ walls) and dense boards
     ($\ge 8$ walls placed). Soft policy targets and win-probability values come from
     the champion network (`race512-search10-ft`) evaluated on CUDA. Empirical game
     results from weak early self-play generations are excluded to prevent regression.
     Base sample weight: 1.0.
  2. **Tier 2 (Reused Past Search Cases, 500k positions)**: Proven search-labeled
     positions from previous generations (`mixed-gen1-500k-search20` and `search-priority-gen1`).
     Sample weight: 2.0.
  3. **Tier 3 (Generic Search Positions, 500k positions)**: 500,000 standard game positions
     labeled with real search (Claustrophobia MCTS 64-simulation search on CUDA and
     ZQuoridor MCAB search), providing search supervision across balanced board topologies.
     Sample weight: 3.5.
  4. **Tier 4 (Dual-Crisis Search, 100k positions)**: 100,000 critical positions
     focusing on wall-depletion crises, opponent sweeps, and tactical branching.
     Supervision blends 75% Claustrophobia MCTS visit distribution with 25% ZQuoridor
     512-node MCAB search, scaled dynamically by teacher disagreement.
     Sample weight: 6.0 to 8.0.
  5. **Tier 5 (Dynamic Branching Rollouts, 10k games)**: 10,000 self-play rollout
     games initialized from crisis seeds, exploring top-4 actions in plies 1 to 4,
     followed by deterministic MCAB play. Step values receive exponential temporal
     discounting ($V_t = \text{sign} \cdot \gamma^{\text{remaining\_plies}}$, $\gamma = 0.98$),
     directly penalizing wasteful wandering and rewarding decisive wall conservation.
     Sample weight: 6.0.
  Fine-tuning proceeds from `race512-search10-ft` weights with reduced learning rate
  ($\eta = 1.5 \times 10^{-5}$, cosine decay to $1.0 \times 10^{-6}$) and 14 parallel CPU
  worker threads, preserving existing strengths while correcting tactical blind spots.

- **Local teaching and external benchmark lab (2026-09-15)**: The local
  runners provision pinned Titanium and Claustrophobia sources in the ignored
  `external_bots` directory. The benchmark uses paired openings and a complete
  referee. It records failures separately from draws, hashes executables,
  weights, openings, and the bridge, and resumes only a matching experiment.
  Claustrophobia keeps its upstream MCTS. A local Rust bridge sends batched
  inputs to a persistent Python TorchScript process. The bridge is practical
  on Windows without an MSVC libtorch build. Every external benchmark uses a
  fixed clock per move. The bridge calibrates a bounded MCTS search, then
  waits until the clock ends. It records the clock, search time, and simulation
  count for each Claustrophobia response. A simulation-only result is invalid
  for a strength claim.

  Teaching can combine direct old-NNUE and Claustrophobia targets with both
  search teachers. Policy and value source weights are independent. Signed
  outcomes use optional temporal discount. An optional ZQuoridor search target
  provides bootstrap supervision. Caches include data, teacher, bridge, and
  settings hashes. Train and validation use complete trajectory groups. The
  pipeline excludes frozen benchmark states and cross-split duplicate states.
  Existing 27-byte self-play records remain excluded because their position
  frame is ambiguous.

  `select_replay_disagreement.py` supplies the selective search path. It ranks
  completed direct-replay rows by Jensen-Shannon policy divergence plus value
  difference, then emits canonical V3 snapshots. Both deep-search relabelers
  consume the identical snapshot file. `build_search_priority_dataset.py`
  checks IDs, blends their targets, and produces the usual `dataset.npz` with
  a larger weight for teacher disagreement and a smaller weight when a teacher
  changes its best move between requested budgets. Snapshot searches have no
  repetition history, so they supplement rather than replace trajectory
  teaching for repetition positions.

  The first experimental network adds 102 sparse race inputs to the existing
  354. These inputs encode distance margin, wall-stock margin, and a compact
  race and resource interaction. They use the existing cached path distances.
  The default engine remains 354 inputs and 256 hidden units. Candidate builds
  opt in through compile definitions and have an architecture manifest. Native
  incremental checks and Python to native parity checks cover the extension.

  A small mixed-teaching campaign completed all four teachers, three candidate
  architectures, and paired external games without protocol failures. It did
  not produce a strength claim. The production weights remain unchanged. The
  larger campaign reserves independent screening, confirmation, and external
  books before data assembly. It requires 100 completed pairs for a reported
  strength-ready result and 400 pairs for the baseline confirmation.

- **Wall-quiescence NNUE fixture refreshed (2026-09-15)**: The frozen deep
  NNUE values in `tests/wall_qextension_reference.inc` used a predecessor
  weight blob. The current production blob changed in a later promotion. The
  heuristic reference remained exact, while every NNUE entry diverged. Refresh
  the eight NNUE values against the current tracked production weights. Do not
  interpret this fixture change as a wall-quiescence tuning result.

- **Browser zero-wall freeze hotfix (2026-09-11)**: the exact empty-handed
  race cache used 1024 lazily allocated slots. Each slot stores four arrays
  for 13,122 states and uses approximately 128 KiB. A browser worker could
  therefore retain more than 128 MiB in this cache alone near the final-wall
  transition. The native engine keeps 1024 slots. Emscripten builds use 128
  slots, which caps these arrays at approximately 16 MiB without changing
  solver correctness. The web worker facade now also resolves pending requests
  when the worker stops. A failed worker search no longer falls through to the
  same long search on the browser main thread.

- **Opening plies removed from the policy loss: NO measurable Elo
  (2026-08-26)**: in montecarlo self-play the temperature window runs no
  search. It samples the move from the policy head itself
  (`selfplay.hpp:474`) and then records that sampled move as the policy
  target (`selfplay.hpp:533`). The policy head therefore trains to imitate
  its own output. On gen6 that window covers 39.8 percent of every shard,
  and `MC_TEMP_OPENING` was 1.00, which flattens the softmax and makes the
  sampled move worse on average than the head's own argmax.

  The fix excluded those samples from the policy loss and kept them in the
  value loss. A new field was not needed: self-play writes one game for
  each `fwrite` call, so the per-game ply index is recoverable from the
  existing shards (`ply_index` in `read_selfplay.py`, tested by
  `training/test_ply_index.py`, 100 percent mover alternation over 9000
  games).

  Training ran 120 epochs over 66,458,326 positions from 469 shards. The
  mask removed 26,725,366 positions (40.2 percent) from the policy loss.
  Best `val_loss` was 1.9028 at epoch 114.

  The arena measured 1000 games at 150 ms for each move, with the same
  local code on both sides so that the weights were the only variable.
  The result was 466 wins against 464, with 70 draws: **+0.7 Elo, margin
  +/-20.8, inconclusive**.

  The lesson: removing a self-referential policy target changes nothing
  measurable at this time control. Effects larger than approximately 20
  Elo are ruled out. Do not repeat this experiment. A stronger test needs
  a different intervention, not more games of the same one. The visit
  distribution proposed in
  `docs/superpowers/specs/2026-08-25-policy-visit-distribution-design.md`
  is a different change, because it replaces the target instead of
  deleting samples, so this result weakens its case but does not settle
  it.

- **MCTS endgame wandering (2026-08-24, branch
  `claude/zquoridor-mcts-bug-hf6uqv`)**: a user reported that the engine
  spends its walls fast, then shuffles its pawn sideways forever in a won
  race. The position was rebuilt exactly from the reported game and lives
  in `benchmarks/repro_wander.cpp`. Player 0 is the engine with 0 walls and
  5 steps to the goal. Player 1 holds 8 walls and needs 13 steps. All four
  DIST counters match the report. At production settings the engine answers
  f6, g6, f6, g6 and holds its own distance at 5 for six moves while the
  opponent closes from 13 to 6.

  The measured cause is the WL head, not the tree. `benchmarks/diag_wander.cpp`
  prints a win probability of 0.137, 0.153, 0.191 and 0.281 for pawn
  distances 5, 4, 3 and 2 in that position, and it only separates the moves
  at distance 1. With the wall stocks swapped, 8 against 0, the same head
  reports 0.997 to 0.999 for every distance. The head therefore reads the
  remaining-wall counts and is almost blind to the pawn race. The policy
  head stays correct and gives the advancing move a prior of 0.73.

  The hybrid amplifies the blind spot. Leaves are the raw net value at
  `leafDepth = 0`, so no leaf ever calls `winner()`. `AvgBlend` averages
  20000 near-equal leaves, the Q values of the root moves land within 0.02
  of each other, their order is inverted against the truth, and `MaxVisits`
  then picks a shuffle. Instrumentation showed the tree reaches ply 17 to 21
  in that position and still creates ZERO terminal nodes, so an MCTS solver
  would have nothing to prove. Pure alpha-beta wanders in the same position
  too, only less: it scores +2 against +1 over seven moves.

  Two candidate fixes were measured and both failed:
  1. `BackupMode::MinimaxHard` plays the repro position perfectly at every
     setting tried, even at a node budget of 2000. It is still NOT the fix.
     It lost the arena 0 wins to 56 in 60 games, approximately -585 Elo at
     200ms, and it also lost the 40-position corpus of
     `benchmarks/bench_mcab_endgame_progress.cpp` (mean progress 5.650
     against 6.625, 5 wandering positions against 0). This repeats the
     earlier rejection under "Rejected" below, without the FPU confound.
  2. An endgame alpha-beta leaf gated on the COMBINED wall stock at or below
     8 also fixes the position, but it fires through most of the game and
     cuts the node rate by 56%. It measured -88.7 +/- 86.7 Elo over 60
     games at 200ms.

  What shipped is `McabParams::endgameMoverWallThreshold`, ON in production
  at threshold 0 since 2026-08-24. It gates the alpha-beta leaf on the wall
  stock of the SIDE TO MOVE at the root. At threshold 0 the rule fires only
  after that side spends its last wall, which is the reported condition, and
  it costs approximately 6% of the node rate. It plays the repro position
  perfectly. It measured +17.4 +/- 37.8 Elo over 300 games at 200ms.

  Read that number as neutral, not as a gain. The interval spans
  approximately -20 to +55, therefore 300 games do not exclude a small
  loss. The rule is on because it removes a reported, reproducible failure
  at no measured cost. Re-measure with more games before you quote it as an
  improvement.

  Two consequences of the default change. Self-play data generation now
  uses alpha-beta leaves in wall-poor endgames, so `.bin` files recorded
  after this date differ from earlier ones in that regime. The WASM and GUI
  build inherits the new default through `McabParams{}`, but only after
  `build/build_wasm.sh` regenerates the bundled HTML.

  `tests/test_mcab_endgame_leaf.cpp` pins the exact threshold, the gate and
  the fixed behavior. Arena flags are `--e1-mcab-endgame-mover-walls` and
  `--e1-mcab-endgame-leaf-depth`.

  Durable lesson: a position where the engine plays badly is not proof that
  the search is at fault. Print the value head across the moves in question
  before you change the tree. Here the tree was doing exactly what a correct
  PUCT does with a flat, wrong value function.

  Open item: retrain the WL head on wall-poor endgames. The head cannot
  currently separate a won race from a lost one once a side runs out of
  walls, and no search rule fixes that at the source.

- **Arch-aware Linux builds (2026-08-23)**: `build_bench.sh`,
  `build_selfplay.sh` and `build_qtp.sh` add `-mavx2 -mfma` only on x86-64
  (`uname -m`); every other architecture compiles with `-march=native`
  alone, which enables NEON on ARM (AArch64 and AArch32 alike). Before
  this change, a non-x86 `g++` rejected the two flags and the scripts
  failed outright. Deployment note that came out of the first AArch64
  server install: a weaker engine there usually means missing NNUE
  weights (the QTP binary then runs `evalSimple` and prints a stderr
  warning) or a lower time budget, not the ISA itself.
- **Race-solver audit round (2026-08-23, branch `inv/race-fuzz`)**:
  adversarial verification of `src/endgame_race.hpp` against an
  independent brute-force oracle (`tests/race_oracle.hpp`: from-scratch
  successor generation, naive win-set fixpoint, separate DTM relaxation).
  Findings and outcomes:
  - **F1, fixed**: `raceDisjointGate` computed its ply arithmetic from raw
    BFS distances without guarding degenerate values. For direct callers of
    the public inline utilities a pawn already on its goal row (rawDist 0)
    produced dtm -1, and a sealed pawn (rawDist -1) flipped the winner with
    a negative dtm. No real game reaches such states (wall legality keeps
    both paths alive), so the production hook is unchanged; the gate now
    refuses when either rawDist < 1 and Service B answers instead.
  - **F5/F6/F7, documented-only**: the race budget globals are process-wide,
    so selfplay threads interfere on solver accounting (perf-only; the
    fallback path is the ordinary heuristic node); the race-hook TT probe
    accepts any EXACT entry regardless of depth (sound approximation);
    the empty-handed root branch ignores repetition history while internal
    nodes rank it first (rare in practice). See `notes_race.md` (branch
    log) for details.
  - **Camping symptom verdict** ("engine stays on the first row"): no bug.
    Survey over 2040 hands-empty roots x tiebreak on/off: about 35% of
    chosen moves keep the root-side pawn on its back two rows, and ALL of
    them achieve the oracle-best child value. Camping while lost is the
    correct maximum-delay defense; camping while won is minimal-DTM
    geometry. `endgameProgressTiebreak` ON cuts lost-side camping by about
    13% relative at zero optimality cost. In the quasi-endgame (no solver)
    camping-while-ahead appears at short budgets but mostly turns into
    wall trades that add +1 to +4 opponent plies under deeper search.
  - **Permanent coverage**: `tests/test_endgame_race_fuzz.cpp` wired into
    both build_tests scripts as entry [16/16] (about 12 s at -O2): 8468
    oracle comparisons, 33872 root-optimality checks across all four
    toggle combos, budget-fallback and cache-eviction stress, degenerate
    probes including 600 randomized ones, and a compact engine-vs-engine
    differential where every decisive game must end in exactly the
    predicted DTM. `benchmarks/bench_race_differential.cpp` scales that to
    366 games standalone (360/360 exact lengths).
  - **Drift fix**: `build/build_tests.sh` still compiled the deleted
    `lazy_acc_parity.cpp` (the Linux suite aborted there) and lacked three
    newer tests; both build_tests scripts now list the same 16 entries.

- **Production adoption round (2026-08-23, branch `prod/findings-2026-08`)**:
  merged `inv/qsendgame-ext`, `inv/ab-policy`, and `inv/contempt-wandering`
  into one integration line, then flipped ONE production default:
  `endgameProgressTiebreak` is now ON (see the setter comment in
  `src/search.hpp`). Everything else ships unchanged: MCAB stays the
  production search with its 20000-node budget; contempt stays -30;
  wall-quiescence caps stay at the old constant values. Validation for
  the default flip: full correctness suite green on the integration tree
  (staging 0 divergences, LMR/PVS agreement 97/100 and 58/58 decisive,
  contempt/repetition T1-T5, mcab core/dispatch/phase9), plus a 600-game
  arena vs `main` at 150 ms/move, MCAB + NNUE both sides:
  **+2.9 +-26.9 Elo (47.2% vs 46.3%, 39 draws) -- statistically neutral**,
  which is the expected result: the tie-break only reorders EXACTLY equal
  solver values, so the payoff is behavioral (loser backward moves
  47% -> ~27% in wall-less endings), never a value change. Rejected for
  production after measurement (see entries below): TT-clear per move,
  parity-anchored race draws, low-wall quiescence bonus, policy-history /
  policy-LMR / policy-LMP tricks, two-stage AB pre-ranking. All rejected
  features remain available as runtime knobs for future retests.
- **inv/ab-policy round (2026-08-23, branch `inv/ab-policy`)**: asked
  whether the engine should rely less on the MCTS side of MCAB. Answer:
  no. Measured on the same binary with runtime knobs (branch has the
  toggles, all default OFF):
  - Node-budget curve vs pure alpha-beta (400/300 games per point,
    NNUE): at 40ms/move the hybrid wins by +54.3 ±33.8 Elo at the
    production 20000-node budget and never hits that budget (about 3.9k
    simulations fit in 40ms; fully time-bound). At 200ms/move it wins
    by +97.4 ±40.0. Budgets below about 10k nodes LOSE strength at
    200ms (-92.5 ±40.0 at 2k) because the engine stops early with time
    left. The node budget is a ceiling, not a tuning knob for speed.
  - Compute map (`benchmarks/map_compute.cpp`): policy passes are only
    about 7 percent of hybrid wall time; leaf evals about 8 percent;
    the MCTS loop itself (per-expansion move generation plus PUCT over
    up to 131 children) is about 84 percent. A quantized policy pass
    costs 790ns, roughly one leaf eval, so per-node policy inside
    alpha-beta is affordable since the vectorized inference work.
  - Cheap policy-inside-AB tricks are all null results at both 40ms
    and 200ms (each 300 to 400 games, Elo margin about +/-34 to +/-39):
    policy-seeded history (B) +10.4/-1.2, policy-scaled LMR (C)
    -1.7/-5.8, policy-mass LMP (D, base 0.15) -11.6 at 200ms. Policy
    LMR doubles nodes-to-fixed-depth (+99 percent at depth 8); the
    others cost or save under 10 percent of nodes.
  - Two-stage root (E: rank root children with shallow AB, keep top-k
    for the MCTS) is REJECTED: -96.2 ±39.6 (depth 2, top 8) and
    -398.4 ±65.0 (depth 3, top 12, truncated ranking) versus
    production MCAB. Cutting the root starves PUCT exploration, and a
    time-truncated ranking drops mostly wall candidates.
  The toggles stay in `search.hpp`/`mcab.hpp` default-off, pinned by
  `tests/test_policy_ab.cpp` (defaults bit-exact in both eval modes).
- **Speed round 2026-08-22 (+194.6 Elo total)**: NNUE forward passes
  vectorized (branchless int32 accumulation, bit-exact integer math),
  `PlayerPathCache` slimmed 740B -> 190B, quiescence NNUE stand-pat exits
  before BFS, TT prefetch, fast sigmoid/exp for MCAB inference
  (`mcabFastExp`; selfplay recording stays exact). Details are in git
  history (`perf/speed-elo-100`, merged).
- **Rejected -- do NOT re-attempt without new evidence**:
  - Direction-mask BFS (precomputed blocked/goal bits): -18% nps; setup
    cost dominates because caches make most BFS calls short.
  - `leafDepth >= 1`: catastrophic both before AND after the 2026-08-22
    speedups (retested at 406 games: about -250 Elo). Direct NNUE leaves win.
  - Progressive widening: -6.9 ±20.7 Elo (measured pre-speedup; revisit
    only alongside SPSA).
  - Minimax-hard backup, FPU > 0: lost to `AvgBlend` / FPU 0.0.
- **Profiler pitfalls (Windows/MinGW)**: use `benchmarks/profile_mcab.cpp`
  as template; cache module base/ImageBase BEFORE starting the sampler
  thread (GetModuleHandleA can deadlock on the loader lock against a
  suspended main thread); link with `--disable-dynamicbase
  --image-base=0x140000000` so nm addresses match runtime RIPs.
- **Repetition semantics**: wall moves split the repetition horizon (walls
  never disappear); `push(hash, irreversible)` exploits this. The subtle
  case (post-wall position recurring via pawn cycles) is pinned by
  `tests/test_repetition_diff.cpp`.
- **Wall-quiescence extension sweep (2026-08-23, inv/qsendgame-ext)**: the
  compile-time `QS_MAX_EXTRA_PLIES` bound became instance params
  (`qsMaxExtraPlies`, plus an endgame rule `qsLowWallsBonus` when
  `qsLowWallsThreshold >= wallsLeft[0]+wallsLeft[1]`). Defaults are
  bit-identical to the old constant (frozen reference values in
  `tests/wall_qextension_reference.inc`; setters resize `nnueAccStack`
  and quiescence clamps itself to free stack space, so raised caps can
  never write past the accumulator stack). The endgame rule was measured
  over bonus {1,2,3} x threshold {2,4,6} from low-wall corpus positions at
  100 ms/move: no significant Elo change vs defaults in heuristic or NNUE
  mode (best pooled result +8.4 +-25.6 at 700 games), despite only ~11%
  extra nodes to depth 10. Defaults stay production; knobs remain exposed
  (`--qs-low-walls-bonus/--qs-low-walls-threshold`). Full data:
  `data/wallext/SUMMARY.md` (untracked logs alongside).
- **Silent printf truncation (2026-08-23)**: `%u` applied to an
  `(unsigned long long)` value prints only the low 32 bits -- this is how
  the first frozen wall-extension reference file stored truncated wall
  bitboards and became unreplayable. Generators that emit state
  coordinates must round-trip every row through reconstruction and assert
  equality (see `tools/wallext/gen_reference.cpp`).
- **Contempt / wandering investigation 2026-08-22** (`inv/contempt-wandering`
  worktree; full data in that branch, `investigation_data/`): the reported
  pawn "wandering" in near-endgames is NOT caused by contempt. Sweep of
  `setContempt` over {-60,-30,-15,-5,0} moved the backward-move rate by
  less than measurement noise in both eval modes and both time controls;
  head-to-head -30 vs 0 gave -44 +/-76 Elo (n.s.). Findings, by share of
  the symptom: (1) In wall-less endings the LOSING side makes ~47%
  backward moves even with exact solver values -- among tied max-DTM
  losses the empty-handed root branch falls back to move-generation
  order. This is game-theoretically optimal resistance, but it looks
  like shuffling; `setEndgameProgressTiebreak(true)` reorders only
  exactly-equal children and cuts it to ~27% with identical results
  (Elo -12 +/-60 vs default). All-draw roots never occurred (0/290).
  (2) The persistent TT carries path-dependent repetition scores across
  real moves; clearing the TT per move removes nearly all of it
  (heuristic mode: repetition-drawn games 41/120 -> 0/120; NNUE mode:
  backward moves 0.125 -> 0.081) but costs about -71 +/-78 Elo from lost
  TT reuse -- rejected for production. (3) Confirmed sign inconsistency:
  race-solver draws use `contempt` from the node mover regardless of ply,
  while repetition draws anchor to root parity; provably unable to flip a
  move choice inside the solver regime (wins/losses are ~1e5 apart), so
  left as-is behind `setParityAnchoredRaceDraw`. Depth >=7 fixed-depth
  searches do not finish near-endgame positions within sane budgets
  (>20M nodes at depth 8); production time controls reach depth 5-6 there.
  New test: `tests/test_contempt_repetition.cpp`; bench:
  `benchmarks/bench_contempt_wandering.cpp`.
- **QFEN lessons (2026-08-23, pinned by `tests/test_notation.cpp`)**:
  - Wall token rank is the slot row + 1 (the wall's south-west cell). An
    off-by-one here survives casual testing because every token still parses;
    it only shows up as import/export disagreement on real positions.
  - The checkable budget invariant from a QFEN alone is
    `placed + wallsLeft0 + wallsLeft1 <= 20`, NOT `placed <= wallsLeft sum`.
    A finished game legitimately exports hands 0/0 with all 20 walls placed.
    Fewer than 20 total is accepted (the editor may drop walls).
  - Small enclosures are impossible with legal walls (a 1x1 box needs two
    crossing walls; a 2x1 box needs colinear-adjacent ones). The smallest
    legal sealed region is 2x2 -- the shape the path-rejection test uses.
- **parity_check.py drift (found and fixed 2026-08-24)**: the Python half
  of the `nnue_verify` cross-check had drifted twice from `nnue.hpp` --
  first the walls-left buckets (it still read 332 of the real 354
  features), then the perspective mirroring of pawn/wall features added
  to `buildAccumulator`. Because it has no file-size check (unlike
  `quantize_nnue.py`), both drifts produced confident-looking garbage
  instead of an error. It now reads 354 features, sets the walls-left
  one-hots, and mirrors pawns/wall slots per perspective exactly like
  the C++ side. Verification rule after the fix: the int8 block must
  match `nnue_verify` digit-for-digit (integer math is deterministic),
  and the float32 block must match within summation-order noise
  (approximately 1e-6). Lesson: when `nnue.hpp` gains a feature or a
  transform, update `train_nnue.py`, `quantize_nnue.py` AND
  `parity_check.py` in the same commit.
- **PowerShell file round-trips corrupt UTF-8**: `Get-Content | Set-Content`
  mangles accented characters in this repo's scripts and adds CRLF/BOM. Patch
  committed files with Python byte I/O or the Edit tool only.
- **GUI bugs the browser test caught that unit tests could not (2026-08-23)**:
  - Tap-to-move was dead since the P2 interaction work: the board pointer
    handler passed a `{r,c}` cell object into code comparing against a
    numeric display index. Silent no-op, no exception -- only a click-driven
    Playwright test catches it.
  - Naming any `EXPORTED_RUNTIME_METHODS` in emcc turns it into an
    allowlist: `HEAPU8` silently vanished from `Module`, and every string
    export crashed only when first used. Export what you touch.
  - A `$` helper bound to `getElementById` does not take descendant
    selectors; `$('#ioFmt .on')` returned null at event time. Keep one
    lookup discipline per codebase.
  - Manual board flip was instantly undone: `doFlip` toggled `B.flipped`,
    then `syncFromEngine` recomputed it from `humanSide`. The flip now lives
    in `S.flipped` (settings) and `syncFromEngine` derives the display
    orientation from it -- derived state must have exactly one writer.
  - Deleting the last entry of the Recent Games sheet re-called
    `showRecentGames()`, which early-returns on an empty list and left the
    stale row visible. Re-rendered surfaces need an explicit empty state.
  - The analysis worker request must carry only the plies up to the review
    cursor (`pliesUpToCursor()`); replaying the full recorded line analysed
    a different position than the main-thread fallback when the user had
    navigated back.
  - Walls painted vertically mirrored since P2: `setData` stored engine-space
    wall slots while the paint and SVG-export paths index them in display
    space (pawns were converted at the same boundary, so only walls were
    wrong). A human wall clicked near their goal appeared near the engine's.
    Engine-state assertions cannot see it -- only a canvas `getImageData`
    probe at the display anchor can. Fix: `setData` mirrors wall slots like
    pawns; `QBoard.wallH/wallV` are display-space by convention.
  - `syncFromEngine` called `setData` (which renders) before
    `buildLegalSets`/`refreshHud`, so the legal dots and the side-to-move
    ring appeared only after the NEXT render. Build all board state first,
    then paint once.
  - Importing an already-finished game (QGN file/hash or editor apply)
    said "Game imported - your move" even though the position was won --
    the quiet end-check never announces. Imports now run the loud
    `checkEnd()`, so the result banner, toast and Recent entry appear.
  - `QBoard.fit` never assigned `this.S`, and `paintStatic` destructures
    `{S}` from the instance: the rounded base fill and every frame style
    drew with NaN coordinates -- silently ignored by canvas draw ops, so
    the board looked fine while NO frame style ever rendered, and only the
    beveled frame threw (createLinearGradient validates its arguments).
    Per-element painting (cells, pawns, walls) hid the hole; a canvas-hash
    sweep over every dressing option caught it in one pass.
  - The `crown` pawn style shared the `pillar` branch, so the
    distinct-shapes mapping (pillar -> crown for side 1) was a visual
    no-op -- an accessibility feature that did nothing for exactly the
    users who picked those styles. Crown now has its own crenellated
    silhouette.
- **Testing discipline for this GUI**: browser tests must drive the GUI
  entry points (`newGame()`, not `__w.newGame()`); calling C-level exports
  directly skips JS-side resets (humanSide, gameOver, clocks, level marks)
  and poisons every later assertion in the run.

---

## 4. Web GUI: NNUE weight loading

`gui_web/app.js` now calls `qr_load_nnue_weights("/data/nnue/nnue_weights_int8.bin")`
at boot. Before this change it never called the loader, so the published page
ran `evalSimple` with the hybrid MCTS off, because the hybrid requires NNUE.
That is a large strength loss, and it is the engine that produced the
reported pawn wandering: those games came from the heuristic evaluation with
pure alpha-beta, not from the MCTS.

Verified on the built bundle in headless Chrome: `qr_eval_mode_is_nnue()`
returns 1, `qr_mcab_active()` returns 1, and the console reports the load.
When the load fails the engine stays in the heuristic mode and a banner names
the path it tried, because a tooltip on a hidden checkbox is not a report.

Still open: the bundled `zquoridor.wasm` was compiled before
`McabParams::endgameMoverWallThreshold` existed, so the endgame wandering fix
reaches the native binaries only. One rebuild carries it into the page:

```bash
build/build_wasm.sh          # needs an active emsdk
cd gui_web && python3 build_standalone.py
```

Note for anyone rebuilding: `gui_web/zquoridor.wasm` is gitignored, so a
stale copy from another branch survives a `git checkout` and silently pairs a
mismatched binary with `zquoridor.js`. The symptom is a bundle that boots but
returns nonsense (`qr_walls_left` gave `undefined`). Delete the file, or
recover the matching one with `gui_web/extract_wasm_from_bundle.py`, before
running `build_standalone.py`.

---

## 5. Module Reference

- **`src/rules.hpp`**: `State`, bitboard walls, move generation, four BFS
  variants sharing one engine, `evalSimple`, BFS node cache
  (`PlayerPathCache`) + cross-node cache (`PlayerPathCacheTable`, ~12MB),
  `RepetitionTable`.
- **`src/dsu.hpp`**: rollback DSU for cheap wall-legality proofs.
- **`src/cat.hpp`**: Corridor Attention Table (wall ordering heat).
- **`src/search.hpp`**: pure alpha-beta (TT, killers/history, LMR+PVS,
  wall quiescence, policy ordering); every heuristic has a runtime toggle.
- **`src/endgame_race.hpp`**: exact solver for wall-less pawn races;
  read its header comment before touching it.
- **`src/nnue.hpp`**: incremental quantized accumulators, forward passes,
  feature definitions.
- **`src/mcab.hpp`**: production hybrid PUCT MCTS; `McabParams` holds all
  production values.
- **`tools/arena/`**, **`tools/selfplay/`**, **`tools/spsa/`**,
  **`training/`**: strength testing, dataset generation, parameter
  tuning, NNUE training.
- **`tools/wallext/`**: wall-quiescence extension experiment kit
  (deterministic low-wall corpus, frozen-reference generator with
  round-trip proof, nodes-to-depth bench, single-process pairwise arena,
  match orchestrator `run_wallext.ps1`).

## 6. Web GUI (`gui_web/`)

Premium interface per `gui-premium.md`. Phases P0-P5 (tokens, canvas board
`board.js` QBoard, wall/pawn interaction, HUD, layouts, play tab) landed
2026-08-22. P6 + P7 landed 2026-08-23; P8, P8b, P8c, P9 and P10 completed
2026-08-23 (same day), making the plan fully implemented:

- **P6 engine surface** (`engine_wasm.cpp`, exports synced in
  `wasm_args.rsp` and both `build_wasm.*` scripts -- keep the three lists
  identical):
  - Full game history in C (`g_histStates`/`g_histMoves` + cursor):
    `qr_goto_ply`, `qr_truncate_history`, `qr_ply_*` getters. Every live
    mutation records a ply; the JS snapshot-diffing of the old GUI is gone.
    `qr_goto_ply` rebuilds the repetition table from real history and resets
    the MCTS tree (tree reuse assumes one continuous line).
  - Scratch position shared by analysis/editor/blunder-check
    (`qr_scratch_from_live/from_ply/reset`, `qr_scr_*` getters). Analysis can
    never mutate the live game.
  - Multi-line analysis `qr_analyze(maxDepth, timeMs, lines)`: line 1 is a
    full search at the scratch root (the same move the game engine would
    play); lines 2+ come from independent searches of child positions after
    candidates ordered by a depth-2 pass. Honest but only line 1 is proven
    best -- documented here because the UI does not say it. PVs are rebuilt
    from the TT by the new public `Negamax::extractPv` (`src/search.hpp`).
    Runs on its own `Negamax g_anEngine` so the game TT stays warm.
  - Editor ops on scratch (`qr_edit_set_pawn/wall/walls_left/turn`,
    `qr_edit_validity` bitmask, `qr_edit_apply`). Wall placement refuses only
    physical conflicts (occupied/crossing/colinear); path problems go to the
    bitmask so the user can build freely and read what is wrong.
  - QFEN import/export (plan section 16.1): canonical export sorts walls by
    orientation/row/column, token rank = slot row + 1 (south-west cell);
    import validates before applying and reports the exact failing token via
    `qr_last_error`. Pinned by `tests/test_notation.cpp` (round-trip over
    random playouts + diagnostics table), which includes
    `gui_web/engine_wasm.cpp` directly to reach the anonymous namespace.
  - `_malloc`/`_free` and the `HEAPU8` view had to join the export lists:
    naming any `EXPORTED_RUNTIME_METHODS` turns it into an allowlist, and the
    JS string bridge reads QFEN bytes through `Module.HEAPU8`.
- **P7 analysis tab** (`app.js` section 14): ENGINE ON/OFF, depth select with
  infinite mode (iterative deepening loop capped at depth 22), 1-5 PV rows
  with eval chip / moves / opponent-distance delta badge / line preview on
  the board (walls translucent, pawn steps numbered), eval graph canvas with
  scrub-to-jump and blunder dots, ply navigation wired to `qr_goto_ply` plus
  `,` `.` `Home` `End` keys and an `A` engine toggle, move log rendered from
  recorded plies with click-to-jump, takeback that rolls back to the human's
  previous turn, hint (single-line analyze drawn as ghost/dot for 4 s),
  blunder check with cancel + progress + per-side accuracy card (win-prob
  drops 0.06/0.13/0.25 -> `?!` `?` `??`; `!` when the played move matches the
  engine best with a decent score).
- **P8 serialization & editor**: full QGN exporter (PGN-shaped headers,
  `{[%ev ...]}` comments from stored analysis scores, blunder symbols,
  result); game importer that strips headers/comments/numbering and applies
  tokens one by one through the legality-checked C surface, stopping at the
  first bad token with its index; dialect normalization for
  orientation-first (`Ha5`, `V f3`), coordinate-pair (`c6-d6`) and bare move
  lists; shape-based routing (`routeImport`) between QFEN and QGN. Text I/O
  modal (format toggle, copy/paste/load/download, diagnostics), drag-and-drop
  onto the board, file picker, `#qfen=`/`#qgn=` URL hash load plus Copy link,
  autosave of the live game to `zq.game` (debounced) with a 10 s Resume chip,
  and a 20-game `zq.recent` ring with Load/Copy/Delete sheet. Editor tab: tool
  palette, wall-budget steppers, side-to-move switch, live validity strip fed
  by `qr_edit_validity`, Apply gated on validity, Copy/Paste QFEN; while the
  pane is open the board renders the SCRATCH position.
- **P8b personalization**: settings schema v1 per plan section 11 with merge
  migration and corrupt-blob reset; presets Classic / Premium Dark / High
  Contrast / Minimal (touching any option flips the chip to Custom); 8 board
  themes including Carrara Marble (deterministic procedural veins drawn once
  into the static layer, mulberry32 seed) and Noir (value-separated pawns;
  light variant redefines wall tokens -- found missing by the contrast gate);
  board dressing switches (frame none/hairline/gilded/beveled, wall finish
  flat/beveled/glossy/etched, cell separation grooves/flat/inlaid,
  coordinates off/edges/all, board scale slider); accent colour presets plus
  custom picker with derived `--gold2`/dim/glow; 6 pawn styles (adds
  pawnChess and beacon), pawn size and shadow, distinct-shapes toggle
  (automatic in noir); sound packs wood/modern/marble/silent as data tables
  with per-event toggles, volume slider and Test button; haptics levels;
  motion full/reduced/off with speed multiplier over the duration tokens;
  density and text-size switches; handedness mirroring dock/toasts.
- **P8c image export**: PNG via `QBoard.renderExport` into a detached fixed
  size canvas (transparent-background, coordinates and footer-wordmark
  options) and SVG via `QBoard.toSVG`.
- **P9 accessibility**: `tools/gui/contrast_check.py` gate (wall/cell and
  wall/groove >= 3:1 for all 8x2 theme combinations, text >= 4.5:1) runs
  inside `build_standalone.py` and fails the bundle on regression; SR live
  region announcements after every ply; keyboard help overlay (`?`), full
  §14 key map including arrow-key pawn movement with Shift diagonals and
  straight-jump continuation; focus-visible outlines kept.
- **P10 standalone**: Google Fonts downloaded at bundle time into the
  committed `gui_web/fonts_cache/` and inlined as base64 WOFF2, so the
  single-file bundle needs no network; graceful fallback to the `<link>` if
  the cache is empty and the network is unavailable.
- **Analysis worker** (`worker.js`, plan section 12): a second WASM module
  instance receives self-contained requests (root QFEN optional plus the
  recorded packed plies), replays them onto its scratch and returns line
  data, keeping the main thread free during analysis. The main-thread
   slicing path remains as the silent fallback (file:// bundles cannot spawn
   the worker). Play-mode engine turns still run synchronously on the main
   thread -- tracked below as remaining work.
- **Post-commit hardening + classic board pass (2026-08-23)**: after the P0-P10
  commit, user reports ("engine plays for both sides", "walls cannot be
  placed", board looked worse than the old GUI) led to a turn-machine and
  input hardening round plus a board visual rework:
  - Turn ownership is now structural: `scheduleEngineTurn(delay)` is the
    single owner of the engine timer (a new schedule cancels any stale one),
    and `engineTurn` refuses to move unless it is really the engine's side --
    the engine can never move for the human even with a duplicated timer.
    Clicking while the engine thinks gives a throttled toast instead of
    queueing input.
  - Wall gestures never fail silently: `snapGhost` always snaps to the
    nearest anchor and reports ok/assisted/bad with a reason; presses near a
    cell center route to the pawn handler, not the wall gesture; `#board`
    has `touch-action:none` so touch drags commit instead of scrolling.
  - The default obsidian look moved back toward the old board: flat uniform
    cells with thin light gridlines, hairline frame, slim ivory walls
    (`--wall:#ece7da`, was gold), subtle goal markers and softer
    highlights. All dressing options keep working.
  - The mcab play budget is node-based (~20k nodes), so engine replies are
    usually faster than the level's nominal time -- tests must not assume
    `engineThinking` stays true for the level duration.
- **Dock press-and-drag was dead (found and fixed 2026-08-24)**: the M1
  gesture (press a wall on the dock, drag onto the board, release) never
  placed anything. `setPointerCapture` retargets every pointermove to the
  dock element, but the dock handler early-returned once `_dragging` was
  set -- so the ghost froze off-board and the release had nothing to
  commit. The board drag (press directly on a groove) was unaffected,
  which is why earlier suites passed. Fix: keep forwarding snapGhost on
  every move while the drag is engaged (`app.js`, dock pointermove).
  Pinned by the new `gui_web/test_deep_click.py`, an 80-check click
  suite covering pawn destinations, wall refusal reasons, keyboard arms,
  confirm chips, settings persistence, worker-off analysis fallback,
  editor validity gating, QFEN/QGN diagnostics, resume chip and recent
  games. Geometry note for assist tests: pressing dead-center on an
  illegal anchor leaves all neighbours at exactly 1 U, outside the
  0.9 U assist radius -- press at approximately 0.48 U toward the legal
  neighbour instead.
- **Per-feature sweep** (`gui_web/test_gui_features.py`, 2026-08-23): one
  pass over EVERY settings control and dressing option (8 board themes,
  6 pawn styles, frame/finish/surface/coords/scale, overlay toggles, UI
  theme, sound controls, multi-PV + line preview + infinite depth + graph
  scrub, SR live region, h/v/Esc keys, level modal, PNG/SVG content,
  `#qgn=` cold load). It found the `this.S` hole and the crown/pillar
  duplication above plus a paths-overlay settings lag (fixed: applySettings
  now calls `togglePaths`). Canvas-hash comparisons need one caveat: a
  value equal to the current default (coords `edges`, pawn `disc`) is a
  no-op by definition -- start the sweep away from defaults.
- **Full game simulation** (`gui_web/test_gameplay_sim.py`, 2026-08-23):
  plays a complete game like a person -- pawn clicks, arm+click walls,
  drag walls, keyboard walls, hint, flip mid-game, takeback+redo, review
  round-trip, level cycle -- asserting turn ownership, ply parity, wall
  conservation and ghost/dots state after EVERY engine reply; then runs
  the full analysis pass mid-game (3 PVs, previews, scrub, blunder check,
  accuracy card), resumes the same game to completion (proving analysis
  leaves the game state intact), and hands the QGN to a second page via
  `#qgn=`. First run reached ply 38 with zero desyncs: the turn machine,
  wall placement and analysis isolation hold under real play.
  Script-writing lessons that cost time:
  - A "nudge" click on the board while the engine thinks can land AFTER
    the reply arrives and then COMMIT a real move -- race, not bug. Nudge
    clicks in tests must target elements outside the board.
  - `B.dots` are display cells. `engPawnToDisp` is self-inverse, so a
    double conversion cancels only when another `cell_pt` follows; an
    inline `cellCenter(engPawnToDisp(d))` lands on the mirrored rank.
  - The return-to-game chip lives in the analysis pane; reviewing from
    the play tab goes back with the `End` key. The level chip opens the
    NEW GAME modal -- mid-game level changes are the `S` key.

- **GUI v2 rebuild (2026-08-25)**: the interface was rebuilt after a user
  review. The complaints were a dark board that did not read as a board,
  pawns that jumped between cells, a takeback button that did nothing, and
  two colored streaks across the middle of the board. The rebuild keeps the
  engine plumbing untouched. `engine_wasm.cpp` and the WASM exports did not
  change.
  - **Look**: a warm wooden board inside the dark Zchezz chrome. The board
    tokens moved to a `wood` default: tan cells, a walnut frame, dark walnut
    walls. `walnut` became a dark board with ivory walls, because its old
    values failed the contrast gate at 1.40 against a required 3.0. Nine
    board themes now exist.
  - **Layout**: one player strip above the board and one below it. Each
    strip carries the name on its outer edge, then the clock, the wall pips
    with a count, and the distance to goal. The old dual HUD bar, the
    desktop HUD rail and the wall dock are gone. The useful buttons sit in
    one Zchezz-style row under the board. There is one breakpoint at 900px.
  - **Wall placement without a mode**: `boardHit()` in `app.js` classifies a
    pointer position as a cell body or as a groove, and names the
    orientation of that groove. A hover over a groove paints a preview
    through the new `QBoard.setHover`. A press starts the drag, and a drag
    of more than `0.45*C` across the other axis flips the orientation. The
    `H` and `V` buttons only force an orientation. They are no longer
    required. The forced orientation expires after 6 seconds.
  - **Anchor geometry**: `anchorFor(o, px, py)` uses floor along the wall's
    own long axis and round across the groove. A first version subtracted 1
    from the vertical row. That version resolved the exact center of a
    groove crossing to the anchor above the correct one, and
    `test_deep_click.py` caught it through 7 failed wall checks.
  - **Pawn animation**: `QBoard.animateMove` tweens a pawn over 200ms and
    arcs a jump. The tween starts before the state sync paints, therefore
    the piece slides from the cell it left. A `requestAnimationFrame` loop
    runs only while an animation is active.
  - **Engine off the main thread**: `worker.js` accepts a new `bestmove`
    command. The worker replays the recorded line into its own live game and
    runs `qr_engine_move`, which is the hybrid search. The analysis path
    cannot serve this: `qr_analyze` runs pure alpha-beta on the scratch
    position, so it would silently change how the engine plays. Measured at
    the Titan level, the worst main-thread stall during an 8 second search
    fell to 5ms. A game from a custom position still searches locally,
    because the worker has no replay root for it.
  - **Takeback**: `#btnTakeback` was never wired to anything. The button is
    wired now, and six further defects around it are fixed. The guard
    refuses a takeback during review. `clockHist` gives the time back.
    Entries in `AN.scores`, `AN.annots` and `levelMarks` past the new end
    are dropped, because those keys are ply indexes and would otherwise
    attach to different moves. The eval bar resets. `engineGen` invalidates
    a search that is still running. `W.scratchFromLive()` restores the
    scratch position that the search loop overwrites.
  - **Path overlay**: the old version walked greedily in engine coordinates
    and handed engine cells to a renderer that expects display cells.
    Therefore both lines came out mirrored and crossed in the middle of the
    board. Those were the streaks in the user report. `shortestPath()` is
    now a real BFS, `recomputePaths()` runs on every state change, and the
    cells are converted with `engPawnToDisp`.
  - **Clock**: the default is 5+3 instead of none, so that the strip has a
    clock to show. `addIncrement()` credits the increment, which the schema
    declared but no code applied.
  - **Modal buttons**: `.btn` forces a 36px width. Every button inside a
    modal was clamped to that width, so the labeled buttons overlapped and
    swallowed each other's clicks. `.modal .btn` now sizes to its label.
  - Tests: all five browser suites pass. `test_deep_click.py` loads the
    bundled `zquoridor.html`, therefore `build_standalone.py` must run
    before that suite reports on current code.

- **GUI v2 review pass (2026-08-25)**: a second user review asked for a
  denser HUD and a desktop layout closer to a chess interface.
  - Each player strip is now one line: name, clock, wall pips with a count,
    then the distance to goal as a bare number behind a hairline. The engine
    level left the strip, because the header chip already names it.
  - Desktop moves the status line and the button row into the side rail, so
    the board keeps the full height of its column. `layoutReflow()` moves the
    same nodes across the 900px breakpoint. There is one copy of each
    control, never two.
  - The move log follows the same reflow. On a phone it fills the space under
    the button row, which was empty before. It is selectable text in both
    layouts.
  - The evaluation bar is a vertical bar on the board's left edge with a gold
    balance line and a numeric readout in a 30px gutter. The gutter is
    desktop only. On a phone the bar is a 4px hairline and the number stands
    down, because every pixel of width is board there.
  - `QBoard.fit()` measured `clientWidth`, which counts the zone padding, and
    it ignored the evaluation bar's share of the row. The board was therefore
    sized larger than its box and spilled over its neighbours at
    width-limited desktop sizes. It now measures the real content box.
  - The player strips align to the board, not to the column. The evaluation
    bar and its gutter sit inside the board zone, so a centred strip was
    visibly off by half of that width.
  - `.btn` forces a 36px width. The two wall buttons carry a live count, so
    they size to glyph plus number. The corner badge was too small to read.
  - `packedToTok()` read the low byte of a packed move for both kinds of
    move. A pawn move carries its destination in bits 16 to 23, so every pawn
    move in an analysis line printed as `a1`, and every wall printed as a
    bare `H`. The decoder now matches `plyNotation`.
  - The move count is a reading, not a control. Recent games is reached
    through the header menu, which already lists it.

- **GUI v2 element audit (2026-08-25)**: a per-element measurement pass, after
  a review found faults that a per-category read had missed.
  - **Levels are named after the chess pieces**, in order of material value:
    Pawn 50 ms, Knight 150 ms, Bishop 400 ms, Rook 1 s, Queen 2.5 s, King 8 s.
    The default is Rook, which holds the budget the old default held. Settings
    schema 2 maps each old key to the piece with the same budget. `curLevel()`
    falls back to the default, because an unknown key used to reach the search
    and throw on a missing time budget.
  - **One evaluation bar**, 12 px. It is vertical beside the board on a wide
    screen and horizontal under the board on a phone, and it is the same
    element in both. The mobile race strip is gone: it read as a second
    evaluation bar. The race meter keeps its labelled place in the Play panel.
    `setEval()` is the single writer and picks the axis; the analysis setter
    and the takeback reset both call it.
  - The wall count moved next to the player name. Beside the distance the two
    numbers read as one.
  - Every element in the board column now aligns to the board itself: both
    player strips, the status row, the button row, the move log and the
    horizontal evaluation bar, at every breakpoint.
  - Faults the measurement pass found: the vertical bar was 35 px taller than
    the board; the two wall buttons had different widths; the horizontal bar
    was indented twice; the button row overflowed a 390 px screen by 4 px
    because the settings gear was duplicated from the header; the level chip
    was a 26 px tap target; and the move counter was still a focusable button
    after it stopped being a control.

- **GUI v2 analysis and readability pass (2026-08-25)**:
  - The vertical evaluation bar is 24 px. Its readout is a small percentage in
    one fixed place under the bar, not a label that slides with the fill. The
    percentage is player 0's share, which is what the bar's geometry shows, and
    it matches the absolute-colour display convention in section 5.
  - The wall pips are 4 by 14 px with a 3 px gap. The board coordinates dropped
    to `0.32 * C` at 72 % alpha, so they label the board without competing with
    it.
  - **Evaluation left the play move log.** The play log lists moves only. The
    Analysis tab owns evaluation and has its own list, `renderAnMoveLog()`,
    with a score for every ply from `AN.scores`. A ply with no score shows a
    dash, and the list says how many are missing and that Blunder check fills
    them.
  - **Pawn drag was never implemented.** Tap to select and tap the destination
    worked, but pressing the pawn and pulling it did nothing. `QBoard.setDragPawn`
    holds the piece under the pointer and the release commits over a legal
    destination. A press that does not move is still the first tap.
  - The evaluation graph hides itself below 4 plies, because an empty plot area
    reads as a broken box.
  - `gui_web/` has a new end-to-end check that drives every feature through the
    real interface: both ways to move a pawn, all four ways to place a wall,
    takeback, hint, flip, paths, the evaluation bar, three analysis lines, the
    blunder check filling every ply, navigation, the editor, the settings modal
    and the level list. 30 checks, all passing.

- **GUI v2 graph, wall matrix and analysis export (2026-08-25)**:
  - **The evaluation graph auto-scales.** A whole game plotted over 0 to 1 is a
    flat line in the middle for every position that is not already decided,
    which shows nothing. The band is centred on 0.5 and opens only as far as
    the data needs, with a floor of `+-8 %` so that noise is not magnified into
    a swing, and a corner label names the band. A quiet game now reads at
    `+-8 %` instead of flat.
  - `--p0-soft` and `--p1-soft` did not exist in the rebuilt stylesheet. The
    graph read them for its area fill, an empty string leaves `fillStyle`
    unchanged, and the area silently never painted. Both tokens are defined,
    and `drawGraph()` falls back to the solid player colour.
  - **Two flexbox faults that swallowed clicks.** `#anGraph` could be shrunk by
    the flex column while its canvas still painted 72px, and `#anLines` could
    be shrunk below its content so its rows overflowed under the graph. In both
    cases the block that painted last covered the controls under it. Only
    `.card.grow` may take the leftover height now; every other child of a pane
    keeps its content height.
  - The Analysis tab has an EXPORT card: Copy QFEN, Copy game, Save .qgn and
    Copy link. QFEN exports the position the cursor is on, so any position in a
    game under review can be taken out. Verified: at the live end it gives
    `e2 e8 10 10 - 0 2`, one ply back `e2 e9 10 10 - 1 1`.
  - New `gui_web/test_wall_matrix.py`: 57 checks over wall placement with a
    mouse and with touch. Drag direction sets the orientation, hovering a
    groove previews that orientation, a cell body previews nothing, and every
    corner and edge anchor accepts both orientations by drag and by press. The
    asserted invariant is that the wall lands on the slot the ghost showed.

- **GUI v2 responsiveness and preference round (2026-08-26)**:
  - **A human move painted itself only after the engine replied.** `playPawn`
    applied the move to the engine but left the board arrays at the old cell.
    The slide tween therefore ended on a snap-back, and the piece reached its
    destination only when the engine reply ran `syncAll()`. A placed wall had
    the same delay. `playPawn`, `commitGhost` and the arrow-key move now call
    `syncFromEngine()` immediately after they apply the move, while the tween
    covers the repaint.
  - The default sound pack is quiet and rounded. Each tone starts with a short
    attack ramp (`atk`) and can pass through a lowpass filter (`lp`), the
    square and sawtooth voices are gone, and the default volume dropped from
    0.6 to 0.5.
  - The shortest-path overlay is opt-in only. Settings schema 3 clears a stored
    `paths` value once, and the path button shows its own on state.
  - The gold ring around the pawn that is to move is gone. The legal-move dots
    already show whose turn it is.
  - Board coordinates are half their previous size: `0.16 * C` on the edges,
    `0.12 * C` in study mode, and `0.15 * C` in the SVG export.
  - Wood-theme walls are pale rails. The tokens are now `--wall:#ecebe5`,
    `--wall-hi:#faf9f4` and `--wall-edge:#a6a49a`. `QBoard.wallDrawRect()`
    paints each beam at approximately 1.6 times its slot thickness, so the
    wall reads against the wood bed. Hit tests keep `wallRect()`.
  - The desktop gap between the evaluation bar and the board grew from 8 px to
    18 px. The board fitter reads the computed gap, so the board size adjusts
    by itself.
  - The settings modal is tabbed: Appearance, Board, Sound, Play & data. Each
    tab shows a one-line summary, rows are at least 36 px tall, and segment
    labels are full words. `wireSettingControls()` tolerates the controls of
    the tabs that are not open. The header menu groups its items under GAME,
    SHARE & FILES and EXTRAS labels.
  - The browser test suites learned the tab navigation. All seven suites pass:
    `test_gui_features`, `test_deep_click`, `test_features_e2e`,
    `test_wall_matrix`, `test_gameplay_sim`, `test_browser_full` and
    `test_standalone`.

### 2026-08-26: presentation pass

Design plan in `docs/superpowers/specs/2026-08-26-gui-refactor-plan.md`. All
seven browser suites pass after the change.

- **Settings modal opened empty (bug).** `$('btnSettings').onclick =
  modalSettings` passed the `MouseEvent` as the `tab` argument, and
  `modalSettings(tab)` starts with `if (tab) settingsTab = tab`. A
  `MouseEvent` is truthy, so `settingsTab` became the event object and every
  later lookup (`cards[...]`, `SETTINGS_TAB_HINT[...]`, the `on` class) missed.
  The first open rendered 45 characters of text; a click on any tab passed a
  real string and the same modal then rendered 459. Every modal opener is now
  wrapped in an arrow function. `modalNewGame()` takes no argument, so it never
  showed the defect.
- **Type and space tokens.** `:root` gained `--space-1` to `--space-8` and
  `--fs-xs` to `--fs-xl`. The type steps use `clamp()`, so they answer to both
  the viewport and the root font-size percentage that the text-size setting
  writes. 58 hardcoded `rem` sizes moved onto the scale. The body base went
  from `.7rem` to `--fs-md`, which measures 13.68 px at 1280 px wide against
  11.2 px before. `.btn` moved from a fixed `width:36px` to `min-width`, and
  `#controls` from `height` to `min-height`, because the larger labels
  overflowed the button row by 26 px.
- **Coordinates.** They were 8.6 px at 72 percent alpha, in a `--coord` that
  measured 2.26:1 against `--frame` in the default wood theme. Six of the nine
  board themes were below 4.5:1. The size is now `max(10, 0.235 * C)` at 92
  percent alpha, which measures 12.7 px on a 600 px board, and the six failing
  `--coord` values moved on lightness only, so the hue and the saturation are
  unchanged. `tools/gui/contrast_check.py` gained a `coord/frame` pair at 4.5,
  so the build fails if this regresses.
- **Evaluation bar.** `#boardZone > #evalWrap` carried `margin-bottom:15px` to
  reserve room for the `#evalNum` label. `#boardZone` centres its children, so
  the margin was centred with the strip and pushed the bar 7.5 px above the
  board's top edge. The label moved inside the strip, the margin is gone, and
  the bar now shares the board's top and bottom edge exactly.
- **Wall depth.** `board.js` hardcoded the shadow, the glossy highlight and the
  two etched inset lines, so a light board got a shadow built for a dark board.
  They are now `--wall-shadow`, `--wall-gloss`, `--wall-etch-dark` and
  `--wall-etch-light`.
- **Pawn.** The pawn never varied with engine strength. `LEVELS` maps a level
  to a search time budget only. The reported base was the contact shadow: a
  hard edged ellipse of radius `R*.92`, drawn at `R*.60` under the centre,
  which reads as a plinth. It is now a radial gradient that fades to zero, and
  the `disc` and `beacon` styles gained a specular highlight. The `pillar` and
  `pawnChess` styles still draw a flared foot, because that is their design.
- **Modal on desktop.** `#overlay` used `align-items:flex-end` at every width,
  so the sheet stuck to the bottom edge of a desktop window. Above 900 px it is
  now a centred dialog with four corners and a `fadeZoom` enter.

### 2026-08-26: second pass, from screenshots

The first pass worked from measurements alone. This pass worked from rendered
screenshots, captured with Playwright because the browser pane does not
composite frames. Three defects were visible that no measurement had caught.

- **The evaluation bar showed no evaluation (bug).** `setEval` runs only after
  an engine move, on takeback, and in the analysis tab. A fresh game never
  called it, so `#evalFill` kept its 0 percent start and the bare track read as
  "one side wins outright". `newGame()` and `boot()` now seat the bar at even.
  The opening position is even, so 50 percent is honest, and it leaks no engine
  evaluation during play.
- **The board frame had no depth.** The default frame style was `hairline`, a
  flat band. `beveled` already existed, with a gradient and an inner rim, and is
  now the default. The groove centre lines fade out as the cell grows: they
  exist so the grid reads when cells are small, and on a large board they read
  as an artefact instead.
- **The walls sat on the board, not in it.** `wallDrawRect` inflated the beam by
  0.32 of the groove, which left a brown margin on both sides of its own
  channel. It is now 0.46.

Also in this pass:

- The move log's empty state is centred and labelled. A bare sentence at the
  top of a 394 px empty column read as a rendering failure.
- 79 hardcoded gap and padding values moved onto `--space-1` to `--space-8`.
- The evaluation bar keeps its 24 px width, which `test_features_e2e` asserts.
  Its readout sits inside the strip at a fixed 9 px, which is off the type
  scale on purpose: the label is bounded by the rail, not by the text
  hierarchy. At `--fs-xs` it touched both edges.

### 2026-08-26: responsive sweep

The presentation passes above were verified at two viewports. A sweep over
thirteen found that the small and rotated layouts were already broken before
this work, and that the type scale made them worse. The audit counts elements
whose content exceeds their own box, and it skips scroll containers and
elements that carry `text-overflow:ellipsis`, because clipped text reports an
overflow by design.

The count went from 41 to 0. Every landscape board is also larger than it was.

| Viewport | Before | After | Board before | Board after |
| --- | --- | --- | --- | --- |
| 844x390 | 9 | 0 | 190 | 231 |
| 740x360 | 11 | 0 | 156 | 202 |
| 1000x440 | 4 | 0 | 246 | 246 |
| 900x420 | 4 | 0 | 227 | 227 |
| 375x667 | 6 | 0 | 283 | 270 |
| 360x640 | 5 | 0 | 262 | 249 |
| 1024x768 | 2 | 0 | 574 | 574 |

The causes, in order of size:

- **The landscape zone was sized from the viewport WIDTH.** The portrait rule
  `#boardZone{height:calc(100vw - 14px)}` also applies in landscape, where it
  asks for an 830px tall zone inside a 390px viewport. Landscape now sizes the
  zone from `100dvh` minus the chrome. The height must be definite: with
  `auto`, the zone measures the canvas while `fit()` measures the zone, and the
  pair collapses to the 150px canvas floor.
- **The button row never wrapped on a portrait phone.** `flex-wrap:wrap` was
  set only inside the `min-width:900px` block, so at 360px the seven buttons
  needed 341px in a 262px row and simply overflowed.
- **`#statusRow` had a fixed `height:20px`** that is shorter than the chips'
  own line box at the smaller viewports.
- Landscape also drops the pips, the STEPS label, the under-board move log
  home and the status row, and it shrinks the eval strip to 8px. Each of those
  repeats information that the visible side rail already carries.

`fit()` re-fills any height freed above it, so the last pixels cannot be
subtracted away: freeing 7px of column grows the board by 6. The residual has
to come from an element that does not feed that loop.

One caution for a later pass. A single failure of `test_deep_click`
("crossing reason named") did not reproduce: the status text was correct when
checked by hand, and two further runs passed 80 of 80. The check reads
`#status` 120 ms after `mousedown`, so it is timing sensitive.

### 2026-08-26: audit of the surfaces nobody had opened

The two passes above worked on the play screen. This one opened the rest:
phone landscape, the settings and Text I/O modals, the toast, the end of a
game, the editor in use, and the ply navigation. Every round found a defect,
which is the reason the round happened at all.

- **Phone landscape reached none of its wall buttons.** The reflow pins the
  block under the board to the board's own width, so the player strips line up
  with it. In landscape the board is limited by height and is far narrower than
  its column: the button row needed about 331 px and had 246. The remainder
  became a sideways scroll whose scrollbar is hidden, so 85 px of the row, both
  wall buttons among it, sat off screen with nothing to say so. That block now
  takes the width its buttons need, up to the column's.
- **The evaluation bar then had to stop being flush left** inside that block,
  or it sat 43 px off the board. It centres, and so does the block. Measured
  offset is 0 in landscape and 0 in portrait.
- **The move log sat in a container with no height** while the rail showed an
  empty MOVE LOG card. The reflow sent it under the board whenever the window
  was narrow, but narrow is not one-column: landscape is narrow AND two-column.
  The target now reads the rail's own computed position.
- **No ending said who won.** `.pbar.win` and `.pbar.lose` have been in the
  stylesheet all along, a green border for the winner, and `newGame` clears
  both. Nothing ever added them. One `markResult(side)` now marks both strips,
  and the four endings call it. A draw passes -1 and marks neither.
- **The ply navigation stayed lit at both ends.** At ply 0 the back pair did
  nothing and looked available, and the same at the last ply for the forward
  pair. `updateNav` now toggles `.btn.off`, the treatment the wall buttons
  already use.
- **`--muted` failed the text ratio in both themes** and nothing caught it,
  because the gate tested only `txt/surf` and only in the dark theme. It sat at
  2.54:1 in dark and 3.60:1 in light. Both clear 4.5:1 now, and the gate checks
  `txt`, `txt2` and `muted` in both themes.
- **The wall pips came back to the phone.** They were dropped below 400 px
  because the strip could not hold them. Moving the strip's gaps onto the
  spacing scale had reclaimed the width: at 360 px the row needs 300 px of a
  338 px strip. The premise had expired, so the rule went.
- **Touch targets.** Ten controls were 32 px tall on a phone. Portrait raises
  them to 44 px tall and 40 px wide. 40, because seven buttons at 44 need
  371 px and the row has 368 on a 390 px phone.

Two results worth keeping because they say what NOT to do.

- **The landscape button height stays at 28 px.** Raising it to 32 and to 36
  both make `#boardCol` and `#underBoard` overflow their own height, measured
  at 844x390 and at 740x360. It is under the 44 px a finger wants and it is
  what fits.
- **The settings groups were already consistent.** Every active item resolves
  to the same gold and every inactive one to `--txt2`. The apparent
  inconsistency was in how a screenshot rendered them.

### 2026-09-19: standalone worker offload, favicon branding, and audit fixes

A comprehensive review of the web deployment resolved three functional and visual defects:

- **Standalone worker offload without network requests.** The standalone HTML
  bundle previously requested `worker.js` and `zquoridor.js` over HTTP.
  This request caused HTTP 404 errors on GitHub Pages. `gui_web/build_standalone.py`
  now embeds the worker script as an inlined Blob URL. The main thread passes
  the compiled WebAssembly binary and weights buffer to the worker through a
  message. Background analysis and engine computation run off the main thread.
  If the page loads from a local `file:` URL, the application uses its offline
  fallback.
- **Branding and icon assets.** The application lacked a dedicated favicon.
  A new vector and bitmap icon set depicts a "Zq" monogram on a dark board.
  Wood barriers form the letter "Z". A red pawn and a vertical barrier form
  the letter "q". The builder embeds this icon as an inlined data URI in the
  HTML head. The repository also tracks `favicon.ico`, `favicon.svg`,
  `icon-192.png`, and `icon-512.png`.
- **Move counter reset and accessibility.** When a user started a new game,
  the move chip retained its previous move count until the first turn.
  `newGame()` now calls `updateMovesChip()` to reset the counter to zero.
  In addition, form inputs in settings and analysis controls now contain
  explicit `aria-label` and `for` attributes.
- **Input gating during engine animations.** The engine turn handler set
  `engineThinking = false` before piece animations finished. This allowed
  human inputs during active animations. The handler now resets the flag after
  the move animation promise resolves.

### 2026-09-19: 5-tier curriculum scaling, GPU inference worker fix, and deep search plan

- **Claustrophobia GPU inference worker fix.** `training/teachers/claustrophobia_inference_worker.py`
  and the companion binary worker checked `sys.argv[2] == "gpu"`. When callers set
  `QUORIDOR_DEVICE=cuda`, `sys.argv[2]` received `"cuda"`, silently falling back to CPU
  execution. The condition now checks `sys.argv[2] in ("gpu", "cuda")`, offloading MCTS batch
  evaluations to CUDA tensor cores on the RTX 4050 laptop GPU. VRAM consumption remains ~1.2 GB
  across 8-14 workers with zero risk of OOM.
- **Hardware allocation enforcement.** Systems with 16 physical cores operate under a strict
  ceiling of at most 14 worker threads, leaving 2 cores for operating system and disk I/O.
  Active workloads divide threads symmetrically: 6 CPU threads for Monte Carlo branching rollouts
  (`generate_rollouts.exe`) and 8 workers for GPU MCTS search (`zq_search_bridge.exe`).
- **Tier 4 consolidation.** 100,000 dual-crisis positions (loss games, 0-2 sweep openings, and
  acute wall stock asymmetries) were consolidated into `data/teaching/tier4-dual-crisis-100k/dataset.npz`
  with strict train/validation isolation (16,763 validation samples).
- **Tier 3.5 Generic Critical Search mining.** Added `tools/teacher/mine_generic_critical.py`
  to extract 100,000 high-uncertainty positions from canonical self-play shards using vectorized
  filtering on policy entropy, race distance margins (|d_own - d_opp| <= 2), wall stock depletion
  (<= 3), and game-swing turning points (`tier3-5-generic-critical-100k/positions.jsonl`). Supervision
  is calibrated to 80% Claustrophobia MCTS search (64 sims on CUDA) and 20% ZQuoridor MCAB tree search
  (512 nodes on CPU) with Jensen-Shannon divergence scaling.
- **Tier 3 bilateral search completion.** Part 1 and Part 2 (100,000 positions total) finished
  Claustrophobia MCTS search on CUDA. Blended with ZQuoridor 512-node MCAB search targets
  into `data/teaching/generic-search-100k-zq/dataset.npz` (100k samples, 19,881 validation samples,
  mean weight 7.15). Tier 3.5 80/20 bilateral search (100k) completed and fused into
  `data/teaching/tier3-5-generic-critical-100k/dataset.npz` (mean weight 7.58, 19,991 val samples).
- **Tier 5 deep search enrichment.** 10,000 crisis and branching positions from Tier 5 rollouts are
  actively being searched with Claustrophobia MCTS at ~500 ms (512 simulations) on CUDA.
- **Champion training and dataset assembly (`race512-multitier-champion`).** The
  assembly script `tools/teacher/assemble_5tier_dataset.py` consolidated 10,810,000
  samples across all five curriculum tiers into `data/teaching/multitier-final-dataset/dataset.npz`.
  The dataset contains 9,012,797 training samples and 1,797,203 validation samples.
  Chunked binary streaming kept peak memory usage under 200 MB. The model trained
  on the RTX 4050 GPU for 60 epochs with cosine learning rate decay down to `5e-7`.
  Validation loss dropped from 0.9571 to 0.8701. Validation policy KL dropped
  from 0.4903 to 0.4206. Validation value mean absolute error dropped from 0.2730
  to 0.2116. The verification binary `incremental_check.exe` evaluated 4,758 positions
  with zero divergence from the reference accumulator.
- **Screening benchmark metrics.** The screening suite evaluated the champion
  across 20 opening pairs at 200 ms per move against three reference engines:
  - Against `main`: 77.5% score (30 wins, 10 losses, 0 draws) with +214.8 Elo
    (95% confidence interval: +117.2 to +358.8). The model exceeded the +58.5%
    promotion threshold.
  - Against Titanium: 70.0% score (28 wins, 12 losses, 0 draws) with +147.2 Elo
    (95% confidence interval: +52.5 to +269.4). The model exceeded the +57.5%
    promotion threshold. The `main` baseline achieved 67.5% in the same conditions.
  - Against Claustrophobia: 38.8% score (15 wins, 24 losses, 1 draw) with -79.5 Elo
    (95% confidence interval: -157.7 to -17.4). The `main` baseline achieved 43.8%
    (17 wins, 22 losses, 1 draw).
- **Tactical failure analysis against Claustrophobia.** Analysis of the 24 losses
  identified three tactical factors:
  - The model won 63.2% of games as the first player (12 wins, 7 losses). However,
    the model won only 15.0% of games as the second player (3 wins, 17 losses).
    Fifteen of the 20 opening pairs resulted in 1-1 splits where the first player won.
  - The model delayed wall exhaustion compared to `main`. The model ran out of
    walls first in 54.2% of losses, compared to 77.3% for `main`. However, Claustrophobia
    consistently retained 3 to 4 walls until plies 45 to 55. It placed decisive
    cutoffs in the late game when the model held zero walls.
  - In Openings 81 and 86, `main` defeated Claustrophobia 2-0, whereas the candidate
    lost 0-2. Trace analysis revealed that `main` played aggressive containment walls
    at ply 10 and ply 18 (`d6h` and `d4v`). In contrast, the candidate played passive
    pawn advances (`e4` and `c2`), which conceded the center corridor.
- **Confirmation benchmark metrics (200 games per opponent, 200 ms per move).**
  The official confirmation benchmark evaluated `race512-multitier-champion`
  across 100 paired openings (200 games per match):
  - Against `main`: 54.8% score (104 wins, 85 losses, 11 draws) with +33.1 Elo.
  - Against Titanium: 57.0% score (113 wins, 85 losses, 2 draws) with +49.0 Elo.
  - Against Claustrophobia: 49.0% score (94 wins, 98 losses, 8 draws) with -6.9 Elo.
    This exactly matched the 49.0% score achieved by the previous champion
    `race512-search10-ft`, demonstrating that the screening score of 38.8% was
    due to sample variance on a small 40-game set.
- **Direct Head-to-Head vs Previous Champion (`race512-search10-ft`).**
  The completed 200-game direct match between `race512-multitier-champion` and
  `race512-search10-ft` confirmed positive progression: `race512-multitier-champion`
  scored 51.5% (96 wins, 90 losses, 14 draws) with +10.4 Elo (bootstrap 95% CI:
  [-13.8, +34.8]). The new network is strictly stronger than the original champion.
- **Extended 400-Game Match against Claustrophobia.**
  A followup 400-game match on 200 paired openings disjoint from the confirmation suite
  completed on GPU with 200 ms per move. Result: 50.0% score (194 wins, 194 losses, 12 draws)
  with 0.0 Elo (paired bootstrap 95% CI: [-28.7, +28.7]).
- **Combined 600-Game Claustrophobia Evaluation.**
  Across all 600 games (300 distinct paired openings with color swap):
  - Total score: 49.67% (288 wins, 292 losses, 20 draws) with -2.32 Elo.
  - Confirmation set (first 100 pairs): 49.00% (94 wins, 98 losses, 8 draws).
  - Extended set (next 200 pairs): 50.00% (194 wins, 194 losses, 12 draws).
  - Conclusively establishes that the engine matches Claustrophobia in overall strength
    across diverse openings.
- **Center Pawn Rush Openings Suite (`tools/external/openings_center_rush_v1.jsonl`).**
  Constructed 33 rule-verified opening continuations where both players advance their
  pawns into the board center (`e2 e8 e3 e7 e4 e6`) followed by 3 tactical plies.
  The catalog covers 5 tactical families: front containment walls, direct pawn jumps,
  vertical corridor funnels, rear barrier walls, and lateral flank movements.
  This suite serves as the standard evaluation benchmark for search and policy improvements.
- **Center-Rush Baseline Benchmark.**
  Evaluation of `race512-multitier-champion` against Claustrophobia across the 33 center-rush
  openings (66 games, paired color swap, 200 ms/move) recorded 32.58% score (21 wins, 44 losses,
  1 draw; -126.4 Elo). Diagnosis revealed severe tactical imbalance: `vertical_channel` (50.0%)
  and `pawn_jump` (50.0%) were balanced, but `reed_rear_wall` (18.8%) and `sidestep_flank` (8.3%)
  suffered from passive play and delayed wall containment.
- **Policy Head Fine-Tuning (`race512-policy-tactical`).**
  Trained exclusively the policy head (`512 -> 209`) with `--train-scope policy` on 244,787
  deep-search and crisis positions (`data/teaching/policy-tactical-245k/dataset.npz`) for 15
  epochs. The accumulator (`fc1`) and value head remained bit-for-bit frozen (0 divergences,
  zero catastrophic forgetting). Validation policy KL dropped from 0.7831 to 0.7458.
  In benchmark evaluation on the 33 center-rush openings against Claustrophobia, the model
  achieved 37.88% score (24 wins, 40 losses, 2 draws; -85.9 Elo), representing a **+40.5 Elo
  improvement** over baseline. Tactical analysis showed a 4x win rate increase on `sidestep_flank`
  (from 8.3% to 33.3%) and a gain in `pawn_jump` (to 56.2%).
- **MCTS Search Parameter Analysis on Tactical Collisions.**
  Evaluated search configurations with `race512-policy-tactical` on the center-rush suite:
  - `cPuct = 1.40`: Achieved 36.36% overall (24 wins, 42 losses). Increased exploration
    sharply boosted `reed_rear_wall` from 18.8% to 31.2% and `sidestep_flank` to 41.7%,
    proving that higher exploration helps escape barrier lockdown, but slightly degraded
    tight corridor play in `vertical_channel`.
  - `cPuct = 1.10`: Recorded 27.27% score (18 wins, 48 losses), indicating non-linear
    branching sensitivity in center clashes.
  - `progressiveWidening`: Recorded 33.33% score (22 wins, 44 losses), confirming that
    prematurely restricting candidate walls to top-16 priors hurts tactical discovery.
  - Veredict: `cPuct = 0.80` with the fine-tuned tactical policy head delivers the strongest
    and most robust performance (+40.5 Elo over the baseline).
- **Center-Rush 200k Dataset and Action-Q Policy Sharpening (`build_center_rush_priority_dataset.py`).**
  Constructed `data/teaching/center-rush-200k-priority/dataset.npz` containing 255,000 samples:
  - 200,000 sharp Center-Rush positions (files c-g, ranks 3-6, active walls >= 5) weighted at 5.0x.
  - 5,000 deep 1024-node search relabeled positions with Action-Q policy sharpening
    ($\pi'_a \propto N_a^\alpha \exp(\beta Q_a)$, $\alpha=1.0$, $\beta=1.5$) weighted at 6.0x-8.0x.
  - 50,000 general background positions weighted at 1.0x to prevent catastrophic forgetting.
  - Total Center-Rush effective signal share: 95.37%. Strict opening-level grouping ensures
    zero train/validation leakage.
- **Dedicated Two-Stage Training (`race512-cr200k-policy` and `race512-cr200k-champion`).**
  - Stage 1 (`race512-cr200k-policy`): Trained exclusively the policy head with `--train-scope policy`
    for 15 epochs on CUDA with cosine schedule ($1.2 \times 10^{-4} \to 5 \times 10^{-6}$).
    The accumulator trunk (`fc1`) and value head remained 100% frozen. Validation policy KL
    dropped 28% from 0.1015 to 0.0734.
  - Stage 2 (`race512-cr200k-champion`): Initialized from Stage 1. Calibrated heads with
    `--train-scope heads` for 15 epochs ($2.5 \times 10^{-5} \to 5 \times 10^{-7}$).
    Validation loss reached 0.6954 (lowest validation loss in project history).
    Accumulator incremental verification (`incremental_check.exe`): 0 divergences across
    4,758 positions.
- **Full 600-Game Confirmation Match vs Titanium.**
  Completed 600 games across 300 unique openings with paired color swap at 200 ms/move.
  Result: 378 wins, 0 draws, 222 losses -> **63.0% score (+92.5 Elo)** (bootstrap 95% CI:
  [+64.4, +121.7]). This is the all-time highest score recorded against Titanium in the
  project history.
- **3-Way Screening and Tactical Center-Rush Results.**
  - vs `main`: 81.25% score (+254.7 Elo, 32W 1D 7L).
  - vs Titanium: 50.0% score (20W 0D 20L).
  - vs Claustrophobia (40g): 46.25% score (-26.1 Elo, +53.4 Elo above `main`).
  - Tactical Center-Rush Suite: 37.12% score (+35 Elo over baseline), winning `pawn_jump` with
- **Full 600-Game Confirmation Match vs Claustrophobia (Completed).**
  600 games across 300 unique openings with paired color swap at 200 ms/move on GPU (RTX 4050).
  Result: 281 wins, 13 draws, 306 losses -> **47.92% score (-14.5 Elo)** (bootstrap 95% CI:
  [44.33%, 51.50%], Elo: [-39.55, +10.43]).
- **Experimental Architecture Evaluations (2026-09-20).**
  - Evaluated `multipath:512` (480 inputs, 24 multi-path & collision features) across 800 total benchmark games:
    - **vs Claustrophobia GPU (600 games)**: **51.08% score (+7.53 Elo)** (306.5 / 600, paired bootstrap 95% CI: [47.50%, 54.58%]).
      Recorded the first positive strength claim against Claustrophobia GPU in project history.
      Central openings improved to 46.62% (34.5 / 74).
    - **vs Titanium (200 games)**: 64.0% normal, 45.0% center rush (Black: 72.0%, White: 18.0%).
  - Evaluated `margin_regime:512` (588 inputs, 132 distance margin $\times$ wall regime interaction features) across 800 games:
    - **vs Claustrophobia GPU (600 games)**: **49.00% score (-6.95 Elo)** (294.0 / 600, paired bootstrap 95% CI: [45.25%, 52.75%]).
      Central openings: 43.92% (32.5 / 74).
    - **vs Titanium (200 games)**: 65.0% normal, 47.0% center rush (Black: 70.0%, White: 24.0%).
  - **Architectural Conclusion**: `multipath:512` outperforms `margin_regime:512` by +14.5 Elo against Claustrophobia and +2.7% in central openings, showing that path redundancy information provides stronger inductive bias than wall-regime distance interactions. Both networks exhibit asymmetric White defense weakness against Center Rush (18% - 24% winrate for White vs 70% - 72% for Black).
  - **Multi-Worker Benchmark Runner**: Enhanced `tools/run_full_candidate_suite.py` and `tools/run_benchmark.py` to support `--workers 6` with manifest isolation, reducing 800-game test suite time from ~3 hours to ~25 minutes.



---

## 7. Evaluation Conventions

| Stage | Function | Range | Perspective |
| --- | --- | --- | --- |
| Search score | `nnueEvalInt` | ~[-30000, 30000] | mover-relative |
| MCTS Q | `scoreToQ` | [0, 1] | mover-relative win probability |
| Heuristic | `evalSimple` | ~[-600, 600] | mover-relative |
| Dataset field | `TrainingSample::evalNNUE` | 0..65535 | absolute White |
| GUI display | `formatEval` | 0%..100% | absolute White |
