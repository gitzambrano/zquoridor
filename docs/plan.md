# Zquoridor: project state, results, and roadmap

Last reviewed: 2026-09-23. This is the single canonical project document.
It answers four questions in order: what is in production, what has already
been measured, what is running now, and what happens next. Raw datasets,
self-play shards, logs, opponent checkouts, and transient checkpoints are local
artifacts and are deliberately not versioned.

## 1. Current production state

| Area | Current state |
| --- | --- |
| Release | Zquoridor 2.02 |
| Search | Hybrid PUCT/MCGS graph search with alpha-beta support, transpositions, persistent tree reuse, repetition escape, and adaptive time management |
| Pondering | Opponent-root pondering with subtree reuse; browser work runs in bounded Web Worker slices |
| NNUE | `multipath_phase:512`: 504 sparse inputs, 512 SCReLU units, WL head `512 → 32 → 1`, policy head `512 → 209`, QAT/int8 |
| Production weights | `data/nnue/nnue_weights.bin` and `data/nnue/nnue_weights_int8.bin` |
| Versioned provenance | `results/experiments/multipath-phase512-searchboost-100ep/` |
| Browser/protocol | WebAssembly build and UCI-style text protocol with fixed-time, clock, and pondering commands |

A candidate must use an executable compiled for its exact NNUE architecture.
Changing search settings creates a separate experiment; it cannot be credited
as an NNUE gain.

## 2. What has been measured

All percentages below are candidate score. A small match is screening evidence,
not proof of superiority. External comparisons use paired colours, shared
openings, fixed **200 ms per move**, and record failures separately from draws.

### Production baseline and search evidence

| Candidate/configuration | Opponent | Result | Sample | Conclusion |
| --- | --- | ---: | ---: | --- |
| Production `multipath_phase:512` | synchronized multipath baseline | 51.25% | 200 games | screening only |
| Production `multipath_phase:512` | Claustrophobia | 50.83% | 600 games | close to even; project goal not met |
| Production `multipath_phase:512` | Titanium, normal book | 60.0% | 100 games | encouraging; needs confirmation |
| Production `multipath_phase:512` | Titanium, Center Rush | 46.0% | 100 games | known weakness |
| Production without/with pondering | Claustrophobia | 51.0% / 51.5% | 100 games each | no significant external conclusion |
| Pondering candidate | same engine without pondering | 63.125% | 400 paired games | internal configuration gain from subtree reuse |
| `multipath_phase_contact:512` | production `multipath_phase:512` searchboost checkpoint | 47.25% | 200 paired games | contact candidate did not win the H2H screen |
| `multipath_phase_contact:512` | Claustrophobia | 50.25% | 600 games | full-suite screening; no strength claim |
| `multipath_phase_contact:512` | Titanium, Center Rush | 49.0% | 100 games | full-suite screening; no strength claim |
| `multipath_phase:512` reliable-search FT | production searchboost checkpoint | 45.5% | 200 paired games | reliable-search FT did not win the H2H screen |
| `multipath_phase:512` reliable-search FT | Claustrophobia | 48.58% | 600 games | full-suite screening; no strength claim |
| `multipath_phase:512` reliable-search FT | Titanium, normal book | 68.5% | 100 games | positive screening result; needs paired confirmation |
| `multipath_phase:512` reliable-search FT | Titanium, Center Rush | 47.0% | 100 games | Center Rush remains unresolved |
| `multipath_phase:512` with MCGS075 | Claustrophobia | 53.875% | 400 games | Stage A result; search setting is not an NNUE claim |

The production candidate failed the recorded Claustrophobia family gate. The
lowest observed family was `reed_rear_wall`; Center Rush also remains a high
priority. Training loss and nodes per second are not strength measurements.

### Network matrix: completed work and result

“Validation loss” only compares runs on the same dataset and recipe. A dash
means the result was not retained or was not comparable; it does not mean that
the network was never trained.

| Network | Inputs / hidden | Training data and recipe | Best validation loss | Arena result retained | Status and next action |
| --- | --- | --- | ---: | --- | --- |
| `base:256` | 354 / 256 | 2M direct replay, QAT, 80 epochs | 0.74569 | historical screening only | archived control |
| `race:256` | 456 / 256 | same 2M direct recipe | 0.74319 | historical screening only | archived control |
| `base:384` | 354 / 384 | same 2M direct recipe | 0.73939 | historical screening only | archived control |
| `race:384` | 456 / 384 | same 2M direct recipe; legacy search fine-tune | 0.73686 | historical screening only | archived control |
| `base:512` | 354 / 512 | direct, then 40-epoch 90/10 search fine-tune | 0.73600 direct | confirmation run completed; detailed raw log local | archived finalist |
| `race:512-search10-ft` | 456 / 512 | direct, then 40-epoch 90/10 search fine-tune | 0.88574 fine-tune* | 49.0% Claustrophobia; 54.0% Titanium | historical baseline |
| `race:512-multitier-champion` | 456 / 512 | 10.8M weighted multitier continuation | 0.86410* | 51.5% vs search10; 49.67% Claustrophobia; 57.0% Titanium | historical champion |
| `race:512-policy-tactical` | 456 / 512 | 245k tactical policy continuation | 0.74580* | 37.88% Center Rush; 55.0% Titanium | did not solve Center Rush |
| `race:512-weakness-ft` | 456 / 512 | 80k mined weakness continuation | 1.14938* | 38.64% Center Rush | did not solve Center Rush |
| `race:512-cr200k-champion` | 456 / 512 | 255k Center Rush policy then heads tuning | 0.69542* | 63.0% Titanium; 46.25% Claustrophobia | strong on Titanium, below target vs Claustrophobia |
| `multipath:512` | 480 / 512 | multipath campaign | — | 51.08% Claustrophobia, 600 games | historical first positive Claustrophobia result |
| `margin_regime:512` | 588 / 512 | weakness/Center Rush campaign | — | 49.0% Claustrophobia, 600 games | versioned research checkpoint; benchmark only if rerun is justified |
| **production `multipath_phase:512`** | **504 / 512** | 11.065M weakness-boosted data, QAT, 100 epochs | — | rows above | **production** |
| `multipath_phase_contact:512` | 858 / 512 | 11.378M local data, warm start, QAT, 160 epochs | 0.95217* | 47.25% H2H; 50.25% Claustrophobia; 49.0% Center Rush | local candidate; no promotion-level gain |
| `multipath_phase:512` reliable-search FT | 504 / 512 | 313,344 selected search/rollout samples, QAT, 40 epochs | 1.16198* | 45.5% H2H; 48.58% Claustrophobia; 68.5% Titanium; 47.0% Center Rush | local unpromoted fine-tune |
| `multipath_phase:512` Arm A | 504 / 512 | 4.26M stored-search, warm start, QAT, 20 epochs | 0.92044* | Val MAE 0.13108 | local control completed |
| `multipath_phase:512` Arm B | 504 / 512 | 4.26M stored-search, mirror-h, warm start, QAT, 20 epochs | 0.95141* | Val MAE 0.13537 | local ablation completed |
| `multipath_phase_bucketed:512` Arm C | 504 / 512 | 4.26M stored-search, 6 buckets, 2 layers, mirror-h, warm start, QAT, 20 epochs | 0.94987* | Val MAE 0.12874 (-29.8% MAE) | local candidate completed |
| `multipath_phase_deep:512` Arm D | 504 / 512 | 4.26M stored-search, 1 head, 2 layers, mirror-h, warm start, QAT, 20 epochs | 0.95029* | Val MAE 0.12956 (-4.3% vs B) | local ablation completed |
| `multipath_phase_contact_bucketed:512` Arm E | 858 / 512 | 4.26M stored-search, 6 buckets, 2 layers, mirror-h, warm start from C, QAT, 20 epochs | queued* | — | contact + bucketed candidate queued |
| `multipath_phase:512` Arm A2 | 504 / 512 | un-biased replay, warm start from A, QAT anneal, LR 1e-5 → 1e-7, 60 epochs | queued* | — | control annealing queued |
| `multipath_phase:512` Arm B2 | 504 / 512 | un-biased replay, mirror-h, warm start from B, QAT anneal, LR 1e-5 → 1e-7, 60 epochs | queued* | — | ablation annealing queued |
| `multipath_phase_bucketed:512` Arm C2 | 504 / 512 | un-biased replay, mirror-h, 6 buckets, 2 layers, warm start from C, QAT anneal, 60 epochs | queued* | — | candidate annealing queued |
| `multipath_phase_deep:512` Arm D2 | 504 / 512 | un-biased replay, mirror-h, 1 head, 2 layers, warm start from D, QAT anneal, 60 epochs | queued* | — | deep ablation annealing queued |
| `multipath_phase_contact_bucketed:512` Arm E2 | 858 / 512 | un-biased replay, mirror-h, 6 buckets, 2 layers, warm start from E, QAT anneal, 60 epochs | queued* | — | contact candidate annealing queued |

\* Do not compare these losses across different datasets, weighting schemes, or
fine-tune stages. Stored replay deduplication aggregates duplicate canonical
states across games by averaging visit policies $\bar{\pi} = \frac{1}{K}\sum \pi_i$
and blended value targets $\bar{V} = \frac{1}{K}\sum V_i$ to eliminate outcome
selection bias. All Version 2 annealing runs train for 60 epochs. Following
training, each candidate model plays a 50-pair (100-game, 200 ms/move) arena match
directly on Colab against the production baseline (`nnue_weights_int8.bin`).



## 3. Architecture facts

| Architecture | Extra signal beyond base features | Incremental cost | Reason to test |
| --- | --- | --- | --- |
| `race` | distance margin, wall-stock margin, race/resource interactions | no extra BFS | exposes race and resource balance |
| `multipath` | open directional exits, exit degree, coarse pawn-contact geometry | no extra BFS | corridor and bottleneck awareness |
| `phase` | wall-stock phase and race-by-phase features | no extra BFS | separates opening, wall fight, and race |
| `margin_regime` | distance-margin by wall-depletion regime | no extra BFS | tests regime-specific race evaluation |
| `multipath_phase` | multipath plus phase | no extra BFS | current production compromise |
| `multipath_phase_contact` | exact pawn displacement, local edge masks, jump/diagonal options | local wall tests only; no extra BFS | richer contact/corridor model for wandering and blocking positions |
| `multipath_phase_bucketed` | `multipath_phase` inputs; 6 value heads selected by total remaining walls; each head `512 → 32 → 32 → 1` | value evaluation 5% slower in isolation | separates wall-fight and race evaluation |
| `multipath_phase_deep` | `multipath_phase` inputs; one head `512 → 32 → 32 → 1` | value evaluation 3% slower in isolation | ablation: second layer without buckets |

The value-head buckets use the same wall boundaries as the `phase` features:
0, 1 to 2, 3 to 5, 6 to 9, 10 to 14, and 15 or more walls.

Expanded architectures warm-start by copying compatible columns and setting new
columns to zero, preserving the old function at epoch zero. Python and C++
feature parity is required before any arena run.

### Long-clock search study (180+2)

This study is separate from the fixed-200 ms promotion protocol. It does not
replace that protocol.

- The historical 180+2 score against Claustrophobia rose from 14.5% before
  native clock support to 29.75% with real-clock budgeting and 34.25% with
  adaptive budgeting. It reached approximately 41% to 42.5% after the DAG,
  MCGS, and repetition work. The current engine stays in that band.
- Search scaling, 200 games per configuration: 32 nodes/ms with a 640k ceiling
  scored 39.0% (-77.7 Elo). The best point was 96 nodes/ms with a 1.28M
  ceiling at 46.0% (-27.9 Elo). The paired bootstrap interval still crosses the
  threshold. Therefore, this configuration is a candidate, not a result.
  128 nodes/ms with a 2.56M ceiling did not improve the point estimate.
- Search architecture, 100 games per configuration: without tree reuse, the
  score fell to 37.0%. FPU 0.2 (27.5%), MaxQ root selection (40.5%), and Gumbel
  root filtering (39.0% for M=32, 16.5% for M=16) are rejected. Progressive
  widening scored 49.0% against a 48.0% same-run baseline. This is a weak
  signal only.
- These settled results stay closed: pondering (400-game confirmation), FPU 0.1
  and MaxVisitsThenQ (neutral or worse), and the AB root prefilter (large
  negative Elo).

## 4. Data and teaching already completed

All training data uses the canonical mover-relative V3 contract: policy over
209 actions and signed value in `[-1, 1]`. Raw V3 self-play is not a training
dataset; replay or teaching produces `dataset.npz` with targets, weights, and a
group-safe validation split.

| Data/campaign | Local artifact | Generator | What it contains | Used by |
| --- | --- | --- | --- | --- |
| Direct replay | 2M replay dataset | `training/prepare_replay.py` | direct policy/value labels | base/race matrix |
| Search fine-tune | 2.001868M mixed dataset | `training/mix_teaching_datasets.py` | 90% direct + 10% selected search by sample weight | base/race 512 fine-tunes |
| Multitier master | 10.8M dataset | `tools/teacher/assemble_5tier_dataset.py` | weighted broad, tactical, weakness, and search tiers | multitier champion |
| Center Rush priority | 255k dataset | `tools/teacher/build_center_rush_priority_dataset.py` | Center Rush, Action-Q, and background positions | CR policy/head stages |
| Weakness boosted | 11.065M dataset | `tools/teacher/assemble_multipath_master_dataset.py` | broad data with weakness/search boosts | production phase network |
| Contact reliable | 11.378344M dataset | same master assembler | weakness-boosted data plus reliable selected search | contact candidate |
| Reliable search fine-tune | 313,344 dataset | selected/relabelled search pipeline | bilateral, critical, crisis, deep Claustrophobia, rollout states | local reliable FT |

Every completed training has its exact command settings in its local
`config.json`, architecture in `student.architecture.json`, and metrics in
`train_report.json`. Only production provenance and one research checkpoint are
versioned; the rest are intentionally local.

## 5. Work running now

`tools/teacher/run_four_million_selfplay.py` is the generic controller for the
local `contact-4m-50ms` V3 corpus with the selected executable and weights.
The audit snapshot contains 4,329,982 unique states, approximately 55% central.
Preserve these states and correct the composition through subsequent shards:

- 50 ms per move and configurable self-play threads;
- final target: 10,000,000 admitted unique states;
- planned final mix: approximately 75% main Stage A data and 25% contact data;
- planned state mix: approximately 75% central states and 25% broad states;
- balanced minimum coverage across required opening families;
- aligned metadata sidecars, manifests, deduplication, and safe resume.

After the corpus reaches 5,000,000 admitted states, the next self-play phase
uses a low-to-high-to-low temperature schedule over MCAB root visits. The
generator records the untempered top-eight visits, renormalized as the policy target.
This keeps search supervision for the policy head while the temperature adds
plausible opening variation.

The local `progress.json` beside that corpus is the source of truth. Never run
a second controller against the same output directory.
The current baseline phase stops at 8,516,655 unique states. Then the contact
phase completes the final targets. Audit producer shares from unique-state
references before that transition. Raw shard row counts include duplicates.

The corrected generators honor full and cheap search budgets on temperature
plies. They label policy and root value only after a search on that ply.
The updated controller records generator hashes and supports a threshold-based
transition. These source changes do not alter the already-running process.

### NNUE experimental candidate training (active)

The three-arm study on 4,262,204 canonical stored-search samples (3,411,166 train,
851,038 validation) warm-started from the production `multipath-phase512-searchboost-100ep`
checkpoint is underway:

- Arm A (Control): `multipath_phase` without mirror. 20 epochs completed.
  Val loss: 0.92044, Policy KL: 0.62121, Value MAE: 0.13108.
- Arm B (Ablation): `multipath_phase` with mirror augmentation. 20 epochs
  in progress.
- Arm C (Candidate): `multipath_phase_bucketed` with mirror augmentation.
  20 epochs completed. Val loss: 0.94987, Policy KL: 0.64976, Value MAE:
  0.12874 (-0.0545 MAE drop from baseline; Bucket 1: 0.1205, Bucket 2: 0.1427,
  Bucket 3: 0.1443).
- Stage 2 Annealing (Recozimento): A2, B2, and C2 fine-tuning runs starting
  from the respective Arm A, B, and C checkpoints. Uses reduced learning rate
  (`lr=1e-5`, `min_lr=1e-7`), slow trunk adaptation (`trunk_lr_scale=0.05`),
  cosine annealing schedule, and QAT to test whether simulated annealing
  improves holdout loss and int8 quantization stability.

- Mirror augmentation: `training/mirror_augmentation.py` flips each training
  sample left to right with probability 0.5. The trainer flips the raw state and
  the policy, and then recomputes all features. Validation is not flipped. Set
  `mirror_h` in the `run_experiment.py` `CONFIG`.
- Mirror verification: `tests/test_mirror_engine.cpp` checked 39,759 positions
  and 1,080,437 legal moves. Path lengths, legal move sets, and successor states
  match under reflection.
- Value heads: the architecture name selects the head. `src/nnue.hpp` reads
  `ZQ_NNUE_VALUE_BUCKETS` (1 or 6) and `ZQ_NNUE_VALUE_DEPTH` (1 or 2). The
  defaults (1 and 1) keep the production weight format unchanged.
- Warm start: the second layer starts as the identity matrix. Therefore, the
  float network is equal to the parent network at epoch zero. A warm-started
  `multipath_phase` export is byte-identical to the production int8 file.
- Parity: in 9,152 positions and all six buckets, the C++ int8 value matches
  the Python int8 value within 2.4e-7. The QAT value matches the C++ int8 value
  within 5e-10. The incremental accumulator matches a full rebuild.
- Speed: search throughput was measured with warm-started weights, 9 positions,
  1 s per position, and 3 alternating runs. The ratios to production are 1.06
  (deep) and 0.98 (6 buckets). The run-to-run spread is approximately 12%.
  Therefore, no speed loss is measurable. An earlier report of 82.9% did not
  use equal weights.

## 6. Promotion protocol

1. Build the exact candidate executable and verify native/Python feature parity.
2. Screen against the frozen production baseline with paired colours, one book,
   one seed, equal cores, and **200 ms per move**.
3. Confirm a screening winner with fresh openings and at least 100 complete
   pairs.
4. Run the same fixed-time protocol against Titanium and Claustrophobia,
   including each required opening family.
5. Promote only when there are no unresolved failures and the confidence
   interval supports the claim. Promotion is never automatic.

## 7. Roadmap

1. Confirm the long-clock search candidate now. This step does not need the
   corpus. Compare 96 nodes/ms with a 1.28M ceiling, alone and with
   progressive widening, against the frozen production baseline and
   Claustrophobia with paired openings. The candidate must also keep the
   fixed-200 ms production gate.
2. Finish and audit the active corpus at 10,000,000 unique states. Preserve all
   existing data. Use corrected visit-temperature generation for the final
   five million states. Start experimental training only after this audit.
3. Train the experimental network matrix on the same data and recipe:
   - A (control): `multipath_phase`, no mirror;
   - B (ablation): `multipath_phase` with mirror augmentation;
   - C (candidate): `multipath_phase_bucketed` (6 heads, 2-layer value) with mirror;
   - D (ablation): `multipath_phase_deep` (1 head, 2-layer value) with mirror;
   - E (candidate): `multipath_phase_contact_bucketed` (858 features, 6 heads, 2 layers) with mirror.

   Follow each arm with a low-rate simulated annealing (recozimento) QAT pass:
   - A2, B2, C2, D2, E2: warm-started from A, B, C, D, E checkpoints respectively;
   - Reduced learning rate (peak `1e-5`, cosine decay to `1e-7`, 20 epochs);
   - Constrained trunk updates (`trunk_lr_scale=0.05`);
   - Test whether simulated annealing improves validation error and int8 stability.
4. Blend root and result targets with a measured discount. Keep genuine
   search-policy labels separate from replay labels. Use the generic
   stored-search replay mode. Retain a broad anchor and cap critical-source
   weights. Report sample counts, effective weight per source, policy KL,
   value loss, value error per bucket, and family holdouts.
5. Use an explicit cosine weight-decay schedule to its minimum, warm up the
   learning rate, retain a sufficient QAT tail, and set patience for the full
   80 to 160 epoch schedule.
6. Run parity and holdout checks, then fixed-200 ms screening. Keep search
   settings unchanged in the comparison.
7. Require paired evidence against production, Titanium, and Claustrophobia.
   Require a point score above 60% in each of the five Claustrophobia families
   at 200 ms before promotion. Use four 3+2 games only to check clock safety.
8. Test low-cost cached BFS and local wall geometry before adding a network
   architecture. Do not add a network without a specific weakness hypothesis
   and a reproducible configuration.

## 8. Durable lessons

- Broad direct labels alone have not solved Claustrophobia corridor and Center
  Rush weaknesses.
- Expensive search labels help most when selected from disagreement or unstable
  positions. Blind self-distillation can reinforce wandering.
- Cached BFS features and local wall geometry are cheap to evaluate, but an
  architecture is only useful if the paired arena shows a gain.
- Neither contact nor reliable-search FT demonstrated a promotion-level gain.
- At 180+2, a larger node budget raised the score from 39.0% to 46.0%. No
  network change produced a comparable screening gain.
- The active corpus is below the 10M target. The 5M root-visit schedule is a
  planned transition, not a completed result.
- Keep raw data and transient artifacts local. Version source, reproducible
  provenance, concise results, and the current roadmap.

## 9. Operating scripts

See [scripts.md](scripts.md) for the canonical runners, internal stages, and
input/output contracts. `docs/datasets.md` remains local-only and must not be
added to Git.


#### 2026-09-23 long-clock finalist confirmation

The 400-game confirmation gates refined the earlier screens:

- Baseline vs Claustrophobia at 180+2: 141W/2D/257L, 35.50%, -103.7 Elo,
  paired-bootstrap 95% [-136.0, -73.2].
- Global 96 nodes/ms / 1.28M ceiling vs Claustrophobia: 169W/6D/225L,
  43.00%, -49.0 Elo, 95% [-80.4, -18.3]. This confirms a large
  opponent-specific improvement over the baseline point estimate.
- The same global scaling lost head-to-head to the frozen baseline at 180+2:
  179W/7D/212L over 396 included games, 45.83%, -29.0 Elo,
  95% [-56.6, -1.8]. Therefore global 96/1.28M is rejected for promotion.
- Adding global progressive widening did not solve the promotion gate:
  42.21% vs Claustrophobia at 180+2; 49.88% H2H vs baseline at 180+2;
  and 46.88% H2H at fixed 200 ms (-21.7 Elo point estimate,
  95% [-50.7, +6.9]). Global PW remains off.

Next experiment: preserve the production 32 nodes/ms / 640k guardrail for
stable roots and unlock the proven 96 nodes/ms / 1.28M working set only when
the existing adaptive root signal classifies a position as uncertain or
volatile. Test volatile-only and uncertain-or-volatile escalation separately.
The escalation must be disabled for fixed-movetime/self-play so the 200-ms
production path remains bit-for-bit unchanged unless a dedicated test enables
it. Promotion still requires gates versus both frozen main and Claustrophobia.
