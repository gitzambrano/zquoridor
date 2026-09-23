# Zquoridor: project state, results, and roadmap

Last reviewed: 2026-09-22. This is the single canonical project document.
It answers four questions in order: what is in production, what has already
been measured, what is running now, and what happens next. Raw datasets,
self-play shards, logs, opponent checkouts, and transient checkpoints are local
artifacts and are deliberately not versioned.

## 1. Current production state

| Area | Current state |
| --- | --- |
| Release | Zquoridor 2.01 |
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

\* Do not compare these losses across different datasets, weighting schemes, or
fine-tune stages.

## 3. Architecture facts

| Architecture | Extra signal beyond base features | Incremental cost | Reason to test |
| --- | --- | --- | --- |
| `race` | distance margin, wall-stock margin, race/resource interactions | no extra BFS | exposes race and resource balance |
| `multipath` | open directional exits, exit degree, coarse pawn-contact geometry | no extra BFS | corridor and bottleneck awareness |
| `phase` | wall-stock phase and race-by-phase features | no extra BFS | separates opening, wall fight, and race |
| `margin_regime` | distance-margin by wall-depletion regime | no extra BFS | tests regime-specific race evaluation |
| `multipath_phase` | multipath plus phase | no extra BFS | current production compromise |
| `multipath_phase_contact` | exact pawn displacement, local edge masks, jump/diagonal options | local wall tests only; no extra BFS | richer contact/corridor model for wandering and blocking positions |

Expanded architectures warm-start by copying compatible columns and setting new
columns to zero, preserving the old function at epoch zero. Python and C++
feature parity is required before any arena run.

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

1. Finish and audit the active corpus at 10,000,000 unique states. Preserve all
   existing data. Use corrected visit-temperature generation for the final
   five million states. Start experimental training only after this audit.
2. Train the main and contact paths on the same data mixture. Blend root and
   result targets with a measured discount. Keep genuine search-policy labels
   separate from replay labels.
   Use the generic stored-search replay mode after the corpus audit. Retain
   a broad anchor and cap critical-source weights. Report sample counts and
   effective weight per source, policy KL, value loss, and family holdouts.
3. Use an explicit cosine weight-decay schedule to its minimum, warm up the
   learning rate, retain a sufficient QAT tail, and set patience for the full
   80 to 160 epoch schedule. Compare the same recipe and dataset.
4. Run parity and holdout checks, then fixed-200 ms screening. Keep search
   settings unchanged in the comparison.
5. Require paired evidence against production, Titanium, and Claustrophobia.
   Require a point score above 60% in each of the five Claustrophobia families
   at 200 ms before promotion. Use four 3+2 games only to check clock safety.
6. Test low-cost cached BFS and local wall geometry before adding a network
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
- The active corpus is below the 10M target. The 5M root-visit schedule is a
  planned transition, not a completed result.
- Keep raw data and transient artifacts local. Version source, reproducible
  provenance, concise results, and the current roadmap.

## 9. Operating scripts

See [scripts.md](scripts.md) for the canonical runners, internal stages, and
input/output contracts. `docs/datasets.md` remains local-only and must not be
added to Git.


### 2026-09-23 long-clock search study

The 180+2 historical campaign and the follow-up search screens establish a
separate long-clock result. They do not replace the fixed-200-ms promotion
protocol.

- Historical 180+2 progression versus Claustrophobia improved from 14.5% before
  native clock support to 29.75% with real-clock budgeting, 34.25% with adaptive
  budgeting, and approximately 41-42.5% after DAG/MCGS/repetition work. The
  current engine remained in that same statistical band; the immediate-win
  fast path showed no measurable strength regression.
- Search-scaling screen, 200 games/configuration: production-style
  32 nodes/ms with a 640k ceiling scored 39.0% (-77.7 Elo); the best directional
  point was 96 nodes/ms with a 1.28M ceiling at 46.0% (-27.9 Elo). The paired
  bootstrap interval still crossed the promotion threshold, so this is a
  candidate, not a result to ship. Raising both knobs further to 128/2.56M did
  not improve the point estimate.
- Architecture screen, 100 games/configuration: disabling tree reuse fell to
  37.0%, confirming that reuse is important at long clocks. FPU 0.2 (27.5%),
  MaxQ root selection (40.5%), and simple Gumbel root filtering (39.0% for
  M=32; 16.5% for M=16) were rejected. Progressive widening scored 49.0%
  against a 48.0% same-run baseline and remains only a weak directional signal.
- Previously settled experiments were not reopened: pondering already has a
  400-game confirmation; FPU 0.1 and MaxVisitsThenQ were previously neutral or
  worse; AB root prefilter was already rejected by large negative Elo results.

Next long-clock confirmation gate: compare 96 nodes/ms / 1.28M alone and the
same scaling plus progressive widening against the same frozen production
baseline and Claustrophobia with paired openings. No candidate is promoted
unless it also preserves the fixed-200-ms production gate.
