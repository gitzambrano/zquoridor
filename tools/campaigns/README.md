# Campaign scripts

This directory holds every script that built one specific dataset or ran one
specific campaign. Each script has fixed default inputs and outputs for that
campaign. The generic runners remain in `tools/`, `tools/teacher/`, and
`training/`. A generic runner never imports a campaign script. Shared logic
goes into a generic module, for example `training/policy_mix.py` or
`training/native_parity.py`.

Run a campaign script again only to reproduce its original output. For a new
dataset or campaign, use a generic runner, or copy the script and change its
`CONFIG`. The provenance record of each dataset names the script that built it.
`docs/datasets.md` shows these records.

A dataset script writes to `data/` when it starts. All scripts in this
directory parse their arguments, so `--help` does not start a run. The
exception is `release_410.py`, which runs only in its workflow.

## Dataset builders

| Script | Default output |
| :--- | :--- |
| `assemble_200k_dataset.py` | `data/teaching/massive-200k-sharpened/` (needs `--run`) |
| `assemble_5tier_dataset.py` | `data/teaching/multitier-final-dataset/`. With `--out`, it also built `reliable-search-critical-ft-313k/`. |
| `assemble_multipath_master_dataset.py` | `data/teaching/multipath-master-11m/` |
| `assemble_unified_clean_dataset.py` | `data/teaching/multipath_unified_clean_15m/` |
| `assemble_weakness_dataset.py` | `data/teaching/weakness-fine-tuning-champion/` |
| `build_center_rush_priority_dataset.py` | `data/teaching/center-rush-200k-priority/` |
| `calibrate_claustrophobia_weakness_weights.py` | `data/teaching/multipath-weakness-boosted-11m/` |
| `extract_live_rollouts.py` | `data/teaching/loss-center-rollouts/` |
| `mine_generic_critical.py` | `data/teaching/tier3-5-generic-critical-100k/` |
| `mine_losses_and_center.py` | `data/teaching/loss-center-seeds/` |
| `prepare_massive_seeds.py` | `data/teaching/massive_seeds_3500.jsonl` |
| `sample_massive_background.py` | `data/teaching/massive-background-sample/` |
| `weight_weakness_seeds.py` | The 60/40 seed schedule of `weakness-selfplay-100ms` |

## Self-play and opening-bank campaigns

| Script | Default output |
| :--- | :--- |
| `generate_targeted_weakness_bank.py` | `tools/external/openings_targeted_weakness_bank.jsonl` |
| `mine_weaknesses.py` | `tools/external/weak_openings_mined.json` |
| `run_central_weakness_rollouts.py` | `data/selfplay/central-weakness-rollouts-16m/` |
| `run_four_million_selfplay.py` | `data/selfplay/corpus-contact-4m-50ms/` |
| `run_targeted_weakness_rollouts.py` | `data/selfplay/targeted-weakness-rollouts/` |

## Training campaigns

| Script | Default output |
| :--- | :--- |
| `run_contact_finetune.py` | `results/experiments/contact-bucketed512-central-20261003/` |
| `run_experimental_anneal.py` | `results/experiments/central-multinetwork-anneal120/` |
| `run_policy_mix_campaign.py` | `experimental/experiments/policy-mix-after-width-20261013/` |
| `run_production_finetune.py` | `results/experiments/production-central-finetune-20261002/` |
| `run_stage_a.py` | `results/experiments/aux_policy_stage_a/` |
| `run_temporal_campaign.py` | `experimental/experiments/temporal-search-100ep-20261013/` |
| `run_width_pilot.py` | `experimental/experiments/width-100ep-20261013/` |

## Evaluation and release campaigns

| Script | Default output |
| :--- | :--- |
| `measure_temporal_strength.py` | `results/benchmarks/temporal-search-100ep-strength-20261013/` |
| `release_410.py` | The 4.10 release files. Only `.github/workflows/zquoridor-410-release.yml` runs it. |
| `run_central_finetune_evaluation.py` | `results/benchmarks/central-finetune/` |
| `run_v3_search_confirmation.py` | The confirmation gates of the `v3-*` workflows |
| `run_v3_search_screening.py` | The screens of the `v3-search-*` workflows |

The `v3-*` workflows start when their own file changes on `main`. A commit that
edits one of these workflow files must contain `[skip ci]` in its message.
