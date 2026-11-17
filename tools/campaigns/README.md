# Campaign scripts

This directory holds scripts that built one specific dataset or ran one specific
campaign. Each script contains fixed input and output paths for that campaign.
The generic runners remain in `tools/`, `tools/teacher/`, and `training/`.

Run a campaign script again only to reproduce its original output. For a new
dataset, use a generic runner or copy the script and change its `CONFIG`.
The provenance record of each dataset names the script that built it.
`docs/datasets.md` shows these records.

| Script | Output |
| :--- | :--- |
| `assemble_5tier_dataset.py` | `data/teaching/multitier-final-dataset/` by default. With `--out`, it also built `reliable-search-critical-ft-313k/`. |
| `assemble_multipath_master_dataset.py` | `data/teaching/multipath-master-11m/` |
| `calibrate_claustrophobia_weakness_weights.py` | `data/teaching/multipath-weakness-boosted-11m/` |
| `assemble_unified_clean_dataset.py` | `data/teaching/multipath_unified_clean_15m/` |
| `assemble_weakness_dataset.py` | `data/teaching/weakness-fine-tuning-champion/` |
| `assemble_200k_dataset.py` | `data/teaching/massive-200k-sharpened/` |
| `build_center_rush_priority_dataset.py` | `data/teaching/center-rush-200k-priority/` |

Other campaign scripts remain at their original paths because GitHub workflows
call them by path. These scripts are `tools/release_410.py`,
`tools/run_v3_search_screening.py`, and `tools/run_v3_search_confirmation.py`.
Each of those workflows starts when its own file changes on `main`.
