# Experimental Workspaces and Model Archive

This directory catalogs historical and active neural network experiments for Zquoridor.
For complete measurement tables, tournament cross-tables, and promotion history,
refer to `docs/plan.md`.

## Experiment Index

| Directory | Architecture | Features / Hidden | Status | Primary Role / Findings |
| :--- | :--- | ---: | :--- | :--- |
| Production 4.10 (no workspace) | `multipath_phase_contact_bucketed` | 858 / 512 | Production | The 4.10 release replaced only the INT8 policy head. `data/nnue/release.json` records the training data and checksums. `docs/releases/4.10.md` records the evidence. |
| [`architecture-lab-v401/`](architecture-lab-v401/) | Multiple | Multiple | Active | Suite of 16 experimental arms exploring Wall Drops, Gumbel simulation targets, and temporal search. |
| [`contact_soup_tri_512/`](contact_soup_tri_512/) | `multipath_phase_contact_bucketed` | 858 / 512 | Promoted (V4.01 Base) | Convex Model Soup (`Soup_Tri_Champion`) combining sprint, surprise, and regularized arms. Resolved H2H deficit. |
| [`contact-bucketed512-central-20261003/`](contact-bucketed512-central-20261003/) | `multipath_phase_contact_bucketed` | 858 / 512 | Superseded | First 858-feature single checkpoint. Strong external performance but initial H2H deficit vs Main (-22.6 Elo). |
| [`production-central-finetune-20261002/`](production-central-finetune-20261002/) | `multipath_phase_bucketed` | 504 / 512 | Benchmark Reference | Central finetuning targeting Center Rush lines with mixed datasets and surprise weights. |
| [`frozen-production-20261002/`](frozen-production-20261002/) | `multipath_phase_bucketed` | 504 / 512 | Benchmark Baseline | Immutable frozen snapshot of production 3.01 (`Soup_Tri_Equal`) used as standard opponent in evaluation batteries. |
| [`aux_policy_stage_a/`](aux_policy_stage_a/) | `multipath_phase_bucketed` | 504 / 512 | Superseded | Exploration of auxiliary policy targets with Model Soup combinations (`Soup_AS1_Heavy`, `Soup_AS2_60`). |
| [`multipath_contact_bucketed_unified/`](multipath_contact_bucketed_unified/) | `multipath_phase_contact_bucketed` | 858 / 512 | Superseded | Candidate training run on the master unified clean 15.6M dataset (`multipath_unified_clean_15m`). |
| [`multipath-phase512-searchboost-100ep/`](multipath-phase512-searchboost-100ep/) | `multipath_phase_bucketed` | 504 / 512 | Historical | 504-feature student model trained for 100 epochs with search boost curriculum. Warm start basis for contact features. |

## Workspace conventions

1. Create a workspace with `python training/new_experiment.py --name <name>`.
2. Train with `python training/run_experiment.py --config experimental/experiments/<name>/config.json`.
3. Record the hypothesis, the results, and the promotion decision in the workspace `README.md`.
4. Add one row for the workspace to the index above.
5. Record the final measurements in `docs/plan.md`.

`config.json` holds the training overrides. After a run starts, `run_experiment.py`
replaces the file with the full training identity. The `config` key of that
identity remains a valid input for `--config`.

A candidate network must outperform the production network in every category:
head-to-head against Main, against Claustrophobia, and against Titanium. Each
category uses the Normal and Center Rush opening books.
