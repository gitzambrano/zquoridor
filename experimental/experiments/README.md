# Experimental Workspaces and Model Archive

This directory catalogs historical and active neural network experiments for Zquoridor.
For complete measurement tables, tournament cross-tables, and promotion history,
refer to `docs/plan.md`.

## Experiment Index

| Directory | Architecture | Features / Hidden | Status | Primary Role / Findings |
| :--- | :--- | ---: | :--- | :--- |
| [`architecture-lab-v401/`](architecture-lab-v401/) | Multiple | Multiple | Active | Suite of 16 experimental arms exploring Wall Drops, Gumbel simulation targets, and temporal search. |
| [`contact_soup_tri_512/`](contact_soup_tri_512/) | `multipath_phase_contact_bucketed` | 858 / 512 | Promoted (V4.01 Base) | Convex Model Soup (`Soup_Tri_Champion`) combining sprint, surprise, and regularized arms. Resolved H2H deficit. |
| [`contact-bucketed512-central-20261003/`](contact-bucketed512-central-20261003/) | `multipath_phase_contact_bucketed` | 858 / 512 | Superseded | First 858-feature single checkpoint. Strong external performance but initial H2H deficit vs Main (-22.6 Elo). |
| [`production-central-finetune-20261002/`](production-central-finetune-20261002/) | `multipath_phase_bucketed` | 504 / 512 | Benchmark Reference | Central finetuning targeting Center Rush lines with mixed datasets and surprise weights. |
| [`frozen-production-20261002/`](frozen-production-20261002/) | `multipath_phase_bucketed` | 504 / 512 | Benchmark Baseline | Immutable frozen snapshot of production 3.01 (`Soup_Tri_Equal`) used as standard opponent in evaluation batteries. |
| [`aux_policy_stage_a/`](aux_policy_stage_a/) | `multipath_phase_bucketed` | 504 / 512 | Superseded | Exploration of auxiliary policy targets with Model Soup combinations (`Soup_AS1_Heavy`, `Soup_AS2_60`). |
| [`multipath_contact_bucketed_unified/`](multipath_contact_bucketed_unified/) | `multipath_phase_contact_bucketed` | 858 / 512 | Superseded | Candidate training run on the master unified clean 15.6M dataset (`multipath_unified_clean_15m`). |
| [`multipath-phase512-searchboost-100ep/`](multipath-phase512-searchboost-100ep/) | `multipath_phase_bucketed` | 504 / 512 | Historical | 504-feature student model trained for 100 epochs with search boost curriculum. Warm start basis for contact features. |

## Experiment Workspace Conventions

1. **Scaffolding**: Use `python training/new_experiment.py --name <name>` to instantiate a standardized experiment workspace.
2. **Configuration**: Every experiment directory retains its canonical `config.json` defining optimizer hyperparameters, learning rate schedules, and dataset paths.
3. **Documentation**: Each workspace maintains a local `README.md` documenting hypothesis, architecture specifications, benchmark outcomes, and final promotion status.
4. **Promotion standard**: A network must strictly outperform the previous production network across every dimension (head-to-head vs Main, vs Claustrophobia, vs Titanium, across Normal and Center Rush opening books).
