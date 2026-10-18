# Zquoridor agent rules

This document governs agent behavior, repository structure, and development safety.

## Repository map

- `README.md`: High-level external overview and usage.
- `docs/plan.md`: Single canonical roadmap, measurement records, experiment history, and dataset inventories.
- `docs/scripts.md`: Catalog of operational runners with input, output, and CLI contracts. Do not place roadmaps or training plans in this file.
- `docs/datasets.md`: Local ignored dataset inventory. Never commit this file.
- `docs/training-data.md`: Canonical fields, source names, target quality, and mixture semantics. Do not place collection status here.
- `src/`: Header-only C++ engine core.
- `build/`: Platform build entry points.
- `tools/`: Benchmarks, arena referees, external bot wrappers, and teaching stages.
- `training/`: Training pipelines, quantization, parity verification, and campaign controllers.

## Core responsibilities

- Keep `docs/plan.md` as the single source of truth for:
  - Current production state and benchmark measurements.
  - Complete history of trained networks and ablation studies.
  - Available dataset inventories and self-play configurations.
  - Future improvement roadmap and promotion gates.
- Keep `docs/scripts.md` strictly as a reference catalog:
  - Document what each script does.
  - Document input files, output artifacts, default `CONFIG`, and CLI overrides.
  - Never place future plans, experiment logs, or roadmap text in `docs/scripts.md`.

## Change rules

- Write all comments, documentation, CLI text, and commit messages in formal technical English.
- Follow the rules in `skills/writing-rules/SKILL.md` for project prose.
- Keep operational Python runners generic:
  - Place default parameters in the top-level `CONFIG` dictionary.
  - Run scripts without arguments to execute default workflows.
  - Use CLI flags strictly as single-invocation overrides.
- Use `training/train_nnue.py` as the public training entry point. Keep model,
  mixture, and campaign implementations in internal modules. Keep experiment
  recipes under `experimental/experiments/`. Do not add one public trainer per model.
- Preserve policy, value, and auxiliary evidence separately when mixing data.
  Require all configured recent worker snapshots before a full architecture campaign.
- Keep datasets, checkpoints, binaries, external bots, logs, and raw benchmark outputs outside Git.
- Update `docs/plan.md` whenever you measure results, observe regressions, or make architectural decisions.

## Build and verification

- Use `build/build_tests.bat` (Windows) or `build/build_tests.sh` (Linux) for correctness tests.
- Use `build/build_all.bat` or `build/build_all.sh` for complete native builds.
- Keep export lists in `build_wasm.bat` and `build_wasm.sh` synchronized.
- Run Python import, syntax, and focused test checks after script modifications.
- Preserve NNUE Python and C++ parity and deterministic engine tests.
- Require paired game evidence before promoting search or network changes.
- Strict promotion requirement: A new candidate network must strictly outperform the previous production network across every dimension: head-to-head vs Main, vs Claustrophobia, vs Titanium, and across all opening books (Normal and Center Rush). Underperformance in any single category disqualifies promotion.

## Engine safety

- Preserve legal move and path guarantees when modifying `src/rules.hpp` or `src/dsu.hpp`.
- Keep search heuristics behind runtime configuration toggles until they pass correctness and strength checks.
- Keep long browser searches in the Web Worker and bounded by time budgets.

## Cloud operations and repository organization

- Keep current Colab implementations under `colab/` according to `colab/layout.json`.
- Read `colab/AGENTS.md` and `colab/README.md` before cloud operations.
- Use `docs/repository-layout.md` for source roles and compatibility rules.
- Keep generated reports and environments under ignored `artifacts/` directories.
- Preserve frozen snapshots and active experiments during source cleanup.
