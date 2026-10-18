# Zquoridor

Follow `AGENTS.md` as the canonical repository guidance.

Use these references:

- `README.md` for the stable external overview.
- `docs/plan.md` for production state, measurements, and roadmap.
- `docs/scripts.md` for generic runners and pipeline contracts.
- `docs/datasets.md` for the local ignored dataset inventory.
- `docs/training-data.md` for target semantics and mixture contracts.
- `skills/writing-rules/SKILL.md` for project prose.

Keep browser search and pondering in the bounded Web Worker. Do not move long
blocking searches onto the UI thread.

## Cloud operations and repository organization

- Keep current Colab implementations under `colab/` according to `colab/layout.json`.
- Read `colab/AGENTS.md` and `colab/README.md` before cloud operations.
- Use `docs/repository-layout.md` for source roles and compatibility rules.
- Keep generated reports and environments under ignored `artifacts/` directories.
- Preserve frozen snapshots and active experiments during source cleanup.
