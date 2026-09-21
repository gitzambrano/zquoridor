# Zquoridor

Use `AGENTS.md` as the canonical repository instructions.

Key references:

- `README.md` — external overview and user instructions.
- `docs/status.md` — technical status and measured results.
- `docs/plan.md` — roadmap.
- `docs/datasets.md` — dataset catalog.
- `skills/writing-rules/SKILL.md` — canonical technical writing rules.
- `.claude/skills/writing-rules/SKILL.md` — Claude entry point.

The production engine is hybrid MCTS/MCGS with alpha-beta support and a 504-input → 512-hidden NNUE. Browser search and pondering stay in the Web Worker; do not move blocking search work onto the UI thread.
