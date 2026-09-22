# Zquoridor project instructions

Zquoridor is a two-player Quoridor engine for the standard 9×9 board with 10 walls per player. Version 2.00 uses a **hybrid MCTS with alpha-beta** search: a persistent PUCT/MCGS graph guided by NNUE value and policy, with alpha-beta integrated for tactical/endgame work and available as a standalone fallback.

## Documentation

- `README.md`: concise external overview, installation, protocol, and current architecture.
- `docs/status.md`: technical status, measured results, design decisions, regressions, and open work.
- `docs/plan.md`: roadmap and longer-term design work.
- `docs/datasets.md`: dataset catalog and data-pipeline notes.
- Other project documentation belongs under `docs/` unless it is a component-local README.

Do not put experiment history or training-data composition into `README.md`. Keep the README high level and stable.

## Writing

All code comments, docs, CLI text, and commit messages are in English.

Follow `skills/writing-rules/SKILL.md` for technical prose. Claude-specific entry point: `.claude/skills/writing-rules/SKILL.md`.

## Core architecture

The engine core is header-only under `src/`:

- `rules.hpp`: state, legal moves, path logic, repetition-relevant state.
- `dsu.hpp`: rollback DSU used by wall-legality checks.
- `cat.hpp`: corridor-attention features for move ordering.
- `search.hpp`: Negamax alpha-beta, TT, PVS/LMR, quiescence, ordering.
- `mcab.hpp`: production hybrid MCTS/MCGS search, graph transpositions, tree reuse, search caches, adaptive time logic.
- `nnue.hpp`: production 504-input → 512 SCReLU NNUE with WL and 209-move policy heads.
- `endgame_race.hpp`: exact pawn-race support.
- `search_tuning.hpp`: runtime search overrides used by experiments and arenas.

Production NNUE layout: 504 sparse inputs, 512 hidden units, WL head `512→32→1`, policy head `512→209`, QAT/int8 inference.

## Artifact retention

Large generated datasets are local artifacts and stay out of Git: self-play
corpora, teaching datasets, training checkpoints, raw benchmark outputs, and
external bots are covered by `.gitignore`.

Production NNUE weights live in `data/nnue/`. Versioned network experiments
under `results/experiments/` are intentionally limited to the production
provenance checkpoint and one current richer research checkpoint. Remove
superseded network binaries instead of building a permanent archive in the
working tree; historical results belong in `docs/status.md` and
`docs/plan.md`.

## Build

Native binaries are individual C++17 translation units. Use the scripts in `build/`.

Common commands:

```bash
./build/build_uci.sh
./build/build_tests.sh
./build/build_arena.sh
./build/build_selfplay.sh
./build/build_all.sh
```

Windows uses the matching `.bat` files.

WASM:

```bash
./build/build_wasm.sh
```

The WASM build regenerates the browser runtime and standalone bundle. Keep the exported function lists in `build_wasm.sh` and `build_wasm.bat` synchronized.

## Text protocol

`tools/external/zquoridor_uci.cpp` implements the UCI-style line protocol. Keep `help` accurate when commands change.

Opponent-root pondering is a production feature. The text adapter exposes `ponder movetime <ms>`. The browser runs ponder work only inside the Web Worker, in short bounded slices. Never move long search work onto the browser UI thread.

## Web

`gui_web/engine_wasm.cpp` is the C/WASM binding.

`gui_web/worker.js` owns the background WASM instance used for engine moves and analysis. It keeps a persistent live game so MCAB tree reuse works across moves. Ponder slices must remain bounded and cancellable by the next worker request.

Run browser regression tests after changing search exports, worker state, WASM bindings, or GUI scheduling.

## Validation rules

- Do not promote a search change from nodes/s alone. Use paired game evidence.
- Preserve deterministic/correctness tests for rules and search semantics.
- Keep NNUE Python/C++ parity checks valid when the architecture changes.
- Any new search heuristic needs a runtime toggle or isolated experimental path until it passes a strength gate.
- Keep benchmark opponents and production search configuration independent of documentation claims.
- Update `docs/status.md` for measured technical results and important regressions.
