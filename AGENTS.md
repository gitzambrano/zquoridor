# Zquoridor agent rules

## Repository map

- `README.md`: stable external overview and usage.
- `docs/plan.md`: versioned production state, measurements, and roadmap.
- `docs/scripts.md`: canonical runners and input/output contracts.
- `docs/datasets.md`: local ignored dataset inventory; never commit it.
- `src/`: header-only engine core.
- `build/`: platform build entry points.
- `tools/`: benchmarks, arena, bots, and reusable teaching stages.
- `training/`: dataset validation, training, quantization, parity, and campaigns.

## Change rules

- Keep comments, documentation, CLI text, and commit messages in English.
- Use the writing-rules skill for project prose.
- Keep operational Python runners generic. Edit their top-level `CONFIG`; run
  without arguments for defaults and use CLI flags only as overrides.
- Keep datasets, checkpoints, binaries, external bots, logs, and raw benchmark
  output outside Git.
- Do not delete a resumable campaign wrapper until a documented generic runner
  reproduces its input, output, and resume contract.
- Update `docs/plan.md` for measured results, regressions, and architectural
  decisions. Keep `README.md` high level.

## Build and test

- Use `build/build_tests.bat` or `.sh` for correctness tests.
- Use `build/build_all.bat` or `.sh` for the complete native build.
- Keep `build_wasm.bat` and `build_wasm.sh` export lists synchronized.
- Run Python import, syntax, and focused pytest checks after script changes.
- Preserve NNUE Python/C++ parity and deterministic engine tests.
- Require paired game evidence before promoting search or network changes.

## Engine safety

- Preserve legal move and path guarantees when changing `rules.hpp` or `dsu.hpp`.
- Keep search heuristics behind runtime toggles until they pass correctness and
  strength checks.
- Keep long browser searches in the Web Worker and bounded/cancellable.
