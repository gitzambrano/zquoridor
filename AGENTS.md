## Project

Zquoridor is a 2-player Quoridor engine (9×9, 10 walls each; the 4-player variant is out of scope). Negamax + alpha-beta search, with an NNUE that is trained on the engine's own self-play games. Sister project of the chess engine **Zchezz**, whose conventions this repo deliberately mirrors.

`readme.md` (620 lines) and `plano-additional.md` (1162 lines) are the design documents, both in Portuguese. `plano-additional.md` is organized by numbered "Prioridades" that the code comments reference directly (e.g. "Prioridade 4" = endgame race solver) — when touching search or eval, read the relevant Prioridade section first; several of them document approaches that were **tried and rejected** and should not be re-attempted.

**Language: comments, commit messages, and docs are in English.**

## Layout & build model

Header-only engine: everything in `src/` except `main.cpp`/`selfplay_main.cpp` is a `.hpp` included directly. There is no build system — each binary is one `g++` invocation over one translation unit, with `-Isrc`. `teste/*.cpp` and `gui_web/engine_wasm.cpp` include the same headers; no `.hpp` is ever duplicated.

Two flag profiles, deliberately different:

- **Performance targets** (`src/main.cpp`, benchmarks, selfplay, tune_spsa, arena): `-O3 -std=c++17 -march=native -mavx2 -mfma`
- **Correctness tests** (`teste/test_*.cpp`, `nnue_verify.cpp`): `-O2 -std=c++17` only — no `-march=native`/AVX2, so numerical-parity results stay reproducible.

`bin/` and `teste/bin/` are gitignored build output.

## Commands

Windows uses `build/*.bat` (needs MinGW-w64 `g++` on PATH; `build_selfplay.bat` hardcodes `C:\mingw64\bin`). Linux/macOS `build/*.sh` are exact equivalents. ARM/Termux uses `build/build_termux.sh` — same optimization levels minus `-mavx2 -mfma` (x86-only flags that make the normal scripts fail).

```bat
build\build_all.bat              :: bench + tests + selfplay + tune_spsa (stops at first error)
build\build_all.bat wasm         :: same plus WASM (needs emsdk already active)
build\build_tests.bat            :: correctness suite only
build\build_bench.bat            :: performance benchmarks only
```

Running tests — each test binary takes no arguments and is run individually; there is no test runner:

```bat
bin\test_rules_sanity.exe
bin\test_search_staging.exe
bin\test_move_ordering.exe
bin\test_endgame_race.exe
bin\test_lmr_pvs.exe
bin\nnue_verify.exe data\nnue\nnue_weights.bin data\nnue\nnue_weights_int8.bin
```

`nnue_verify` is the C++ half of a cross-language parity check; `training/parity_check.py` is the Python half — they must be compared against each other, not run in isolation.

Self-play data generation:

```bat
build\build_selfplay.bat
bin\selfplay.exe --games 20000 --chunk-games 2000 --threads 12 --time-ms 200 ^
    --out "data\selfplay\selfplay_{shard:03d}.bin"
python training\run_selfplay.py   :: orchestrator; edit the CONFIG block at the top, no CLI flags
```

NNUE training (Python, from `training/`):

```bash
python train_nnue.py --data ../data/selfplay_*.bin --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots
python quantize_nnue.py <in_f32.bin> <out_int8.bin>   # positional args only
```

Strength testing — `teste/arena.cpp` is **not** in the build scripts. `teste/run_arena.py` compiles it twice (once per git ref) into `teste/bin/` and plays the two builds against each other with Elo + confidence interval:

```bash
python teste/run_arena.py --ref1 "" --ref2 main --games 200 --time 500 --threads 14
# --ref1 "" / None = current working tree including uncommitted changes
# --e1-nnue / --e2-nnue point each side at quantized weights
```

Web GUI (`gui_web/`): compiled `zquoridor.js`/`.wasm` are gitignored, but the bundled `gui_web/zquoridor.html` and root `index.html` are committed — `build_wasm` runs `build_standalone.py`, which regenerates both. Only rebuild after touching `rules.hpp`/`search.hpp`/`engine_wasm.cpp`; commit the regenerated bundles (that's what GitHub Pages serves). `build_wasm.bat` calls `C:\emsdk\emsdk_env.bat`; the `.sh` expects emsdk already sourced. Note the two scripts have **drifted**: the `.bat` exports more functions (NNUE toggles: `_qr_load_nnue_weights`, `_qr_set_eval_heuristic`, `_qr_eval_mode_is_nnue`) than the `.sh` — keep both `EXPORTED_FUNCTIONS` lists in sync when adding an export.

## Architecture

**`rules.hpp`** — `State` (two pawn cells, two 64-bit wall bitboards H/V, side to move, Zobrist hash), move generation, and `evalSimple`. Wall legality (both players must keep a path to goal) is the expensive part: a cheap geometric pre-filter rejects most candidates before the real check, which uses the rollback union-find in `dsu.hpp`. Four BFS variants share one engine: `hasPathToGoal`, `shortestPathLen`, `shortestPathTouchSlots`, `pathRobustness`.

**`search.hpp`** — negamax/alpha-beta, transposition table, killers + history, LMR+PVS, RFP+LMP, wall quiescence, 3-fold repetition with `CONTEMPT = -30`. Nearly every heuristic has a runtime toggle (`setLmrPvsEnabled`, `setRfpEnabled`, `setLmpEnabled`, `setQuiescenceEnabled`) so a benchmark or bisect can isolate it without recompiling — preserve that pattern when adding heuristics.

**`cat.hpp`** — Corridor Attention Table: per-cell "heat" computed once per node (2 BFS, not per wall candidate) measuring deviation from the opponent's optimal path. Drives wall ordering.

**BFS caching** is load-bearing for speed: `PlayerPathCache` (per node) plus `PlayerPathCacheTable` (~48MB, keyed on `wallsH/wallsV/pawnCell/player` — the wall *topology*, not the full position, so sibling nodes and transpositions share entries). Measured ~57% + ~5%. Anything that recomputes a BFS outside these caches is a regression.

**`endgame_race.hpp`** — when `wallsLeft[0]==0 && wallsLeft[1]==0` the wall topology is frozen and the game is an exact pawn race, solved instead of searched: a cheap disjoint-*region* gate plus exact retrograde DP over 81×81×2 states, cached per topology with a real-time budget (~3% of the move budget) so a cache miss storm can never make nodes/s worse than baseline. **Read the long header comment at the top of the file and Sections 4d/4e of `plano-additional.md` before changing anything here** — it documents four rounds of corrections, including a move-*choice* bug (not a value bug) at the real game root that lost most games while reporting correct evaluations.

**`nnue.hpp`** — 332 → 256 accumulator (SCReLU) → three independent heads: WL/outcome `256→32→1` (what search consumes), auxiliary `256→32→1` imitating `evalSimple` (training scaffold, to be removed once self-play comes from the net itself), policy `256→209` (move ordering). Everything is perspective-relative (`buildAccumulator(state, perspective)`); the net never knows "who is white". Features: 81+81 pawn, 64+64 wall, 21+21 one-hot BFS-distance buckets. The accumulator is always updated incrementally — no move triggers a full rebuild.

**Quantization is QAT**, not post-hoc: `QA=255`/`QB=64` are fixed *before* training and weights are clamped each optimizer step. **These constants appear in three places — `src/nnue.hpp`, `training/train_nnue.py` (`--qa`/`--qb`), and `training/quantize_nnue.py` — and must be changed together**, or `nnue_verify` parity breaks.

**Self-play binary format is the dataset.** `selfplay.exe` writes the exact struct the trainer reads; there is no preprocessing step. `TrainingSample` is 27 bytes, asserted in `training/read_selfplay.py` (`SAMPLE_DTYPE.itemsize == 27`) — changing the C++ struct requires updating that dtype in lockstep.

**Eval mode**: `Negamax::setEvalMode(EvalMode::Heuristic | EvalMode::NNUE)` selects `evalSimpleW` vs. the quantized net. NNUE mode requires weights loaded and maintains `nnueAccStack` incrementally across the search stack; heuristic mode leaves `accForSearch` null. Selfplay, arena, bench, and the WASM shell all expose this switch. Note `readme.md` Section 3/Phase B still says the NNUE is not plugged into search — that is now stale; the wiring exists.

## Conventions worth keeping

- New search heuristics get: a runtime toggle, a correctness test in `teste/` comparing against a reference search with the heuristic off, and a benchmark measuring nodes-to-equal-depth. Heuristic tests assert *agreement thresholds* (e.g. ≥85% score agreement, ≥90% on decisive positions, never an illegal move), not zero divergence — unlike `test_search_staging.cpp`, which must match its reference exactly.
- Eval-weight changes are meant to go through `teste/tune_spsa.cpp` (SPSA over the `evalSimple` weights; checkpoints to `spsa_checkpoint.txt`, run from repo root). Several thresholds (`QS_CRITICAL_*`, `RFP_MARGIN_*`, `LMP_COUNT_*`, `robustnessWeight`) are still uncalibrated placeholders inherited from another engine.
- Real strength claims come from `run_arena.py` games, not from nodes/s.
