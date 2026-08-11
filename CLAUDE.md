## Project

Zquoridor is a 2-player Quoridor engine (9×9, 10 walls each; 4-player variant is out of scope). Negamax + alpha-beta search, with an NNUE trained on self-play games. Sister project of chess engine **Zchezz**, mirroring its conventions.

`readme.md` is a concise, feature-level reference for external readers (install/build/run, capabilities).

`status.md` holds the technical reference: Future Plans & Roadmap, module/function specifications, design decisions, benchmarks, bugs fixed, uncalibrated values, and dated changelog.

**Language: comments, commit messages, and docs are in English.**

## Layout & build model

Header-only engine: everything in `src/` except `main.cpp`/`selfplay_main.cpp` is a `.hpp` included directly. No build system — each binary is a single `g++` invocation over one translation unit, with `-Isrc`. `teste/*.cpp` and `gui_web/engine_wasm.cpp` include the same headers.

Two build profiles:
- **Performance targets** (`src/main.cpp`, benchmarks, selfplay, tune_spsa, arena): `-O3 -std=c++17 -march=native -mavx2 -mfma`
- **Correctness tests** (`teste/test_*.cpp`, `nnue_verify.cpp`): `-O2 -std=c++17` only — no `-march=native`/AVX2, keeping parity reproducible.

`bin/` and `teste/bin/` are gitignored build outputs.

## Commands

Windows uses `build/*.bat` (MinGW-w64 `g++` on PATH; `build_selfplay.bat` hardcodes `C:\mingw64\bin`). Linux/macOS `build/*.sh` are exact equivalents. ARM/Termux uses `build/build_termux.sh` (minus `-mavx2 -mfma`).

```bat
build\build_all.bat              :: bench + tests + selfplay + tune_spsa (stops at first error)
build\build_all.bat wasm         :: same plus WASM (needs emsdk active)
build\build_tests.bat            :: correctness suite only
build\build_bench.bat            :: performance benchmarks only
```

Running tests (individual binaries):
```bat
bin\test_rules_sanity.exe
bin\test_search_staging.exe
bin\test_move_ordering.exe
bin\test_endgame_race.exe
bin\test_lmr_pvs.exe
bin\nnue_verify.exe data\nnue\nnue_weights.bin data\nnue\nnue_weights_int8.bin
```

Self-play data generation:
```bat
build\build_selfplay.bat
bin\selfplay.exe --games 20000 --chunk-games 2000 --threads 12 --time-ms 100 --mc-mode ^
    --out "data\selfplay\gen5-montecarlo\selfplay_{shard:03d}.bin"
python training\run_selfplay.py   :: orchestrator; edit CONFIG at the top
```

NNUE training & quantization (from `training/`):
```bash
python train_nnue.py --data ../data/selfplay/*.bin --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots
python quantize_nnue.py <in_f32.bin> <out_int8.bin>
```

Strength testing:
```bash
python teste/run_arena.py --ref1 "" --ref2 main --games 200 --time 500 --threads 14
# --ref1 "" / None = current working tree including uncommitted changes
```

Web GUI (`gui_web/`):
`build_wasm` runs `build_standalone.py` to regenerate bundled HTML files. Rebuild after touching `rules.hpp`/`search.hpp`/`engine_wasm.cpp`. Keep `EXPORTED_FUNCTIONS` in `build_wasm.bat` and `build_wasm.sh` in sync.

## Architecture

- **`rules.hpp`**: `State` (pawn cells, 64-bit wall bitboards H/V, turn, Zobrist hash), move generation, `evalSimple`. Wall legality uses a geometric pre-filter followed by rollback union-find in `dsu.hpp`. Four BFS variants share one engine (`hasPathToGoal`, `shortestPathLen`, `shortestPathTouchSlots`, `pathRobustness`).
- **`search.hpp`**: Negamax alpha-beta, transposition table, killers + history, LMR+PVS, RFP+LMP, wall quiescence, 3-fold repetition with `CONTEMPT = -30`, policy-assisted move ordering (gated by `policyOrderingMinDepth`, default 3). All heuristics have runtime toggles.
- **`cat.hpp`**: Corridor Attention Table (CAT) heat map for wall move ordering.
- **BFS Caching**: `PlayerPathCache` (per node) + `PlayerPathCacheTable` (~48MB, keyed on wall topology) speed up BFS calls (~57% + ~5%).
- **`endgame_race.hpp`**: Exact pawn-race solver when `wallsLeft==(0,0)` using disjoint-region gate and retrograde DP over 81×81×2 states with real-time budget.
- **`nnue.hpp`**: 354 → 256 accumulator (SCReLU) → two heads: WL outcome (`256→32→1`) and Policy (`256→209`). Canonical perspective-relative features (81+81 pawn, 64+64 wall, 21+21 BFS distance, 11+11 remaining walls). Incremental accumulator updates.
- **Quantization (QAT)**: `QA=255`/`QB=64` fixed during training; weights clamped each step. Constants in `nnue.hpp`, `train_nnue.py`, and `quantize_nnue.py` must stay in sync.
- **Self-play format**: `TrainingSample` is 32 bytes packed, storing mirrored canonical state, `evalNNUE` (network win probability), `mover`, `ownCatTotal`, `oppCatTotal`, and policy target. Fast Monte Carlo policy-temperature sampling (`--mc-mode`) generates diverse openings without search overhead. Thread memory uses stack arrays to prevent TLS destructor heap corruption.
- **Eval mode**: `Negamax::setEvalMode(EvalMode::Heuristic | EvalMode::NNUE)` selects `evalSimple` vs quantized net.

## Evaluation: what each stage uses

Five different things all get called "the evaluation" depending on where you look, and they don't share a sign convention, perspective, or range:

| Stage | Symbol / entry point | What it computes | Perspective / sign | Range |
| --- | --- | --- | --- | --- |
| Search | `nnueEvalInt` (`nnue.hpp`), consumes `forwardValueWLQuant` | NNUE WL head raw logit, scaled ×`NNUE_EVAL_SCALE` (200) to an int on the same rough scale as `evalSimple` | mover-relative: positive = side to move is ahead | ~[-30000, 30000]; mate/certain-win positions saturate near the extremes |
| Heuristic fallback | `evalSimple` (`rules.hpp`), used only in `EvalMode::Heuristic` or via `--heuristic` on selfplay/arena/WASM | hand-tuned material + positional heuristic, uncalibrated weights | mover-relative | ~[-600, 600] in typical midgame positions |
| Self-play `.bin` | `TrainingSample::evalNNUE` (`selfplay.hpp`, `teste/arena.cpp`) | NNUE WL head's own opinion of the position, sigmoid'd to a win probability, computed via `buildAccumulatorQuant`+`nnueWinProbQuant` **before** the move is chosen | **absolute color**, not mover-relative: 0 = Black certain win, `EV_SCALE` (65535) = White certain win — chosen this way so the field can be read without needing `mover` alongside it | `uint16_t` 0..65535 (`EV_SCALE`) |
| Training target | `wl_target` in `to_chunk_tensors` (`train_nnue.py`) | blend of real game outcome and recorded `evalNNUE`: `wl_target = k · game_result_prob + (1-k) · ev_prob`, where `ev_prob` is `evalNNUE` reprojected via `mover` field, `k` per data source in `DATA_SOURCES_DEFAULT` (default `1.0`) | mover-relative probability (`game_result_prob = (game_result+1)/2`) | float `[0, 1]` |
| HTML/WASM display | `formatEval`/`evalToWhitePercent` (`gui_web/app.js`) | sigmoid of the raw search score (White-perspective) / 200 | **absolute color**, always: 0% = Black certain win, 100% = White certain win, 50% = balanced | `0–100%`, rounded to an integer |

Practical notes:
- `k=1.0` is required for any `.bin` recorded before `evalNNUE` was introduced (where that 2-byte field held heuristic `evalSimple` scores). `k=1.0` makes `wl_target` reduce to `game_result_prob` exactly.
- `k<1.0` is a bootstrapping knob letting positions vote with network opinion during self-play.

## Conventions

- New search heuristics get: runtime toggle, correctness test in `teste/`, and benchmark.
- Continuous-parameter tuning (`contempt`/`policyOrderScale`/`catScoreScale`) uses `teste/tune_spsa.cpp` (`--mode spsa|sweep-mindepth|hybrid`).
- Real strength claims require `run_arena.py` matches, not nodes/s alone.
- Log bug fixes, regressions, architectural decisions, and uncalibrated values in `status.md` with dated changelog entries.

