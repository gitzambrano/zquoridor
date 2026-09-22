# Zquoridor 2.00

**[Play Zquoridor in your browser](https://gitzambrano.github.io/zquoridor/)**

## What it is

Zquoridor is an engine for two-player Quoridor on the standard 9×9 board with
10 walls per player. It can play through a text protocol, in automated arenas,
or directly in the browser.

## Engine features

- Hybrid PUCT-guided MCTS/MCGS graph search with alpha-beta support for tactical
  and endgame positions.
- Value and policy guidance from a compact quantized NNUE.
- Persistent tree reuse, transpositions, repetition handling, and opponent-root
  pondering.
- Fast wall-legality checks and exact pawn-race support when both players have
  no walls.
- UCI-style protocol for external GUIs, benchmarks, and automation.
- Native C++ engine plus the same core compiled to WebAssembly; browser search
  and pondering run in a Web Worker.
- Self-play, teaching, training, quantization, and paired benchmark tools.

## Download and build

Clone the repository, then build the text engine. You need a C++17 compiler
(`g++` or compatible) and Python 3.

```bash
git clone https://github.com/gitzambrano/zquoridor.git
cd zquoridor
chmod +x build/*.sh
./build/build_uci.sh
```

On Windows, install MinGW-w64, put `g++` on `PATH`, then run:

```bat
build\build_uci.bat
```

The executable is written to `bin/zquoridor` (`bin\zquoridor.exe` on Windows).
Build all native tools with `build/build_all.sh` or `build\build_all.bat`.
WebAssembly builds additionally require Emscripten; run `build/build_wasm.sh`
or `build\build_wasm.bat` after activating emsdk.

## NNUE architecture

The production network is `multipath_phase:512`. It has 504 sparse inputs, a
512-unit SCReLU accumulator, a WL value head (`512 → 32 → 1`), and a 209-action
policy head. The inputs describe both pawns, wall occupancy, shortest-path and
wall-stock buckets, race margin, game phase, and inexpensive multi-path/contact
geometry. It is trained with quantization-aware training and evaluated as int8
in the native and WebAssembly engines.

## Run and validate

```bash
bin/zquoridor --nnue data/nnue/nnue_weights_int8.bin
```

The UCI-style protocol supports `uci`, `isready`, `ucinewgame`, `position`,
`go movetime <ms>`, clock-based `go`, `ponder movetime <ms>`, `help`, and
`quit`. Pawn moves use coordinates such as `e2`; walls use `d4h` or `d4v`.

```bash
./build/build_tests.sh
```

Native tests cover rules, search, repetition, time management, self-play I/O,
and NNUE parity. Browser tests live in `gui_web/` and use Playwright.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Engine core: rules, search, NNUE, MCTS/MCGS, DSU, endgames |
| `tools/` | Arenas, self-play, protocol adapters, experiments, benchmarks |
| `training/` | NNUE training and data utilities |
| `build/`, `tests/` | Build scripts and native regression tests |
| `gui_web/` | Browser UI, worker, WASM binding, browser tests |
| `docs/` | Technical status, roadmap, and operational guides |
| `data/` | Production weights and local data assets |

## Documentation

- [Project state and roadmap](docs/plan.md)
- [Engine lab](docs/engine-lab.md)
- [Local teaching tools](docs/local-teaching.md)
- [Script catalog](docs/scripts.md)

## Acknowledgements

- [Claustrophobia](https://github.com/Plaaasma/Claustrophobia)
- [Titanium](https://github.com/titaniummachine1/titanium-engine)

Both projects are valuable public reference engines and external benchmark
opponents for Zquoridor.
