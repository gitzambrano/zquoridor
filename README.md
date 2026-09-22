# Zquoridor 2.02

Zquoridor is a high-performance engine for two-player Quoridor on the standard 9×9 board with 10 walls per player.

The production engine uses **hybrid MCTS with alpha-beta**. A PUCT/MCGS search graph is guided by an NNUE value-and-policy network, while alpha-beta remains integrated for tactical and endgame search and is also available as a standalone search mode. The engine includes transposition reuse, persistent search trees, opponent-root pondering, repetition handling, fast wall-legality checks, a self-play/training toolchain, benchmark arenas, and a WebAssembly browser interface.

**Current release: 2.02**

Play in the browser: https://gitzambrano.github.io/zquoridor/

## Highlights

- Hybrid MCTS/MCGS search with alpha-beta support.
- PUCT policy guidance from the NNUE policy head.
- Transposition graph with tree reuse between moves.
- Opponent-root pondering: work done while the opponent thinks is reused after the real reply.
- Pure Negamax alpha-beta remains available with `--no-mcab`.
- Rollback-DSU wall legality and cached path calculations.
- Exact pawn-race endgame support when both players have no walls.
- Quantized NNUE inference for native and WebAssembly builds.
- UCI-style text protocol for external GUIs, arenas, and automation.
- Browser engine runs search and pondering in a Web Worker so the UI stays responsive.
- Self-play, training, quantization, testing, and strength-measurement tools are included.

## Search architecture

The default search is a hybrid of Monte Carlo tree/graph search and alpha-beta:

1. The NNUE policy head provides priors for legal moves.
2. PUCT selects branches in a persistent MCTS/MCGS graph.
3. Transpositions can share search information across equivalent states.
4. NNUE value evaluation supplies the default leaf signal.
5. Alpha-beta is available inside the hybrid search for selected tactical/endgame cases and as a complete fallback search.
6. The tree, transposition table, and evaluation caches are reused across moves.
7. During the opponent's turn, opponent-root pondering develops the current root. When the reply arrives, the engine reroots onto the matching child and keeps useful work.

The main production defaults live in `src/mcab.hpp` and `src/search.hpp`.

## NNUE

Version 2.02 uses the production **multipath + phase** network:

- **504 sparse input features**
- **512-neuron SCReLU hidden layer**
- **Outcome head:** `512 → 32 → 1`
- **Policy head:** `512 → 209`
- Quantization-aware int8 inference

The feature set combines:

- pawn locations;
- horizontal and vertical wall occupancy;
- shortest-path distance buckets;
- remaining-wall buckets;
- race margin and wall-stock interactions;
- game-phase features;
- cheap multi-path and contact geometry features.

The network is perspective-relative and supports incremental accumulator updates during search.

## Build and install

### Requirements

For the native engine:

- C++17 compiler (`g++` or compatible)
- Python 3 for helper scripts and some tools

For the browser build:

- Emscripten / emsdk
- Python 3

Clone the repository:

```bash
git clone https://github.com/gitzambrano/zquoridor.git
cd zquoridor
```

### Linux / macOS

```bash
chmod +x build/*.sh
./build/build_uci.sh
```

The text engine is written to:

```text
bin/zquoridor
```

Build the complete native toolset:

```bash
./build/build_all.sh
```

### Windows

With MinGW-w64 `g++` available on `PATH`:

```bat
build\build_uci.bat
```

Build the complete native toolset:

```bat
build\build_all.bat
```

### WebAssembly

Activate emsdk first, then run:

```bash
./build/build_wasm.sh
```

On Windows:

```bat
build\build_wasm.bat
```

The build produces the WASM runtime and regenerates the standalone browser bundle and root GitHub Pages page.

## Text protocol

Zquoridor exposes a compact **UCI-style** line protocol through `bin/zquoridor`.

Start it with the production NNUE:

```bash
bin/zquoridor --nnue data/nnue/nnue_weights_int8.bin
```

Type:

```text
help
```

to print the supported commands.

Core commands:

```text
uci
isready
ucinewgame
position startpos [moves <move> ...]
go movetime <ms>
go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <n>]
ponder movetime <ms>
help
quit
```

Move notation uses board coordinates:

- `e2` — pawn move
- `d4h` — horizontal wall
- `d4v` — vertical wall

Example:

```text
uci
isready
ucinewgame
position startpos moves e2 e8
go movetime 200
```

The engine responds with an `info` line followed by:

```text
bestmove <move>
```

### Pondering

For host-controlled pondering, set the position through the engine's last move and grant an opponent-time budget:

```text
position startpos moves ...
ponder movetime 200
```

When the opponent move arrives, send the extended position and then `go`. The engine reuses the matching subtree when available.

The browser performs the same idea automatically in short Web Worker slices, so pondering does not block the UI thread.

## Browser GUI

The browser version uses the same rules, NNUE, and hybrid search core as the native engine.

For local development:

```bash
cd gui_web
python3 dev_server.py 8000
```

Then open:

```text
http://127.0.0.1:8000/style.html
```

Search runs in a Web Worker. Opponent-root pondering is also confined to that worker and uses short bounded slices.

## Tests

Build the native tests:

```bash
./build/build_tests.sh
```

Representative checks include rules, search staging, move ordering, repetition, MCTS/MCGS, time management, and NNUE parity.

Browser tests live in `gui_web/` and use Playwright for real-browser validation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Header-only engine core: rules, search, NNUE, MCTS/MCGS, DSU, endgames |
| `tools/` | Arena, self-play, protocol adapters, experiment and benchmark tools |
| `training/` | NNUE training, quantization, and data utilities |
| `tests/` | Native correctness and regression tests |
| `benchmarks/` | Performance and diagnostic benchmarks |
| `build/` | Native, protocol, arena, and WebAssembly build scripts |
| `gui_web/` | Browser UI, Web Worker, WASM binding, and browser tests |
| `docs/` | Project plan, technical status, dataset catalog, and operational documentation |
| `data/` | Production weights and data assets |
| `checkpoints/` | Versioned model checkpoints and manifests |
| `results/` | Recorded experiment and benchmark results |

## Version 2.02 changelog

- Added an exact root win-in-1 fast path: when a legal pawn move wins immediately, the engine returns it before MCAB, alpha-beta endgame work, policy evaluation, or tree expansion.
- Preserved the exact no-wall pawn-race solver and MCAB equivalence validation paths.
- Fixed the browser acceptance test so board flip state is restored through the production path before direct wall-gesture testing.
- Revalidated native search, WebAssembly, browser interaction, and release packaging.

## Version 2.00 changelog

- Promoted the MCGS/transposition-graph search path.
- Added opponent-root pondering and subtree reuse.
- Added persistent browser-worker search state and bounded background pondering.
- Updated the production NNUE to the 504-input, 512-hidden multipath + phase architecture.
- Improved adaptive real-clock search behavior and repetition handling.
- Added an official UCI-style text-engine build target and `help` command.
- Reorganized project documentation under `docs/`.
- Refreshed native and WebAssembly build/test paths.

## Documentation

- [Technical status](docs/status.md)
- [Project plan](docs/plan.md)
- [Dataset catalog](docs/datasets.md)
- [Engine lab](docs/engine-lab.md)
- [Local teaching tools](docs/local-teaching.md)

## Acknowledgements

[**Titanium**](https://github.com/titaniummachine1/titanium-engine) and [**Claustrophobia**](https://github.com/Plaaasma/Claustrophobia) have been valuable reference engines for the project. Zquoridor has borrowed and adapted several search and engineering ideas from their public implementations, and both engines are used as external benchmarks when measuring Zquoridor's progress.
