# zQuoridor 4.10

**[Play Zquoridor in your browser](https://gitzambrano.github.io/zquoridor/)**

## What it is

Zquoridor is an engine for two-player Quoridor on the standard 9×9 board with
10 walls per player. It can play through a text protocol, in automated arenas,
or directly in the browser.

## Engine features

- Hybrid PUCT-guided MCTS/MCGS graph search with alpha-beta support for tactical
  and endgame positions. Version 4.10 retains uniform Lazy Expansion + Sparse Policy for both fixed move times
  and all native clocks (1+1, 3+2, 5+3, etc.), with corrected accounting for
  tree preparation and reuse.
- Multi-bucketed value and policy guidance from a compact quantized NNUE
  (`multipath_phase_contact_bucketed:512` with 6 wall-count value heads and 2-layer deep MLPs).
- Persistent tree reuse, transpositions, repetition handling, and opponent-root
  pondering.
- Adaptive long-clock node scaling: volatile roots may expand to 96 nodes/ms with
  a 1.28M-node ceiling; stable/non-volatile roots retain the 32 nodes/ms / 640k budget.
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

The production network is `multipath_phase_contact_bucketed:512`. It has 858 sparse inputs, a
512-unit SCReLU accumulator, 6 wall-count value heads with 2-layer MLPs, and a 209-action
policy head. The inputs describe both pawns, wall occupancy, shortest-path and
wall-stock buckets, race margin, game phase, contact features, and inexpensive multi-path
geometry. It is trained with quantization-aware training and evaluated as int8
in the native and WebAssembly engines.

The [architecture lab](experimental/experiments/architecture-lab-v401/README.md)
defines eight experimental networks and one control. Run
`python training/train_nnue.py` to inspect the default plan. The same entry
point prepares the shared mixture and trains selected models through
`--cli-command prepare` and `--cli-command train`. Read the
[data contract](docs/training-data.md) for source quality and target semantics.

## Run and validate

```bash
bin/zquoridor --nnue data/nnue/nnue_weights_int8.bin
```

The UCI-style protocol supports `uci`, `isready`, `ucinewgame`, `position`,
`go movetime <ms>`, clock-based `go`, `ponder [clock] movetime <ms>`, `help`, and
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

## Release 4.10 — Search-guided policy update

Production preserves the 4.03 unified Lazy+Sparse search, tree reuse, adaptive clocks and pondering. The quantized 858-input/512-hidden NNUE has a newly distilled 209-action policy head; the accumulator and six bucketed value heads are unchanged. Production INT8 SHA-256: `fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`.

The retained float32 file still has the previous policy, so do not re-quantize it over production INT8 weights. Browser and native assets must embed the exact same checkpoint. See [4.10 release notes](docs/releases/4.10.md).

On the earlier 4.02 search, 400 paired games scored 54.13% against the prior policy and 65.38% against Claustrophobia CPU, versus 61.13% for the prior policy. This does not prove superiority at 3+2 or 5+3.

## Release 4.03 — Unified search and clock correctness

**Production search:** Lazy Expansion + exact legal-action Sparse Policy at
all time controls. The UCI front end supports both `go movetime 200` and
`go wtime ... btime ... winc ... binc ...`; only the allocated time and
Adaptive Time differ. Unlike 4.02, changing the protocol does **not**
switch search algorithms or clear the existing MCGS tree.

The native clock now starts *before* preparing, compacting or rerooting the
graph, preventing reused-tree preprocessing from silently consuming time
outside the move budget. This is independent of named time controls.
`--eager-search` exists solely as a diagnostic comparison flag.

**NNUE policy/value checkpoint:** unchanged from the 4.01 858-input,
512-hidden temporal network. No Policy weights are promoted in 4.03.
Browser/WASM must ship that identical original checkpoint. Exact long-clock
performance and timing were checked locally, but **no competitive 3+2/5+3
Elo improvement is claimed**.

## Release 4.02 — Search (historical)

- In version 4.02, `go movetime` enabled **Lazy Expansion + Sparse Policy**.
- In version 4.02, native clocks used **Eager Expansion + Sparse Policy**. Both paths are superseded by unified Lazy in 4.03.
- Sparse Policy computes the same quantized logits as dense policy for the **legal actions**, using the model's existing canonical 209-action map. It is **not tied to any particular set of policy-head weights**.
- The release gate verified legal-policy logits/priors and the two time-control modes. Clock forfeits and invalid-game records were absent in the completed short external checks.
- The shipped weight file is still the **4.01 temporal, 858-input / 512-hidden NNUE** until a new policy-head checkpoint is installed and all native and web assets are synchronized. The search release did **not** replace the value network.
- Policy Policy25 is a separate **policy-only candidate**; promote it with the integrity-checked checkpoint and rebuild of the standalone bundle. The original float32 weights do **not** automatically represent its policy head.

See [NNUE release notes](data/nnue/README.md) for weight identity and compatibility.

## Release 4.01

The release promotes the selected epoch-97 network from a 100-epoch campaign.
Training uses 1,598,963 unique states, weighted averages for duplicate targets,
temporal search values, surprise weights, and a training-only auxiliary policy.
The architecture and search defaults remain compatible with version 4.0.

The candidate scored 60.5% against version 4.0 across 200 paired games at
200 ms per move. The estimated gain is +74 Elo, with a paired bootstrap
95% interval of +30 to +119 Elo. The complete screen contains 600 games
with no failures. External scores improved in both opening books against
Claustrophobia CPU and Titanium, but the improvement intervals include zero.
Claustrophobia used approximately 275 ms of actual search despite a 200 ms
request. These results do not certify equal elapsed time against Claustrophobia
or describe its GPU strength. See [the release evidence](data/nnue/release_evidence.json)
and [the project measurements](docs/plan.md).

## Release 3.0

- Updates production weights to the central fine-tune model (`multipath_phase_bucketed:512`), trained on 21.1M stored-search and balanced canonical positions.
- Resolves historical Center Rush and central opening weaknesses across all external benchmark suites.
- Outperforms historical baselines with +64.1 Elo against Claustrophobia (59.13% overall, including 59.00% on Center Rush) and +141.2 Elo against Titanium (69.00% overall).
- Demonstrates statistically verified superiority over the previous production baseline in head-to-head testing (55.50%, +38.4 Elo).
- Updates the WebAssembly engine and standalone browser bundle with the new quantized int8 weights.

## Release 2.10

- Promotes the `multipath_phase_bucketed:512` unified champion architecture with 6 wall-count value heads and 2-layer MLPs.
- The engine returns an immediate winning pawn move before any search work.
- The exact no-wall pawn-race solver and the MCAB equivalence checks are
  maintained.
- WebAssembly engine and standalone browser bundle updated with production weights.

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

## Version 4.10 configuration

Production uses dense edge feature deltas and BFS replay with the 858-input,
512-hidden temporal network. The 4.03 **search baseline** is uniform Lazy+Sparse at both fixed time
and native clocks, with corrected accounting for reroot preparation. The independent
NNUE checkpoint is identified by `data/nnue/release.json` and the SHA-256 of
`data/nnue/nnue_weights_int8.bin`; search version and network checkpoint
must not be conflated. Archived networks and search profiles reside in
`experimental/`; the 768-hidden experiment remains outside production.

## Repository organization and Colab operations

See [repository layout](docs/repository-layout.md) for source roles and
[Colab operations](colab/README.md) for the CLI and browser transports.
All seven accounts use private local identity configuration. Source organization
and privacy checks are available through `tools/audit_repository.py` and
`python -m colab.privacy_check`.
