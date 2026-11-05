# MCGS production hot-path performance study: 2026-11-04

## Reproducibility

- Baseline: `main` commit `bdffab86ad9e7f21b64b8841fa7f1620668f3fe9` (zQuoridor 4.10).
- Experimental branch: `exp/perf-main-hotpath-20261104`.
- Runner: `benchmarks/bench_main_hotpath_20261104.cpp`.
- Initial workflow: https://github.com/gitzambrano/zquoridor/actions/runs/41951050378
- Corrected workflow: https://github.com/gitzambrano/zquoridor/actions/runs/41951371776
- All six jobs in the corrected workflow completed successfully.
- NNUE checksum: `fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`.
- Production search and weights are unchanged. The benchmark creates deterministic, generated legal positions with a fixed seed. The benchmark does not represent a complete competition corpus.

## Key findings

The numbers below are measurements on GitHub-hosted Linux CPUs. Each comparison within one job uses the same virtual machine. Do not compare absolute speed across independent jobs.

| Study | Setting | Simulations/s | Other evidence |
| --- | --- | ---: | --- |
| Compiler | Portable A | 84,031 | 17,500 expansions and 49,406 simulations |
| Compiler | Portable B | 89,358 | Same work |
| Compiler | Native AVX2/FMA A | 118,512 | Same work |
| Compiler | Native AVX2/FMA B | 118,875 | Same work |
| Edge Delta | Dense enabled | 104,633 | Same fixed-node search trajectory |
| Edge Delta | Dense disabled, feature delta retained | 83,046 | Same trajectory |
| Edge Delta | Feature Delta and BFS Replay disabled | 76,321 | Same trajectory |
| Eval cache | ON A / ON B | 63,651 / 64,321 | No policy or value cache hits in these graph-enabled fixed-node runs |
| Eval cache | OFF A / OFF B | 63,059 / 62,101 | Same 49,406 simulations |
| Graph | ON A / ON B | 95,335 / 94,474 | 49,406 simulations for 17,500 expansions |
| Graph | OFF A / OFF B | 109,234 / 108,742 | 55,290 simulations for 17,500 expansions |
| Timed Lazy | Lazy, 200 ms, uncapped diagnostics | 80,845 | 64,821 simulations; 22,695 expansions |
| Timed Eager | Eager, 200 ms, uncapped diagnostics | 68,282 | 54,753 simulations; 54,511 expansions |

The native compiler comparison indicates approximately 37% higher simulation throughput on the runner. This result requires confirmation on each release CPU and in the full executable. Native flags must not replace the portable distribution build.

Dense Edge Delta improves throughput by approximately 26% relative to the feature-only option and 37% relative to the no-delta option in this study. This observation supports retaining the current production implementation.

The policy and value caches produced zero hits in graph-enabled fixed-node tests. They still incurred lookup and insertion costs. However, cache ON and OFF times were close. The study does not justify removal without additional trials.

The graph-disabled search is not equivalent to the production graph search. The number of simulations, visited positions, cache hits, and backup semantics differ. Therefore, do not promote graph OFF based on throughput.

The Lazy and Eager search schedules differ in what counts as a simulation and when expansion occurs. The timed comparison shows higher simulation throughput for Lazy, not established Elo superiority. The timed diagnostic used a high expansion ceiling to avoid an early node-cap stop.

## Adaptive clock check

The first workflow used an incorrect fixed 256-expansion cap in its adaptive study. Do not use its adaptive timing numbers.

The corrected study applies `zqtime::allocate` to a 180,000 ms + 2,000 ms Fischer clock at the fixture ply. It uses the resulting optimum and maximum budgets with the normal Adaptive Time engine. These are two independent root searches, not full multi-move games.

| Measure (two positions combined) | Eval cache ON | Eval cache OFF |
| --- | ---: | ---: |
| Wall time | 16,451.6 ms | 16,450.0 ms |
| Expansions | 286,409 | 287,592 |
| Simulations | 734,649 | 738,634 |
| Simulations/s | 44,655 | 44,902 |
| Policy cache hits | 0 | 0 |
| Value cache hits | 9 | 0 |
| Graph transposition hits | 166,455 | 166,744 |
| Approximate peak tree memory | 660.8 MiB | 658.6 MiB |

The graph ON cache has 9 value-cache hits out of more than 723,000 value misses. This observation suggests low cache reuse in this workload. It is not an Elo result.

The peak tree memory estimate is a high-priority investigation target. It counts pool capacity and edge buffers but excludes allocator fragmentation. A compact representation for unexpanded nodes may reduce its cost.

## Next experiments

1. Use hardware counters and scoped timers to measure PUCT selection, graph lookup, NNUE feature update, and memory stalls.
2. Implement an optional compact node layout behind a build-time flag. Preserve the exact legal-action order, graph identity, and backup semantics.
3. Test an incremental node value sum in PUCT against the reference implementation. Check float rounding differences and tree fingerprints.
4. Test cache ON and OFF on reused trees. Include realistic Center Rush openings and varied wall stocks.
5. Verify native release builds and 200 ms plus 3+2 full games against the frozen baseline before any production change.

## Decision

Do not promote a search change or claim Elo improvement based on this screen. Keep the established Lazy, graph correction, and Dense Edge Delta defaults. Prioritize memory-layout profiling and CPU-targeted builds.
