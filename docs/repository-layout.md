# Repository layout

| Directory | Role |
| --- | --- |
| `src/` | The current header-only engine implementation. |
| `gui_web/` | The browser UI and its engine integration. |
| `build/` | Native and WebAssembly build entry points. |
| `training/` | Training, quantization, and campaign implementations. |
| `tools/` | Local benchmarks, arena, self-play, and data utilities. |
| `colab/` | All current cloud transport, monitoring, remote workload, and notebook implementations. |
| `tests/` | Automated regression and contract checks. |
| `docs/` | Roadmap, runner catalog, and repository contracts. |
| `artifacts/` | Ignored generated logs, audit reports, caches, and local environments. |
| `data/` and `results/` | Engine artifacts and ignored datasets or experiment outputs according to `.gitignore`. |
| `experimental/` | Isolated experimental lines and frozen release snapshots. |

## Colab organization

Both Zquoridor and Zchezz use the same cloud layout:

```text
colab/
  cli/                    terminal commands and artifact monitoring
  playwright/             reusable browser commands and monitoring
    diagnostics/          targeted historical inspection and repair scripts
  remote/                 implementations that execute inside a runtime
  notebooks/              notebook entry points
  config.py               public account slots and workload settings
  private.py              external local identity configuration
  layout.json             directory roles and compatibility map
```

The current implementation of each cloud runner is inside `colab/`.
The former `tools/selfplay`, `tools/arena`, and `training` Colab paths remain
small compatibility entry points for existing notebooks. Their implementations
are in `colab/remote`. `layout.json` records each forwarding path.
The local `scripts/colab` package is a legacy import adapter. Its browser profile
data is private local state, not a second implementation of the orchestration.

The native engine and local training interfaces retain their established paths.
Do not rename a frozen snapshot or delete a matching file solely because its
bytes equal the current source. The snapshot preserves an independent experiment.
Agent configuration copies also remain necessary for the respective clients.

## Repeatable checks

Run `python tools/audit_repository.py` for source layout and duplicate classification.
Run `python -m colab.privacy_check` before staging cloud changes.
Reports stay in ignored artifact directories. Private identities and tokens stay
in the external local configuration. Neither check rewrites Git history.
