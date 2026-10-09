# Experimental archive

This directory retains local network checkpoints, search profiles, and campaign
records. Production uses the 858-input contact network and dense edge deltas.
The release decision and complete evidence history are in `docs/plan.md`.

| Directory | Contents |
| --- | --- |
| `releases/v3_01_504/` | Previous production headers, build scripts, adapter, and 504 weights |
| `experiments/` | Existing training experiments, component networks, and model soups |
| `edge_acc/` | Archived Edge source, runners, and verification tools |
| `benchmarks/` | Original Edge campaigns, frozen references, manifests, logs, and games |
| `artifacts/` | New binaries from an explicit archived profile build |

`profiles.json` records each network and search combination. The profile builder
does not replace production weights, start games, or change the active release.

```powershell
py -3 experimental/build_profile.py --list
py -3 experimental/build_profile.py --profile 504_off
py -3 experimental/build_profile.py --profile 858_v3_node_dense_bfs
```

Generated datasets, weights, checkpoints, binaries, and campaign records remain
outside Git. Source and selected provenance metadata remain in the repository.
Historical Windows paths use directory junctions to these artifact directories.
These aliases preserve absolute paths in existing manifests and local scripts.
New experiments use this directory directly. They do not require a branch or a
Git worktree.

To restore version 3.01, copy the archived headers and both archived weight files
to their production paths. Restore the archived build scripts and adapter.
Remove the 4.0 configuration header if any restored source includes that header.
Rebuild native and WebAssembly artifacts. Verify the 504 weight checksum against
the registry before use. This procedure does not run automatically.

The previous production commit is `fe5d5fec9451c0258d0a8cfee7cc07eafad5b9a2`. Git retains both release weight
files at `data/nnue/`. Local archives contain the same files outside Git.
A fresh clone can recover these weights from that commit without a branch or
worktree. Historical evidence statements in the archive describe version 3.01.
The current production state is version 4.0 in `docs/plan.md`.

The archived Edge headers and adapter preserve the evaluated implementation.
Generic tools and training pipelines remain canonical at the repository root.
Local archive copies of generic tools remain outside Git. The profile builder
uses only the archived headers, adapter, profile definitions, and local weights.
