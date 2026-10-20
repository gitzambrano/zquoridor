# NNUE Weights and Model Releases

**Active search release: zQuoridor 4.03.** Active NNUE checkpoint:
**4.01 temporal model**, 858 features / 512 hidden / 209 policy outputs.
The native and web production INT8 SHA-256 is
`45fa3ea7db3127a0b907a23d268b0c55c2020e1018513d54bb920cdbe134abe0`.
The production 4.03 Lazy+Sparse search does not depend on Apex Policy25.
Apex Policy25 is a separate future NNUE candidate, not shipped with 4.03.


This directory stores production weights and evaluated experimental networks for Zquoridor.

## File Types

* `nnue_weights.bin`: Active production float32 network weights used by the engine.
* `nnue_weights_int8.bin`: Active production quantized int8 network weights used by native search and WebAssembly builds.
* `student.architecture.json`: Architecture specification accompanying network binaries.

## Subdirectories

* `champions/`: Historical milestone weights promoted through head-to-head evaluation gates.
* `experiments/`: Versioned experimental networks produced by recent training campaigns, including width ablations (512 vs 768) and replay mixing studies.

## Architecture

Current networks employ the `multipath_phase_contact_bucketed` architecture:
* Feature count: 858 input features capturing pawn positions, wall placements, and local contact adjacencies.
* Value head: Bucketed outputs indexed by total walls remaining on the board.
* Inference: Quantized accumulator evaluation with squared clipped linear activation.


## Separate Apex Policy25 candidate (NOT in 4.03)

The separately tested candidate changes **only the 209-logit INT8 policy head**. The
feature accumulator and all six value heads remain exactly the same as in 4.01.
The original float32 checkpoint is not an Apex checkpoint, so use the supplied
Apex INT8 weights directly; re-quantizing the original float32 file would revert
the policy.

Expected Apex SHA-256: `fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`.
Previous baseline: `45fa3ea7db3127a0b907a23d268b0c55c2020e1018513d54bb920cdbe134abe0`.

For a production promotion, native weights and browser package
(`gui_web/zquoridor.data`) must have the same hash. The standalone builds
(`gui_web/zquoridor.html`, `docs/index.html`) must be regenerated with
`python gui_web/build_standalone.py`. Never update a release manifest alone.

**Policy compatibility**: Sparse inference is generic to weights with the same
858-input, 512-hidden, 209-logit quantized policy schema, output action
indexing and side canonicalization. A different hidden width, feature set or
output action vocabulary is a different model architecture and requires a
matching engine build and new exact-policy parity tests.
