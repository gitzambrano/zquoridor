# NNUE Weights and Model Releases

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
