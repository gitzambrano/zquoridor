# Policy target from the MCab root visit distribution

Date: 2026-08-25
Status: task (a) done and measured. Task (c) is on hold, see "Result of
task (a)" below.

## Result of task (a), 2026-08-26

Task (a) is complete and it produced a negative result.

The mask excluded 26,725,366 of 66,458,326 positions, which is 40.2
percent, from the policy loss. Training ran 120 epochs and reached a best
`val_loss` of 1.9028 at epoch 114. The arena then measured 1000 games at
150 ms for each move, with the same local code on both sides so that the
NNUE weights were the only variable.

The candidate won 466 games, the production network won 464, and 70 games
drew. The difference is **+0.7 Elo with a margin of +/-20.8**, which the
arena reports as inconclusive.

The conclusion is that the self-imitating policy target is real, and that
removing it changes nothing measurable at this time control. Effects
larger than approximately 20 Elo are ruled out.

This weakens the case for task (c), and it does not settle it. Task (a)
deletes samples. Task (c) replaces the target with the visit
distribution, which adds information that no sample in the dataset
carries today. The two are not the same intervention, and task (c) never
depended on task (a).

Two corrections to an earlier reading of this document.

First, task (c) is not a proposal that this document invented. Item 1 of
"Future Plans" in `status.md` already listed "root visit distribution as
policy target" as part of the Gen 6 work, before this document existed.
This document only specifies the record format and the top-8 truncation.

Second, and more important: **task (c) does not address the reported
wandering.** The measured cause of the wandering is the WL head, not the
policy head. `benchmarks/diag_wander.cpp` shows that the WL head reads
the remaining-wall counts and is almost blind to the pawn race, while
"the policy head stays correct and gives the advancing move a prior of
0.73". See the note "MCTS endgame wandering" in `status.md`. Any work on
the policy target improves a component that is already correct in the
failing position.

Task (c) therefore stands on its own merit for general strength, as the
roadmap intended. It must not be sold as a fix for the wandering.

## Purpose

The policy head of the NNUE learns from one label for each recorded
position. The label is the index of the move that self-play played.
This design replaces that label with the visit distribution of the MCab
root, stored as a sparse top-8 list. This design also excludes the
unsearched opening plies from the policy loss.

## Background

The policy head maps 256 accumulator values to 209 outputs. The 209
outputs cover 81 pawn destinations and 128 wall placements. See
`POLICY_OUT` in `src/nnue.hpp`.

`tools/selfplay/selfplay.hpp` records one `uint16_t policyTarget` for
each position. The value is the index of the move that self-play chose.
`training/train_nnue.py` applies cross entropy against that single
index. The target is therefore one-hot. It states that one move is
correct and that the other 208 moves are equally incorrect.

On a searched ply, MCab runs 20000 simulations and fills
`std::vector<float> N` at the root. See `MCABNode` in `src/mcab.hpp`.
That vector holds the visit count of every legal move. `choose()`
returns the move with the most visits and drops the vector. The file
keeps one label out of 20000 simulations of information.

The one-hot target has two costs. First, cross entropy moves
probability mass away from the moves that the search preferred but did
not select. Second, the target carries no information about how close
the alternatives were.

A wall move and a pawn advance are often close in visit count. A
one-hot target teaches full confidence in the move that won that close
count. The policy prior feeds the P term of the PUCT formula in MCab.
Therefore an overconfident prior reduces the exploration of the
alternative move, which produces a more lopsided visit count in the
next generation of data.

This mechanism is present in the pipeline. This design does not claim
that the mechanism is the proven cause of the wall timing behavior that
the project reports. Task (a), described in the section "Sequencing"
below, is the measurement that tests the claim.

## Decisions

The user approved three decisions.

1. The record grows to 64 bytes. It keeps the 8 moves with the most
   visits.
2. A ply without an MCab tree does not contribute to the policy loss.
   The same ply still contributes to the value loss.
3. Training mixes the legacy shards and the new shards. A per source
   weight controls the contribution of the legacy policy target.

## Record format

The current 32 bytes keep their offsets and their meaning. The new
fields come after them. The change is additive.

```c
// tools/selfplay/selfplay.hpp -- TrainingSample, after ownCatTotal/oppCatTotal
uint16_t policyTopIdx[8];   // policy indices 0..208, mirrored, visits descending
uint16_t policyTopProb[8];  // normalized to sum 65535 over the 8 kept entries
static_assert(sizeof(TrainingSample) == 64, "TrainingSample must stay 64 bytes");
```

The design keeps no visit total. The 8 entries use the full 32 new
bytes. Therefore `policyTopProb[0] == 0` marks a position that no
search visited. The node budget is constant for one run, so the visit
total carries little diagnostic value.

The probabilities normalize over the 8 kept entries. The target is
therefore a proper distribution. The mass of the discarded tail spreads
over the 8 entries in proportion to their visits.

The record keeps the existing `policyTarget` field. That field names
the move that self-play played. It also gives one strong invariant. When
`root-select` is `visits`, `policyTopIdx[0]` equals `policyTarget` on
every searched ply.

## Engine changes

`src/mcab.hpp` adds an optional output parameter to `choose()` and to
`chooseMoveMCAB()`. The parameter defaults to `nullptr`. It copies the
move and the visit count of every root edge. The default value keeps
the call site in `tools/arena/arena.cpp` unchanged. The default value
also keeps the `supported == false` fallback path compilable.

`tools/selfplay/selfplay.hpp` sorts the pairs by visits in descending
order. It keeps the first 8 pairs. It maps every kept move through
`mirrorMoveForPerspective` and `moveToPolicyIndex`. That is the same
mirror that `policyTarget` already uses. It then normalizes the
probabilities and writes the record. An unsearched ply writes zeros in
both new arrays.

`tools/arena/arena.cpp` holds a duplicate of `TrainingSample`. The
duplicate must stay identical byte for byte. Both copies get the
`static_assert` above.

## Training changes

`training/read_selfplay.py` adds `SAMPLE_DTYPE_V3` and a third branch
to the size detection. The loader already detects a 27 byte record and
a 32 byte record. A legacy shard upcasts to the new dtype with both new
arrays set to zero.

`training/train_nnue.py` replaces `F.cross_entropy(logits, index)` with
a soft cross entropy. It scatters the sparse top-8 target into a dense
209 vector. It then computes `-(target * log_softmax(logits)).sum(-1)`.

The policy loss uses a per sample weight. The weight is zero when
`policyTopProb[0] == 0`. The value loss ignores that weight. A sample
from an unsearched ply therefore trains the value head and does not
train the policy head.

## Legacy data

A legacy sample has no visit distribution. Its policy target stays
one-hot. The loss keeps a one-hot branch for that case.

`DATA_SOURCES_DEFAULT` gains a `kp` key beside the existing `k` key.
`kp` weights the policy loss of that source. The default value is 1.0.
A lower value fades the legacy policy signal as new shards accumulate.

## Testing

The project requires a correctness test for every new search behavior.
This design adds the tests below.

1. A C++ test builds a position, runs MCab with a small node budget,
   and reads the record that self-play writes. It confirms that
   `policyTopIdx[0]` equals `policyTarget`, that the probabilities sum
   to 65535, that the entries sort in descending order, and that every
   index falls in the range 0 to 208.
2. The same test confirms that an unsearched ply writes zeros in both
   new arrays.
3. A Python test confirms that `read_selfplay.py` detects a 27 byte
   file, a 32 byte file, and a 64 byte file, and that it upcasts the
   two older formats without an error.
4. A build check confirms that `sizeof(TrainingSample)` is 64 in
   `selfplay.hpp` and in `arena.cpp`.

The project requires an arena match for a strength claim. Any claim
about the resulting network needs `tools/arena/run_arena.py`, not a
training curve alone.

## Sequencing

Task (a) runs first, because it measures the problem before the format
change commits to a solution.

Task (a) depends on one assumption. The per game ply index must be
recoverable from the existing shards. `selfplay.hpp` writes one game
for each `fwrite` call, so the samples of one game stay contiguous and
keep their ply order. A ply 0 record has no wall, has 10 walls for each
player, and has a BFS distance of 8 for each player.

A validation script tested that assumption on 2026-08-25. The
reconstruction works. The script segmented four shards and found 9000
games. The `mover` field alternates strictly in 100 percent of the
segments. No segment has a length of 0 plies or 1 ply. No segment
exceeds the safety cut of 300 plies. The mean game length is
approximately 65 plies.

The measured fraction of records below ply 26 is 39.8 percent. That
fraction is stable across every shard the script tested.

The script also corrected one assumption of this design. A shard does
not always hold 3000 games. `selfplay_020.bin` holds 2000 games and
`selfplay_037.bin` holds a similar reduced count, because self-play
wrote them as partial chunks. Therefore the expected game count comes
from the record count of the file, never from a constant.

The order of work is:

1. Validate the ply reconstruction on the existing shards. Complete on
   2026-08-25.
2. Implement task (a). Exclude the opening plies from the policy loss,
   train, and measure the result in an arena match.
3. Implement task (c), the format change described above.

Task (b) is complete. `MC_TEMP_OPENING` in
`tools/selfplay/run_selfplay.py` changed from 1.00 to 0.60 on
2026-08-25.

## Open risks

The truncation to 8 entries loses the tail of the distribution. A
position with more than 8 playable moves of similar value loses real
signal. The project has not measured how often that case occurs.

The format change invalidates no existing shard. However, it does
create a second target format. The training code must carry both
branches until the legacy shards retire.
