# race512 phase/topology experiment

Base: `main@ce6eb8449a8303f53107746f9afdf4fcef3b61a5`.

## Why this branch exists

The current evidence does not support making the network wider by itself.

- `base512-search10-ft` is the strongest broadly confirmed candidate so far.
- The original endgame wandering reproducer is handled by the production
  zero-wall endgame fallback, but the documented root cause remains a weak
  value/WL representation in wall-poor races.
- The earlier Gen11 teacher-feature pilots preserved the wandering repro but
  lost substantial Elo, so their small-data/wide-network recipe must not be
  repeated.
- `race` already encodes shortest-path margin, wall-stock margin, and a
  coarse race-by-reserve interaction. New features should add information,
  not duplicate those relations.

## Hypothesis

Add a small amount of information that distinguishes positions with the same
BFS distances and wall reserves:

1. **phase** — coarse total-wall / wall-spent phase buckets.
2. **topology-lite** — cheap local obstruction structure around both pawns and
   goal-facing directions, computed from existing wall bitboards with no new
   BFS.
3. **race-phase interaction** — a small categorical interaction between race
   lead/tie/trail and phase.
4. **training-only race/value auxiliary target** — emphasize wall-poor and
   forced-race states during training; do not add inference-time cost.

Do not add all-shortest-path maps, corridor maps, or another BFS in the first
ablation. Those are follow-ups only if the cheap package earns Elo.

## Ablation order

Use the same `race:512` hidden width and the same search fine-tune recipe.

1. `race512 + phase`
2. `race512 + topology-lite`
3. `race512 + phase + topology-lite`
4. best feature variant + targeted value/race auxiliary training

Each step must preserve Python/C++ feature parity and incremental NNUE parity
before arena testing.

## Wandering gate

A candidate is not considered improved merely because it wins the historical
single reproducer. Measure separately:

- historical zero-wall winning-race reproducer;
- wall-poor races with 0/1/2 walls remaining;
- unnecessary lateral/backward pawn moves while objectively ahead;
- repetition/shuffle rate;
- distance-to-goal progress relative to an exact/deep-search teacher.

Keep oracle-optimal maximum-delay defense out of the failure count.

## Strength gate

Use the frozen confirmation protocol:

- 200 ms/move;
- workers=1;
- paired colors;
- `tools/external/openings_confirmation_v1.jsonl`;
- same seed and wall-clock protocol for candidate and reference;
- broad confirmation against production main, pinned Titanium, and
  Claustrophobia.

No promotion from training loss or a small screening alone.
