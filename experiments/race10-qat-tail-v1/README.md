# race10 QAT-tail ablation

This branch tests whether the student benefits from learning in float first and
adapting to the deployed integer grid only in the final part of the fine-tune.

- `qat-tail20`: epochs 1–20 float, epochs 21–40 QAT.
- `qat-tail10`: epochs 1–30 float, epochs 31–40 QAT.
- Production reference: QAT from epoch 1.

At the QAT transition the model is clipped once to the representable range.
Best-checkpoint selection is reset at that transition so a float-only
checkpoint cannot be exported as the winner of a QAT experiment.
