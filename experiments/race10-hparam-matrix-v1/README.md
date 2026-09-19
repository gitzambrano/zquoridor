# race10 training hyperparameter matrix

Base candidate: `race512-search10-ft-s20260917`.

This branch changes one training dimension at a time while keeping the same
architecture, initialization, data, optimizer family, LR schedule, QAT, and
seed.

Variants:

- `value05`: value loss weight 0.5.
- `value20`: value loss weight 2.0.
- `value40`: value loss weight 4.0.
- `trunk03`: trunk LR scale 0.3.
- `trunk10`: trunk LR scale 1.0.

The production recipe is value weight 1.0 and trunk LR scale 0.1.

Run `python training/run_race10_hparam_matrix.py --dry-run` to inspect exact
commands. The heavy training dataset is intentionally not committed to Git.
