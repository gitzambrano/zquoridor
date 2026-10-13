# Training networks and experiment records

`registry.json` lists exported experimental networks and their SHA-256 values.
Float and int8 weights reside in `data/nnue/experiments/`. These artifacts
support local training and native evaluation. They do not replace production.
The production network remains version 4.01 under `data/nnue/`.

Widths 512 and 768 have completed matched training. Width 1024 is an available
architecture option without a completed strength test. The apex replay control
and mixed candidate have trained weights; their paired strength screens remain
pending. See [the project measurements](../../docs/plan.md) for interpretation.
The two older apex value-only experiments retain their exported weights as
reproducible comparison artifacts. Their checkpoints did not initialize 4.01.

Training reports identify source fingerprints. Published report paths are
relative to the repository. Existing optimizer checkpoints and datasets are
not included. Start new local runs from the exported float weights.
Prepared data and raw trajectory metadata must exist on the destination machine.

## Independent options

```sh
python training/run_local_training.py --hidden 768 --target-mode temporal --dry-run
python training/run_local_training.py --hidden 512 --target-mode temporal --apex-value-fraction 0.1 --epochs 10 --lr 0.000005 --min-lr 0.000001 --init-from data/nnue/experiments/width_512_100ep/student.bin
python training/run_local_training.py --hidden 1024 --target-mode current --apex-value-fraction 0
```

`--hidden` selects network capacity. `--apex-value-fraction` selects an
additional value-teaching source. `--target-mode` selects current or temporal
search values. These options do not select or imply each other.

Temporal targets default to offsets 0, 4, 8, and 16 plies with coefficients
0.5, 0.25, 0.125, and 0.0625. Available search evidence receives normalized
weight. The game outcome receives 0.1 of the final target. `--future-horizons`,
`--future-weights`, and `--result-weight` override these settings.
Future values use the original mover perspective. Current-position policy
targets remain separate. Apex-only samples have no main or auxiliary policy
loss and retain their provided values without fabricated future trajectories.

```sh
python training/run_local_training.py --target-mode temporal --future-horizons "[0,2,6]" --future-weights "[0.5,0.25,0.125]" --result-weight 0.1 --dry-run
```

Changing target settings requires raw games or paired binary shards and metadata.
The runner prepares a separate cache when existing targets do not match.
Pass `--data` for a prepared dataset and `--downloads` for a local artifact
ledger. Use `--out-dir` for a new run and `--device cpu` or `--device cuda`
to select the execution device. `training/requirements-local.txt` lists the
Python dependencies. Native export verification requires a C++17 compiler.
