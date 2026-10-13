# Data Directory Overview

This directory stores engine datasets, training targets, network checkpoints, and evaluation artifacts.

## Directory Layout

* `selfplay/`: Raw self-play match corpora in canonical V3 binary format (`.bin`).
* `teaching/`: Curated training archives (`.npz` files) and assembled curriculum tiers.
* `nnue/`: Production weights and experimental network files (`.bin`).
* `checkpoints/`: Intermediate training checkpoints (`.pt` files).
* `runs/`: Logs and outputs from exploratory training arms.
* `arena/`: Playout records from bot evaluation tournaments.
* `contact-selfplay-flat/`: Contact feature generation records.
* `plots/`: Metric visualizations and training curve plots.
* `logs/`: Operational logs from local generation pipelines.

## Retention and Safety

* Datasets and training files in this directory are preserved assets. Do not delete them.
* Raw binary and NPZ archives remain outside version control as defined in `.gitignore`.
