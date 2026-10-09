# Production Central Fine-Tune and 21.121M Master Dataset

## 1. Overview

This directory preserves the 21,121,013-position master training dataset used to train the Zquoridor 3.00 baseline and the Stage A models that formed the Zquoridor 3.01 tri-model champion.
The dataset resides in memory-mapped NumPy array format (`fields/*.npy`) partitioned into three specialized weighting regimes.

## 2. Dataset Composition and Mix Ratio

The master dataset combines two distinct training cohorts:
- **Search Rollouts (75%)**: 15,840,760 positions derived from cloud and local Monte Carlo search rollouts with clean, unpolluted search value targets.
- **Tactical Master Corpus (25%)**: 5,280,253 positions focused on opening variations, Center Rush lines, and deep defensive positions against adversarial reference engines.

## 3. Data Partitions and Policy Surprise Weighting

1. `mixed_dataset/`:
   The baseline unweighted mixture.
   Sample weights are uniform or based solely on source cohort balance.
2. `mixed_dataset_surprise/`:
   Ponderated with Policy Surprise Weighting.
   Parameters: $\alpha = 0.5$, $S_{\max} = 4.0$.
   The script computes Kullback-Leibler divergence between the stored search distribution and baseline network priors.
   Positions where search identified critical unexpected tactical moves receive up to 4.0 times higher gradient weight.
3. `mixed_dataset_surprise_v2/`:
   Refined surprise weighting with doubled auxiliary policy regularization ($\beta = 0.25$).
   This dataset trained candidate `AS2_surprise_v2_t20_b25` (epoch 14), which achieved the lowest validation loss in project history (1.72065).

Note: The large 17.6 GB policy distribution array (`fields/policy.npy`) is shared across all three partitions using filesystem hardlinks to avoid redundant disk storage.

## 4. Field Specification

| Field Name | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `own_pawn` | `uint8` | `(21121013,)` | Active player cell index (0 to 80) |
| `opp_pawn` | `uint8` | `(21121013,)` | Opponent player cell index (0 to 80) |
| `walls_h` | `uint64` | `(21121013,)` | 64-bit mask of placed horizontal walls |
| `walls_v` | `uint64` | `(21121013,)` | 64-bit mask of placed vertical walls |
| `own_dist` | `uint8` | `(21121013,)` | Shortest path BFS distance for active player (0 to 20) |
| `opp_dist` | `uint8` | `(21121013,)` | Shortest path BFS distance for opponent (0 to 20) |
| `walls_left_own` | `uint8` | `(21121013,)` | Remaining walls in stock for active player (0 to 10) |
| `walls_left_opp` | `uint8` | `(21121013,)` | Remaining walls in stock for opponent (0 to 10) |
| `policy` | `float32` | `(21121013, 209)` | Monte Carlo search visit probabilities across 209 legal moves |
| `value` | `float32` | `(21121013,)` | Search game outcome or discounted value evaluation |
| `weight` | `float32` | `(21121013,)` | Sample loss multiplier (surprise weight) |
| `is_val` | `bool` | `(21121013,)` | Validation partition flag (20% holdout) |

## 5. Reuse in Upcoming Training Iterations

This 21.121M master corpus serves as the historical anchor for the next generation of Zquoridor models.
When blending in fresh rollouts from the Zquoridor 3.01 cloud campaign, this corpus will be combined with the new shards to prevent catastrophic forgetting and preserve tactical sharpness.
