# External Tools and Opening Catalogs

This directory contains external bot wrappers, referee harnesses, and canonical opening catalogs for engine benchmarks and self-play generation.

## Opening Books Catalog

The repository maintains four canonical evaluation books and two self-play seed banks.

| File | Rows | Ply Depth | Category | Intended Use |
| :--- | :--- | :--- | :--- | :--- |
| `openings_normal_screen_100.jsonl` | 100 | 6 | Normal / Wall Defense | Fast directional screening and 50-pair (100 games) candidate evaluation batteries. |
| `openings_normal_confirm_400.jsonl` | 400 | 6 | Normal / Wall Defense | Broad confirmation batteries for promotion finalists (up to 400 pairs / 800 games). |
| `openings_center_rush_sound_5k.jsonl` | 5,000 | 8 to 12 | Center Rush / Sprint | Modern tactical pawn rush and wall clash benchmark, verified for search soundness. |
| `openings_titanium_mined_40.jsonl` | 40 | 6 | Titanium Empirical | Historical 40-opening comparison set mined from Titanium games. Pinned benchmark set. |
| `openings_irregular_bank.jsonl` | 2,500 | 6 to 9 | Self-Play Seed Bank | Irregular and perimeter opening seeds for self-play data diversity. |
| `openings_targeted_weakness_bank.jsonl` | 2,500 | 8 to 10 | Self-Play Seed Bank | Tactical clash and weakness seeds for targeted self-play rollouts. |

## Disjoint Sets and Benchmarking Protocol

1. `openings_normal_screen_100.jsonl` and `openings_normal_confirm_400.jsonl` share zero common positions. They form a strictly disjoint 500-position catalog of standard six-ply openings.
2. `openings_titanium_mined_40.jsonl` is a frozen external benchmark. Training pipelines explicitly block all positions and prefixes in this set to prevent data contamination.
3. Arena runners select openings deterministically by seed. When a test requires $N$ pairs, the runner shuffles the catalog with the specified seed and slices the first $N$ rows.

## Retired Legacy Files

The following files were removed to eliminate duplication and confusion:

- `openings_600g_300pairs.jsonl`: Removed. This file was an exact duplicate of lines 1 to 300 of `openings_normal_confirm_400.jsonl`.
- `openings_claustro_followup_200pairs.jsonl`: Removed. This file contained 200 openings already present in `openings_normal_confirm_400.jsonl`.
- `openings_center_rush_v1.jsonl`: Removed. This file contained the first 33 lines of the older 50-pair book.
- `openings_center_rush_50pairs.jsonl`: Removed. This early 50-pair opening set was completely superseded by `openings_center_rush_sound_5k.jsonl`.

## Data Schemas

Opening lines use one of two standardized schemas:

### Simple Opening Schema
```json
{"moves": ["e8h", "g4h", "d2h", "c4h", "a7v", "e1h"]}
```

### Position V1 Schema
```json
{
  "schema": "zquoridor.position.v1",
  "id": "cr_sound_00000",
  "category": "classic_rush_sound",
  "base_plies": 6,
  "branch_plies": 6,
  "total_plies": 12,
  "eval_prob": 0.5172,
  "moves": ["e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e5", "d4h", "d5h", "c5v", "d3v"]
}
```
