# Gen11 teacher and feature experiment

Baseline source: `ec29daa081025439d6f5596aa10250057a3e0af1`.

| Variant | Elo vs baseline | Arena NPS candidate/base | MCAB expansion ratio | Wander progress candidate/base |
|---|---:|---:|---:|---:|
| wide320_teacher | -122.0 ±55.6 | 40,490/41,610 | 0.871 | 5/5 vs 5/5 |
| race320_teacher | -147.2 ±56.5 | 36,535/43,650 | 0.806 | 5/5 vs 5/5 |
| race320_no_value_teacher | -240.8 ±63.2 | 39,038/42,710 | 0.829 | 5/5 vs 5/5 |

Race320 adds exact race differential, wall-stock differential, and a 5x5 race-by-reserve interaction bucket. These features reuse cached distance and wall counts and add no BFS.
Teacher variants use k=0.55, which blends 55% final outcome with 45% searched MCAB root value. All three variants retain the top-8 MCAB visit distribution as the policy teacher.
