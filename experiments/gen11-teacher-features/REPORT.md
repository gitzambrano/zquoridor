# Gen11 teacher and feature experiment

Baseline source: `ec29daa081025439d6f5596aa10250057a3e0af1`.

| Variant | Elo vs baseline | Arena NPS candidate/base | MCAB expansion ratio | Wander progress candidate/base |
|---|---:|---:|---:|---:|
| wide320_teacher | failed/no result | - | - | - |
| race320_teacher | failed/no result | - | - | - |
| race320_no_value_teacher | failed/no result | - | - | - |

The race feature package adds exact race differential, wall-stock differential, and a 5x5 interaction bucket. It performs no additional BFS.
The value-teacher variants use `k=0.55`, so the WL target blends 55% final game outcome with 45% searched MCAB root value. All variants use the existing top-8 MCAB visit distribution as the policy teacher.
