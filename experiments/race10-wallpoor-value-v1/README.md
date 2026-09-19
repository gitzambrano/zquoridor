# wall-poor value supervision

Purpose: fix the documented WL-head blind spot after the mover spends its walls.

The current mixed fine-tune is teacher distillation only. This branch joins the
same canonical states back to the historical replay and blends the mover-relative
real game outcome into the value target.

Default intervention:

- all states: 10% real outcome / 90% existing teacher value;
- mover with 0 or 1 wall: 50% real outcome / 50% existing teacher value;
- policy targets are unchanged;
- sample weights are unchanged by default (`wallpoor_weight_boost=1`).

Keeping the weight boost at 1 isolates target quality from resampling. A second
experiment can raise only `wallpoor_weight_boost` after this target ablation.
