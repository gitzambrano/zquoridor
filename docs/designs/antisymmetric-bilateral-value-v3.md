# Antisymmetric bilateral value v3

## Goal

The v2 bilateral head exposes both player accumulators to the value network, but it does not enforce the zero-sum identity

`V(own, opp) = -V(opp, own)`.

A Quoridor value logit is always interpreted from the side-to-move perspective, so violating this identity wastes capacity and can create inconsistent leaf values under a player swap.

## Architecture

Keep the shared 354 -> 256 accumulator and the existing 256 -> 209 policy head unchanged.

Replace the v2 value head

`concat(own, opp) [512] -> 64 -> 1`

with a shared scorer used in both orders:

`F(own, opp) = w2 * clipped_relu(W * concat(own, opp) + b)`

`V(own, opp) = F(own, opp) - F(opp, own)`

Use 32 hidden units for `F`.

This guarantees exactly, including after quantization,

`V(opp, own) = -V(own, opp)`

provided the final subtraction is performed in the common integer scale and the output bias is omitted/cancelled.

## Compute

v2 first-head MACs: `512 * 64 = 32768`.

v3 first-head MACs: two ordered evaluations of `512 * 32 = 16384`, total `32768`.

Therefore v3 keeps approximately the same multiply count as v2 while halving value-head parameters and imposing the correct symmetry.

## Training

PyTorch computes both ordered scores with shared `value1_wl` and `value2_wl` weights and returns their difference. Policy remains unilateral. No antisymmetry penalty is necessary because the constraint is architectural.

The output scale should be checked because subtracting two scorer outputs can increase logit variance. Start without a 0.5 factor; inspect calibration and only rescale if the trained distribution saturates.

## Gates

1. Exact swap test on random positions: `logit(a,b) + logit(b,a) == 0` within float tolerance and exactly in quantized accumulator units where practical.
2. Incremental/cold accumulator parity.
3. WL monotonic calibration probe.
4. Parametric wall-poor wandering suite.
5. NPS comparison against v2.
6. Elo against the same Gen8 + cPuct 1.20 baseline.

Do not replace v2 until v3 beats it at equal search settings.