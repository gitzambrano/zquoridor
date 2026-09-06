#!/usr/bin/env python3
"""Patch train_nnue.py with a wall-resource monotonicity regularizer.

The constraint is game-theoretically safe: with the same board, pawns and side
to move, an additional unused wall cannot reduce optimal value because the
player may ignore it. Likewise, giving the opponent an additional wall cannot
increase our optimal value.

The regularizer is a zero-margin hinge on VALUE LOGITS, not a hand-authored
wall bonus:

    relu(v(x) - v(x with ownWalls+1))
    relu(v(x with oppWalls+1) - v(x))

It is disabled by default and sampled on a configurable fraction of each
already-shuffled training batch to control cost.
"""
from pathlib import Path

PATH = Path("training/train_nnue.py")
s = PATH.read_text(encoding="utf-8")

replacements = []

replacements.append((
"""W_OUTCOME_DEFAULT = 1.0               # peso da loss da cabeca de resultado (BCE)\nW_POLICY_DEFAULT = 1.0                # peso da loss de policy (CE)\n""",
"""W_OUTCOME_DEFAULT = 1.0               # peso da loss da cabeca de resultado (BCE)\nW_POLICY_DEFAULT = 1.0                # peso da loss de policy (CE)\n# Counterfactual resource-consistency loss. Zero keeps production training\n# bit-for-bit on the old objective; experiments enable it explicitly.\nWALL_MONO_LAMBDA_DEFAULT = 0.0\nWALL_MONO_SAMPLE_FRACTION_DEFAULT = 0.25\n"""))

replacements.append((
"""    \"w_outcome\", \"w_policy\", \"qa\", \"qb\", \"grad_clip_norm\",\n)\n""",
"""    \"w_outcome\", \"w_policy\", \"wall_mono_lambda\",\n    \"wall_mono_sample_fraction\", \"qa\", \"qb\", \"grad_clip_norm\",\n)\n"""))

anchor = '''class WeightClipper:\n'''
helper = '''def _value_logit_only(model: "QuoridorNNUE", x: torch.Tensor) -> torch.Tensor:\n    """Value forward without the policy head, used for counterfactual states."""\n    acc = model.fc1(x)\n    a = screlu(acc)\n    h = clipped_relu(model.value1_wl(a))\n    return model.value2_wl(h).squeeze(-1)\n\n\ndef wall_monotonicity_loss(model: "QuoridorNNUE", x: torch.Tensor,\n                            base_value: torch.Tensor, sample_fraction: float) -> torch.Tensor:\n    """Zero-margin hinge enforcing wall-resource dominance on synthetic twins.\n\n    The batch is already shuffled by the training loop, so taking a prefix is\n    an unbiased cheap subsample without adding another RNG/checkpoint state.\n    Gradients flow through both the observed state and its counterfactual twin.\n    """\n    if sample_fraction <= 0.0 or x.shape[0] == 0:\n        return base_value.new_zeros(())\n    n = max(1, min(int(x.shape[0]), int(round(float(x.shape[0]) * sample_fraction))))\n    xs = x[:n]\n    vb = base_value[:n]\n    wall_base = 290 + 2 * DIST_BUCKETS  # 332\n    own = torch.argmax(xs[:, wall_base:wall_base + WALLS_LEFT_BUCKETS], dim=1)\n    opp_base = wall_base + WALLS_LEFT_BUCKETS\n    opp = torch.argmax(xs[:, opp_base:opp_base + WALLS_LEFT_BUCKETS], dim=1)\n    terms = []\n\n    own_mask = own < (WALLS_LEFT_BUCKETS - 1)\n    if bool(own_mask.any()):\n        xc = xs[own_mask].clone()\n        cur = own[own_mask]\n        rows = torch.arange(xc.shape[0], device=xc.device)\n        xc[rows, wall_base + cur] = 0.0\n        xc[rows, wall_base + cur + 1] = 1.0\n        v_more_own = _value_logit_only(model, xc)\n        terms.append(F.relu(vb[own_mask] - v_more_own).mean())\n\n    opp_mask = opp < (WALLS_LEFT_BUCKETS - 1)\n    if bool(opp_mask.any()):\n        xc = xs[opp_mask].clone()\n        cur = opp[opp_mask]\n        rows = torch.arange(xc.shape[0], device=xc.device)\n        xc[rows, opp_base + cur] = 0.0\n        xc[rows, opp_base + cur + 1] = 1.0\n        v_more_opp = _value_logit_only(model, xc)\n        terms.append(F.relu(v_more_opp - vb[opp_mask]).mean())\n\n    if not terms:\n        return base_value.new_zeros(())\n    return torch.stack(terms).mean()\n\n\n'''
if anchor not in s:
    raise SystemExit("missing WeightClipper anchor")
s = s.replace(anchor, helper + anchor, 1)

replacements.append((
'''    g_loss.add_argument("--w-policy", type=float, default=W_POLICY_DEFAULT)\n    g_loss.add_argument("--wl-gamma", type=float, default=WL_GAMMA_DEFAULT,\n''',
'''    g_loss.add_argument("--w-policy", type=float, default=W_POLICY_DEFAULT)\n    g_loss.add_argument("--wall-mono-lambda", type=float, default=WALL_MONO_LAMBDA_DEFAULT,\n                        help="peso da regularizacao contrafactual: mais muros proprios nao podem "\n                             "reduzir o value logit e mais muros do oponente nao podem aumenta-lo")\n    g_loss.add_argument("--wall-mono-sample-fraction", type=float,\n                        default=WALL_MONO_SAMPLE_FRACTION_DEFAULT,\n                        help="fracao de cada batch usada nos twins contrafactuais (0..1)")\n    g_loss.add_argument("--wl-gamma", type=float, default=WL_GAMMA_DEFAULT,\n'''))

replacements.append((
'''                loss_outcome = weighted_outcome_loss(value_wl, t["wl_target"], t["wl_w"])\n                loss_policy = weighted_policy_loss(policy_logits, t["policy_top_idx"], t["policy_top_prob"], pw)\n                loss = args.w_outcome * loss_outcome + args.w_policy * loss_policy\n\n                opt.zero_grad(set_to_none=True)\n''',
'''                loss_outcome = weighted_outcome_loss(value_wl, t["wl_target"], t["wl_w"])\n                loss_policy = weighted_policy_loss(policy_logits, t["policy_top_idx"], t["policy_top_prob"], pw)\n                if args.wall_mono_lambda > 0.0:\n                    loss_mono = wall_monotonicity_loss(\n                        model, t["x"], value_wl, args.wall_mono_sample_fraction)\n                else:\n                    loss_mono = value_wl.new_zeros(())\n                loss = (args.w_outcome * loss_outcome + args.w_policy * loss_policy\n                        + args.wall_mono_lambda * loss_mono)\n\n                opt.zero_grad(set_to_none=True)\n'''))

# Add argument validation close to the end of parse_args, after data-source JSON.
replacements.append((
'''    if args.data_sources:\n        try:\n            args.data_sources = json.loads(args.data_sources)\n        except json.JSONDecodeError as e:\n            p.error(f"--data-sources: JSON invalido ({e})")\n \n    return args\n''',
'''    if args.data_sources:\n        try:\n            args.data_sources = json.loads(args.data_sources)\n        except json.JSONDecodeError as e:\n            p.error(f"--data-sources: JSON invalido ({e})")\n    if args.wall_mono_lambda < 0.0:\n        p.error("--wall-mono-lambda must be >= 0")\n    if not (0.0 <= args.wall_mono_sample_fraction <= 1.0):\n        p.error("--wall-mono-sample-fraction must be in [0,1]")\n \n    return args\n'''))

for old, new in replacements:
    if old not in s:
        raise SystemExit("expected patch anchor not found:\n" + old[:200])
    s = s.replace(old, new, 1)

PATH.write_text(s, encoding="utf-8")
print("applied wall-resource monotonicity regularizer")
