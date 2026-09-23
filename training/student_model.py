"""Compact NNUE students with explicit architecture and portable exports."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .quantize_nnue import quantize, write_quantized
except ImportError:  # Direct execution from the training directory.
    from quantize_nnue import quantize, write_quantized

FEATURES = {
    "base": 354,
    "race": 456,
    "multipath": 480,
    "margin_regime": 588,
    "phase": 480,
    "margin_phase": 612,
    "multipath_phase": 504,
    "multipath_phase_contact": 858,
    "multipath_phase_bucketed": 504,
    "multipath_phase_deep": 504,
    "multipath_phase_contact_bucketed": 858,
}
ARCH_CONFIGS = {
    "base": {"buckets": 1, "depth": 1, "base_feature": "base"},
    "race": {"buckets": 1, "depth": 1, "base_feature": "race"},
    "multipath": {"buckets": 1, "depth": 1, "base_feature": "multipath"},
    "margin_regime": {"buckets": 1, "depth": 1, "base_feature": "margin_regime"},
    "phase": {"buckets": 1, "depth": 1, "base_feature": "phase"},
    "margin_phase": {"buckets": 1, "depth": 1, "base_feature": "margin_phase"},
    "multipath_phase": {"buckets": 1, "depth": 1, "base_feature": "multipath_phase"},
    "multipath_phase_contact": {"buckets": 1, "depth": 1, "base_feature": "multipath_phase_contact"},
    "multipath_phase_bucketed": {"buckets": 6, "depth": 2, "base_feature": "multipath_phase"},
    "multipath_phase_deep": {"buckets": 1, "depth": 2, "base_feature": "multipath_phase"},
    "multipath_phase_contact_bucketed": {"buckets": 6, "depth": 2, "base_feature": "multipath_phase_contact"},
}
LAYOUT = ("w1", "b1", "wv1_wl", "bv1_wl", "wv2_wl", "bv2_wl", "wp", "bp")


# Value-head bucket for 0 to 20 remaining walls. The boundaries match the
# phase features and getPhaseBucket in nnue.hpp.
_PHASE_BUCKET = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4] + [5] * 6, dtype=np.int64)


def get_phase_bucket(total_w, num_buckets=6):
    """Return the value-head bucket for the total remaining walls."""
    if num_buckets not in (1, 6):
        raise ValueError("value heads support 1 or 6 buckets")
    if isinstance(total_w, torch.Tensor):
        if num_buckets == 1:
            return torch.zeros_like(total_w, dtype=torch.int64)
        return torch.as_tensor(_PHASE_BUCKET, device=total_w.device)[total_w.long().clamp(0, 20)]
    total_w = np.asarray(total_w, dtype=np.int64)
    if num_buckets == 1:
        return np.zeros_like(total_w)
    return _PHASE_BUCKET[np.clip(total_w, 0, 20)]


def _make_edge_tables():
    orth = np.full((81, 4), -1, dtype=np.int8)
    edge_h = np.zeros((81, 4), dtype=np.uint64)
    edge_v = np.zeros((81, 4), dtype=np.uint64)
    for cell in range(81):
        r, col = cell // 9, cell % 9
        if r > 0: orth[cell, 0] = cell - 9
        if r + 1 < 9: orth[cell, 1] = cell + 9
        if col > 0: orth[cell, 2] = cell - 1
        if col + 1 < 9: orth[cell, 3] = cell + 1

        if r > 0:
            wr = r - 1
            if col > 0: edge_h[cell, 0] |= np.uint64(1 << (wr * 8 + col - 1))
            if col < 8: edge_h[cell, 0] |= np.uint64(1 << (wr * 8 + col))
        if r < 8:
            wr = r
            if col > 0: edge_h[cell, 1] |= np.uint64(1 << (wr * 8 + col - 1))
            if col < 8: edge_h[cell, 1] |= np.uint64(1 << (wr * 8 + col))
        if col > 0:
            wc = col - 1
            if r > 0: edge_v[cell, 2] |= np.uint64(1 << ((r - 1) * 8 + wc))
            if r < 8: edge_v[cell, 2] |= np.uint64(1 << (r * 8 + wc))
        if col < 8:
            wc = col
            if r > 0: edge_v[cell, 3] |= np.uint64(1 << ((r - 1) * 8 + wc))
            if r < 8: edge_v[cell, 3] |= np.uint64(1 << (r * 8 + wc))
    return orth, edge_h, edge_v


_ORTH_NEIGHBORS, _EDGE_H_MASKS, _EDGE_V_MASKS = _make_edge_tables()


def encode_features(data, indices, architecture="base"):
    if architecture in ARCH_CONFIGS:
        architecture = ARCH_CONFIGS[architecture]["base_feature"]
    from train_teacher_policy import dense_features
    x = dense_features(data, indices)
    if architecture == "base":
        return x
    if architecture not in ("race", "multipath", "margin_regime", "phase", "margin_phase", "multipath_phase", "multipath_phase_contact"):
        raise ValueError(f"unknown architecture: {architecture}")
    n = len(indices)
    extra = np.zeros((n, 102), dtype=np.float32)
    own = np.clip(data["own_dist"][indices].astype(np.int64), 0, 20)
    opp = np.clip(data["opp_dist"][indices].astype(np.int64), 0, 20)
    ow = np.clip(data["walls_left_own"][indices].astype(np.int64), 0, 10)
    pw = np.clip(data["walls_left_opp"][indices].astype(np.int64), 0, 10)
    rows = np.arange(n)
    extra[rows, np.clip(own - opp, -16, 16) + 16] = 1
    extra[rows, 33 + ow - pw + 10] = 1
    race = np.sign(own - opp) + 1
    extra[rows, 54 + race * 16 + np.minimum(ow, 3) * 4 + np.minimum(pw, 3)] = 1
    x = np.concatenate((x, extra), axis=1)
    if architecture == "race":
        return x

    if architecture in ("margin_regime", "margin_phase"):
        mr = np.zeros((n, 132), dtype=np.float32)
        delta = np.clip(own - opp, -16, 16) + 16
        regime = np.zeros(n, dtype=np.int64)
        regime[(ow == 0) & (pw > 0)] = 1
        regime[(ow > 0) & (pw == 0)] = 2
        regime[(ow == 0) & (pw == 0)] = 3
        mr[rows, regime * 33 + delta] = 1.0
        x = np.concatenate((x, mr), axis=1)
        if architecture == "margin_regime":
            return x

    if architecture in ("phase", "margin_phase", "multipath_phase", "multipath_phase_contact"):
        ph = np.zeros((n, 24), dtype=np.float32)
        total_w = ow + pw
        phase_bucket = np.zeros(n, dtype=np.int64)
        phase_bucket[(total_w >= 1) & (total_w <= 2)] = 1
        phase_bucket[(total_w >= 3) & (total_w <= 5)] = 2
        phase_bucket[(total_w >= 6) & (total_w <= 9)] = 3
        phase_bucket[(total_w >= 10) & (total_w <= 14)] = 4
        phase_bucket[(total_w >= 15)] = 5
        ph[rows, phase_bucket] = 1.0
        ph[rows, 6 + race * 6 + phase_bucket] = 1.0
        x = np.concatenate((x, ph), axis=1)
        if architecture not in ("multipath_phase", "multipath_phase_contact"):
            return x

    # multipath: 24 additional cheap features (zero extra BFS)
    mp = np.zeros((n, 24), dtype=np.float32)
    own_pawn = data["own_pawn"][indices].astype(np.int64)
    opp_pawn = data["opp_pawn"][indices].astype(np.int64)
    walls_h = data["walls_h"][indices].astype(np.uint64)
    walls_v = data["walls_v"][indices].astype(np.uint64)

    # Directional exits: mover perspective (Forward=1/South, Backward=0/North, Left=2/West, Right=3/East)
    own_fwd = (_ORTH_NEIGHBORS[own_pawn, 1] >= 0) & (((walls_h & _EDGE_H_MASKS[own_pawn, 1]) | (walls_v & _EDGE_V_MASKS[own_pawn, 1])) == 0)
    own_bwd = (_ORTH_NEIGHBORS[own_pawn, 0] >= 0) & (((walls_h & _EDGE_H_MASKS[own_pawn, 0]) | (walls_v & _EDGE_V_MASKS[own_pawn, 0])) == 0)
    own_lft = (_ORTH_NEIGHBORS[own_pawn, 2] >= 0) & (((walls_h & _EDGE_H_MASKS[own_pawn, 2]) | (walls_v & _EDGE_V_MASKS[own_pawn, 2])) == 0)
    own_rgt = (_ORTH_NEIGHBORS[own_pawn, 3] >= 0) & (((walls_h & _EDGE_H_MASKS[own_pawn, 3]) | (walls_v & _EDGE_V_MASKS[own_pawn, 3])) == 0)

    # Opponent perspective (Forward=0/North, Backward=1/South, Left=2/West, Right=3/East)
    opp_fwd = (_ORTH_NEIGHBORS[opp_pawn, 0] >= 0) & (((walls_h & _EDGE_H_MASKS[opp_pawn, 0]) | (walls_v & _EDGE_V_MASKS[opp_pawn, 0])) == 0)
    opp_bwd = (_ORTH_NEIGHBORS[opp_pawn, 1] >= 0) & (((walls_h & _EDGE_H_MASKS[opp_pawn, 1]) | (walls_v & _EDGE_V_MASKS[opp_pawn, 1])) == 0)
    opp_lft = (_ORTH_NEIGHBORS[opp_pawn, 2] >= 0) & (((walls_h & _EDGE_H_MASKS[opp_pawn, 2]) | (walls_v & _EDGE_V_MASKS[opp_pawn, 2])) == 0)
    opp_rgt = (_ORTH_NEIGHBORS[opp_pawn, 3] >= 0) & (((walls_h & _EDGE_H_MASKS[opp_pawn, 3]) | (walls_v & _EDGE_V_MASKS[opp_pawn, 3])) == 0)

    mp[rows, 0] = own_fwd.astype(np.float32)
    mp[rows, 1] = own_bwd.astype(np.float32)
    mp[rows, 2] = own_lft.astype(np.float32)
    mp[rows, 3] = own_rgt.astype(np.float32)

    mp[rows, 4] = opp_fwd.astype(np.float32)
    mp[rows, 5] = opp_bwd.astype(np.float32)
    mp[rows, 6] = opp_lft.astype(np.float32)
    mp[rows, 7] = opp_rgt.astype(np.float32)

    # Exit count (branching factor: 1=bottleneck, 2=corridor, 3=T-split, 4=open)
    own_exits = np.clip(own_fwd.astype(np.int64) + own_bwd.astype(np.int64) + own_lft.astype(np.int64) + own_rgt.astype(np.int64), 1, 4)
    opp_exits = np.clip(opp_fwd.astype(np.int64) + opp_bwd.astype(np.int64) + opp_lft.astype(np.int64) + opp_rgt.astype(np.int64), 1, 4)
    mp[rows, 8 + own_exits - 1] = 1.0
    mp[rows, 12 + opp_exits - 1] = 1.0

    # Pawn contact and jump geometry
    own_r, own_c = own_pawn // 9, own_pawn % 9
    opp_r, opp_c = opp_pawn // 9, opp_pawn % 9
    dr = opp_r - own_r
    dc = opp_c - own_c
    abs_dc = np.abs(dc)
    manhattan = np.abs(dr) + abs_dc

    head_on = (dr == 1) & (dc == 0)
    lateral = (dr == 0) & (abs_dc == 1)
    imminent = (manhattan == 2) & ~head_on
    prox3 = (manhattan == 3)
    far = (manhattan >= 4) & ~lateral

    mp[rows, 16] = head_on.astype(np.float32)
    mp[rows, 17] = lateral.astype(np.float32)
    mp[rows, 18] = imminent.astype(np.float32)
    mp[rows, 19] = prox3.astype(np.float32)
    mp[rows, 20] = far.astype(np.float32)

    mp[rows, 21] = (dc == 0).astype(np.float32)
    mp[rows, 22] = (abs_dc == 1).astype(np.float32)
    mp[rows, 23] = (abs_dc >= 2).astype(np.float32)

    x = np.concatenate((x, mp), axis=1)
    if architecture != "multipath_phase_contact":
        return x

    contact = np.zeros((n, 354), dtype=np.float32)
    contact[rows, (dr + 8) * 17 + dc + 8] = 1.0

    def edge_mask(cell):
        mask = np.zeros(n, dtype=np.int64)
        for direction in range(4):
            is_open = (_ORTH_NEIGHBORS[cell, direction] >= 0) & (
                ((walls_h & _EDGE_H_MASKS[cell, direction]) |
                 (walls_v & _EDGE_V_MASKS[cell, direction])) == 0)
            mask |= (~is_open).astype(np.int64) << direction
        return mask

    own_mask = edge_mask(own_pawn)
    opp_mask = edge_mask(opp_pawn)
    contact[rows, 289 + own_mask] = 1.0
    contact[rows, 305 + opp_mask] = 1.0

    adjacent_dir = np.full(n, -1, dtype=np.int64)
    adjacent_dir[(dr == -1) & (dc == 0)] = 0
    adjacent_dir[(dr == 1) & (dc == 0)] = 1
    adjacent_dir[(dr == 0) & (dc == -1)] = 2
    adjacent_dir[(dr == 0) & (dc == 1)] = 3
    option = np.zeros(n, dtype=np.int64)
    for direction in range(4):
        selected = adjacent_dir == direction
        if not selected.any():
            continue
        straight = (_ORTH_NEIGHBORS[opp_pawn, direction] >= 0) & (
            ((walls_h & _EDGE_H_MASKS[opp_pawn, direction]) |
             (walls_v & _EDGE_V_MASKS[opp_pawn, direction])) == 0)
        perpendicular = (2, 3) if direction < 2 else (0, 1)
        left = (_ORTH_NEIGHBORS[opp_pawn, perpendicular[0]] >= 0) & (
            ((walls_h & _EDGE_H_MASKS[opp_pawn, perpendicular[0]]) |
             (walls_v & _EDGE_V_MASKS[opp_pawn, perpendicular[0]])) == 0)
        right = (_ORTH_NEIGHBORS[opp_pawn, perpendicular[1]] >= 0) & (
            ((walls_h & _EDGE_H_MASKS[opp_pawn, perpendicular[1]]) |
             (walls_v & _EDGE_V_MASKS[opp_pawn, perpendicular[1]])) == 0)
        mask = straight.astype(np.int64)
        mask |= ((~straight) & left).astype(np.int64) << 1
        mask |= ((~straight) & right).astype(np.int64) << 2
        option[selected] = 1 + direction * 8 + mask[selected]
    contact[rows, 321 + option] = 1.0
    return np.concatenate((x, contact), axis=1)


def _round_ste(x, scale):
    rounded = torch.round(x * scale) / scale
    return x + (rounded - x).detach()


class Student(nn.Module):
    def __init__(self, architecture="base", hidden=256, qat=False,
                 value_buckets=None, value_depth=None):
        super().__init__()
        if architecture not in FEATURES or hidden not in (128, 256, 384, 512):
            raise ValueError(f"architecture must be one of {list(FEATURES.keys())}; hidden must be 128/256/384/512")
        cfg = ARCH_CONFIGS.get(architecture, {"buckets": 1, "depth": 1})
        self.architecture = architecture
        self.hidden = hidden
        self.qat = qat
        self.value_buckets = int(value_buckets if value_buckets is not None else cfg["buckets"])
        self.value_depth = int(value_depth if value_depth is not None else cfg["depth"])

        self.fc1 = nn.Linear(FEATURES[architecture], hidden)
        self.policy = nn.Linear(hidden, 209)

        self.value1_heads = nn.ModuleList([nn.Linear(hidden, 32) for _ in range(self.value_buckets)])
        if self.value_depth == 2:
            self.value2_heads = nn.ModuleList([nn.Linear(32, 32) for _ in range(self.value_buckets)])
            self.value3_heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(self.value_buckets)])
        else:
            self.value2_heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(self.value_buckets)])
            self.value3_heads = None

        # Aliases for backwards compatibility with single-head code
        self.value1_wl = self.value1_heads[0]
        self.value2_wl = self.value2_heads[0]

    def _extract_buckets(self, x):
        # In dense_features, walls_left are at 332:343 (own) and 343:354 (opp)
        own_w = x[:, 332:343].argmax(dim=-1)
        opp_w = x[:, 343:354].argmax(dim=-1)
        return get_phase_bucket(own_w + opp_w, self.value_buckets)

    def forward(self, x, buckets=None):
        if self.value_buckets > 1 and buckets is None:
            buckets = self._extract_buckets(x)

        if self.qat:
            # Match the deployed integer scales, including the SCReLU truncation.
            # Accumulate exact integer-valued floats before division. Summing
            # dequantized rows can cross a truncation boundary from roundoff alone.
            a_int = F.linear(x, _round_ste(self.fc1.weight * 255, 1),
                             _round_ste(self.fc1.bias * 255, 1))
            a_int = a_int.clamp(0, 255).square() / 255
            a_int = a_int + (torch.floor(a_int) - a_int).detach()
            a = a_int / 255
            p = F.linear(a, _round_ste(self.policy.weight, 64), _round_ste(self.policy.bias, 255 * 64))
        else:
            a = self.fc1(x).clamp(0, 1).square()
            a_int = None
            p = self.policy(a)

        if self.value_buckets == 1:
            return self._value_head(0, a, a_int), p
        v = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for b in range(self.value_buckets):
            mask = buckets == b
            if mask.any():
                v[mask] = self._value_head(b, a[mask], None if a_int is None else a_int[mask])
        return v, p

    def _value_head(self, b, a, a_int):
        """Return the value logit of head ``b``. ``a_int`` is set only for QAT."""
        l1, l2 = self.value1_heads[b], self.value2_heads[b]
        if a_int is None:
            h1 = l1(a).clamp(0, 1)
            if self.value_depth == 1:
                return l2(h1).squeeze(-1)
            return self.value3_heads[b](l2(h1).clamp(0, 1)).squeeze(-1)
        if self.value_depth == 1:
            h1 = F.linear(a, _round_ste(l1.weight, 64), _round_ste(l1.bias, 255 * 64)).clamp(0, 1)
            return F.linear(h1, _round_ste(l2.weight, 64),
                            _round_ste(l2.bias, 255 * 64 * 64)).squeeze(-1)
        # Integer-valued floats follow forwardValueWLQuant: layer 1 at scale
        # QA*QB, floor division by QB to scale QA, and layer 2 at scale QA*QB.
        # A floor of a dequantized float can cross a boundary from roundoff.
        l3 = self.value3_heads[b]
        h1 = F.linear(a_int, _round_ste(l1.weight * 64, 1),
                      _round_ste(l1.bias * 255 * 64, 1)).clamp(0, 255 * 64) / 64
        h1 = h1 + (torch.floor(h1) - h1).detach()
        h2 = F.linear(h1, _round_ste(l2.weight * 64, 1),
                      _round_ste(l2.bias * 255 * 64, 1)).clamp(0, 255 * 64)
        out = F.linear(h2, _round_ste(l3.weight * 64, 1), _round_ste(l3.bias * 255 * 64 * 64, 1))
        return (out / (255 * 64 * 64)).squeeze(-1)

    @torch.no_grad()
    def warm_start(self, old):
        if self.hidden < old.hidden or self.fc1.in_features < old.fc1.in_features:
            raise ValueError("warm start cannot shrink a network; use output distillation from scratch")
        for param in self.parameters():
            param.zero_()
        if self.architecture == "multipath_phase" and old.architecture == "multipath":
            self.fc1.weight[:old.hidden, :456].copy_(old.fc1.weight[:, :456])
            self.fc1.weight[:old.hidden, 480:504].copy_(old.fc1.weight[:, 456:480])
        elif self.architecture in ("multipath_phase_contact", "multipath_phase_contact_bucketed") and old.architecture == "multipath":
            self.fc1.weight[:old.hidden, :456].copy_(old.fc1.weight[:, :456])
            self.fc1.weight[:old.hidden, 480:504].copy_(old.fc1.weight[:, 456:480])
        else:
            self.fc1.weight[:old.hidden, :old.fc1.in_features].copy_(old.fc1.weight)
        self.fc1.bias[:old.hidden].copy_(old.fc1.bias)
        self.policy.weight[:, :old.hidden].copy_(old.policy.weight)
        self.policy.bias.copy_(old.policy.bias)

        old_buckets = getattr(old, "value_buckets", 1)
        old_depth = getattr(old, "value_depth", 1)

        for b in range(self.value_buckets):
            src_b = b if b < old_buckets else 0
            if hasattr(old, "value1_heads"):
                self.value1_heads[b].weight[:, :old.hidden].copy_(old.value1_heads[src_b].weight)
                self.value1_heads[b].bias.copy_(old.value1_heads[src_b].bias)
            else:
                self.value1_heads[b].weight[:, :old.hidden].copy_(old.value1_wl.weight)
                self.value1_heads[b].bias.copy_(old.value1_wl.bias)

            if self.value_depth == 2:
                if old_depth == 2 and hasattr(old, "value2_heads"):
                    self.value2_heads[b].weight.copy_(old.value2_heads[src_b].weight)
                    self.value2_heads[b].bias.copy_(old.value2_heads[src_b].bias)
                    self.value3_heads[b].weight.copy_(old.value3_heads[src_b].weight)
                    self.value3_heads[b].bias.copy_(old.value3_heads[src_b].bias)
                else:
                    # Identity 2nd layer: clipped_relu(I * h1 + 0) = h1, preserving function at epoch 0
                    self.value2_heads[b].weight.copy_(torch.eye(32))
                    self.value2_heads[b].bias.zero_()
                    old_v2 = old.value2_heads[src_b] if hasattr(old, "value2_heads") else old.value2_wl
                    self.value3_heads[b].weight.copy_(old_v2.weight)
                    self.value3_heads[b].bias.copy_(old_v2.bias)
            else:
                old_v2 = old.value2_heads[src_b] if hasattr(old, "value2_heads") else old.value2_wl
                self.value2_heads[b].weight.copy_(old_v2.weight)
                self.value2_heads[b].bias.copy_(old_v2.bias)

        if self.hidden > old.hidden:
            nn.init.normal_(self.fc1.weight[old.hidden:], std=0.01)
            self.fc1.bias[old.hidden:].fill_(0.1)

    def layout_keys(self):
        keys = ["w1", "b1"]
        if self.value_buckets == 1 and self.value_depth == 1:
            keys.extend(["wv1_wl", "bv1_wl", "wv2_wl", "bv2_wl"])
        elif self.value_depth == 2:
            for b in range(self.value_buckets):
                keys.extend([f"wv1_wl_{b}", f"bv1_wl_{b}", f"wv2_wl_{b}", f"bv2_wl_{b}", f"wv3_wl_{b}", f"bv3_wl_{b}"])
        else:
            for b in range(self.value_buckets):
                keys.extend([f"wv1_wl_{b}", f"bv1_wl_{b}", f"wv2_wl_{b}", f"bv2_wl_{b}"])
        keys.extend(["wp", "bp"])
        return tuple(keys)

    def arrays(self):
        def a(t):
            return t.detach().cpu().numpy().astype("<f4")
        d = dict(w1=a(self.fc1.weight.T), b1=a(self.fc1.bias))
        if self.value_buckets == 1 and self.value_depth == 1:
            d["wv1_wl"] = a(self.value1_heads[0].weight.T)
            d["bv1_wl"] = a(self.value1_heads[0].bias)
            d["wv2_wl"] = a(self.value2_heads[0].weight).reshape(32)
            d["bv2_wl"] = a(self.value2_heads[0].bias)
        elif self.value_depth == 2:
            for b in range(self.value_buckets):
                d[f"wv1_wl_{b}"] = a(self.value1_heads[b].weight.T)
                d[f"bv1_wl_{b}"] = a(self.value1_heads[b].bias)
                d[f"wv2_wl_{b}"] = a(self.value2_heads[b].weight.T)
                d[f"bv2_wl_{b}"] = a(self.value2_heads[b].bias)
                d[f"wv3_wl_{b}"] = a(self.value3_heads[b].weight).reshape(32)
                d[f"bv3_wl_{b}"] = a(self.value3_heads[b].bias)
        else:
            for b in range(self.value_buckets):
                d[f"wv1_wl_{b}"] = a(self.value1_heads[b].weight.T)
                d[f"bv1_wl_{b}"] = a(self.value1_heads[b].bias)
                d[f"wv2_wl_{b}"] = a(self.value2_heads[b].weight).reshape(32)
                d[f"bv2_wl_{b}"] = a(self.value2_heads[b].bias)
        d["wp"] = a(self.policy.weight)
        d["bp"] = a(self.policy.bias)
        return d

    @torch.no_grad()
    def load_float(self, path):
        raw = np.fromfile(path, dtype="<f4")
        shapes = [v.shape for v in self.arrays().values()]
        sizes = [int(np.prod(shape)) for shape in shapes]
        if raw.size != sum(sizes) or not np.isfinite(raw).all():
            raise ValueError("weight size or values do not match the requested architecture")
        parts, offset = {}, 0
        for key, shape, size in zip(self.layout_keys(), shapes, sizes):
            parts[key] = torch.from_numpy(raw[offset:offset + size].copy().reshape(shape))
            offset += size
        self.fc1.weight.copy_(parts["w1"].T)
        self.fc1.bias.copy_(parts["b1"])
        self.policy.weight.copy_(parts["wp"])
        self.policy.bias.copy_(parts["bp"])
        if self.value_buckets == 1 and self.value_depth == 1:
            self.value1_heads[0].weight.copy_(parts["wv1_wl"].T)
            self.value1_heads[0].bias.copy_(parts["bv1_wl"])
            self.value2_heads[0].weight.copy_(parts["wv2_wl"].reshape(1, 32))
            self.value2_heads[0].bias.copy_(parts["bv2_wl"])
        elif self.value_depth == 2:
            for b in range(self.value_buckets):
                self.value1_heads[b].weight.copy_(parts[f"wv1_wl_{b}"].T)
                self.value1_heads[b].bias.copy_(parts[f"bv1_wl_{b}"])
                self.value2_heads[b].weight.copy_(parts[f"wv2_wl_{b}"].T)
                self.value2_heads[b].bias.copy_(parts[f"bv2_wl_{b}"])
                self.value3_heads[b].weight.copy_(parts[f"wv3_wl_{b}"].reshape(1, 32))
                self.value3_heads[b].bias.copy_(parts[f"bv3_wl_{b}"])
        else:
            for b in range(self.value_buckets):
                self.value1_heads[b].weight.copy_(parts[f"wv1_wl_{b}"].T)
                self.value1_heads[b].bias.copy_(parts[f"bv1_wl_{b}"])
                self.value2_heads[b].weight.copy_(parts[f"wv2_wl_{b}"].reshape(1, 32))
                self.value2_heads[b].bias.copy_(parts[f"bv2_wl_{b}"])

    @torch.no_grad()
    def clip_weights(self):
        self.fc1.weight.clamp_(-32767 / 255, 32767 / 255)
        self.fc1.bias.clamp_(-32767 / 255, 32767 / 255)
        self.policy.weight.clamp_(-127 / 64, 127 / 64)
        for h in self.value1_heads:
            h.weight.clamp_(-127 / 64, 127 / 64)
        for h in self.value2_heads:
            h.weight.clamp_(-127 / 64, 127 / 64)
        if self.value3_heads is not None:
            for h in self.value3_heads:
                h.weight.clamp_(-127 / 64, 127 / 64)


def export(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = model.arrays()
    layout = model.layout_keys()
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as output:
        for key in layout:
            output.write(np.ascontiguousarray(arrays[key], dtype="<f4").tobytes())
    temp.replace(path)
    quant_path = path.with_name(path.stem + "_int8.bin")
    temp = quant_path.with_suffix(".tmp")
    write_quantized(quantize(arrays), temp)
    temp.replace(quant_path)
    manifest = dict(schema="zquoridor.student.v1", architecture=model.architecture,
                    features=FEATURES[model.architecture], hidden=model.hidden, value_hidden=32,
                    value_buckets=model.value_buckets, value_depth=model.value_depth,
                    policy_out=209, qa=255, qb=64, qat=model.qat,
                    cpp_flags=[f"-DZQ_NNUE_RACE_FEATURES={int(model.architecture in ('race', 'multipath', 'margin_regime', 'phase', 'margin_phase', 'multipath_phase', 'multipath_phase_contact', 'multipath_phase_bucketed', 'multipath_phase_deep', 'multipath_phase_contact_bucketed'))}",
                               f"-DZQ_NNUE_MULTIPATH_FEATURES={int('multipath' in model.architecture)}",
                               f"-DZQ_NNUE_MARGIN_REGIME_FEATURES={int('margin_regime' in model.architecture or 'margin_phase' in model.architecture)}",
                               f"-DZQ_NNUE_PHASE_FEATURES={int('phase' in model.architecture)}",
                               f"-DZQ_NNUE_CONTACT_FEATURES={int('contact' in model.architecture)}",
                               f"-DZQ_NNUE_HIDDEN={model.hidden}",
                               f"-DZQ_NNUE_VALUE_BUCKETS={model.value_buckets}",
                               f"-DZQ_NNUE_VALUE_DEPTH={model.value_depth}"],
                    float_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    int8_sha256=hashlib.sha256(quant_path.read_bytes()).hexdigest())
    path.with_suffix(".architecture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
