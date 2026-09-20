"""Compact NNUE students with explicit architecture and portable exports."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from quantize_nnue import quantize, write_quantized

FEATURES = {"base": 354, "race": 456, "multipath": 480}
LAYOUT = ("w1", "b1", "wv1_wl", "bv1_wl", "wv2_wl", "bv2_wl", "wp", "bp")


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
    from train_teacher_policy import dense_features
    x = dense_features(data, indices)
    if architecture == "base":
        return x
    if architecture not in ("race", "multipath"):
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

    return np.concatenate((x, mp), axis=1)


def _round_ste(x, scale):
    rounded = torch.round(x * scale) / scale
    return x + (rounded - x).detach()


class Student(nn.Module):
    def __init__(self, architecture="base", hidden=256, qat=False):
        super().__init__()
        if architecture not in FEATURES or hidden not in (128, 256, 384, 512):
            raise ValueError(f"architecture must be one of {list(FEATURES.keys())}; hidden must be 128/256/384/512")
        self.architecture, self.hidden, self.qat = architecture, hidden, qat
        self.fc1 = nn.Linear(FEATURES[architecture], hidden)
        self.value1_wl = nn.Linear(hidden, 32)
        self.value2_wl = nn.Linear(32, 1)
        self.policy = nn.Linear(hidden, 209)

    def forward(self, x):
        if not self.qat:
            a = self.fc1(x).clamp(0, 1).square()
            return self.value2_wl(self.value1_wl(a).clamp(0, 1)).squeeze(-1), self.policy(a)
        # Match the deployed integer scales, including the SCReLU truncation.
        # Accumulate exact integer-valued floats before division. Summing
        # dequantized rows can cross a truncation boundary from roundoff alone.
        a = F.linear(x, _round_ste(self.fc1.weight * 255, 1),
                     _round_ste(self.fc1.bias * 255, 1))
        a = a.clamp(0, 255).square() / 255
        a = (a + (torch.floor(a) - a).detach()) / 255
        h = F.linear(a, _round_ste(self.value1_wl.weight, 64),
                     _round_ste(self.value1_wl.bias, 255 * 64)).clamp(0, 1)
        v = F.linear(h, _round_ste(self.value2_wl.weight, 64),
                     _round_ste(self.value2_wl.bias, 255 * 64 * 64)).squeeze(-1)
        p = F.linear(a, _round_ste(self.policy.weight, 64), _round_ste(self.policy.bias, 255 * 64))
        return v, p

    @torch.no_grad()
    def warm_start(self, old):
        if self.hidden < old.hidden or self.fc1.in_features < old.fc1.in_features:
            raise ValueError("warm start cannot shrink a network; use output distillation from scratch")
        for param in self.parameters():
            param.zero_()
        self.fc1.weight[:old.hidden, :old.fc1.in_features].copy_(old.fc1.weight)
        self.fc1.bias[:old.hidden].copy_(old.fc1.bias)
        self.value1_wl.weight[:, :old.hidden].copy_(old.value1_wl.weight)
        self.value1_wl.bias.copy_(old.value1_wl.bias)
        self.value2_wl.load_state_dict(old.value2_wl.state_dict())
        self.policy.weight[:, :old.hidden].copy_(old.policy.weight)
        self.policy.bias.copy_(old.policy.bias)
        # New units need nonzero inputs to receive gradients; zero output
        # columns preserve the original function until those columns train.
        if self.hidden > old.hidden:
            nn.init.normal_(self.fc1.weight[old.hidden:], std=0.01)
            self.fc1.bias[old.hidden:].fill_(0.1)

    def arrays(self):
        def a(t):
            return t.detach().cpu().numpy().astype("<f4")
        return dict(w1=a(self.fc1.weight.T), b1=a(self.fc1.bias),
                    wv1_wl=a(self.value1_wl.weight.T), bv1_wl=a(self.value1_wl.bias),
                    wv2_wl=a(self.value2_wl.weight).reshape(32), bv2_wl=a(self.value2_wl.bias),
                    wp=a(self.policy.weight), bp=a(self.policy.bias))

    @torch.no_grad()
    def load_float(self, path):
        raw = np.fromfile(path, dtype="<f4")
        shapes = [v.shape for v in self.arrays().values()]
        sizes = [int(np.prod(shape)) for shape in shapes]
        if raw.size != sum(sizes) or not np.isfinite(raw).all():
            raise ValueError("weight size or values do not match the requested architecture")
        parts, offset = [], 0
        for shape, size in zip(shapes, sizes):
            parts.append(torch.from_numpy(raw[offset:offset + size].copy().reshape(shape)))
            offset += size
        for param, value in zip((self.fc1.weight, self.fc1.bias, self.value1_wl.weight,
                                 self.value1_wl.bias, self.value2_wl.weight, self.value2_wl.bias,
                                 self.policy.weight, self.policy.bias),
                                (parts[0].T, parts[1], parts[2].T, parts[3], parts[4].reshape(1, 32),
                                 parts[5], parts[6], parts[7])):
            param.copy_(value)

    @torch.no_grad()
    def clip_weights(self):
        self.fc1.weight.clamp_(-32767 / 255, 32767 / 255)
        self.fc1.bias.clamp_(-32767 / 255, 32767 / 255)
        for layer in (self.value1_wl, self.value2_wl, self.policy):
            layer.weight.clamp_(-127 / 64, 127 / 64)


def export(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = model.arrays()
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as output:
        for key in LAYOUT:
            output.write(np.ascontiguousarray(arrays[key], dtype="<f4").tobytes())
    temp.replace(path)
    quant_path = path.with_name(path.stem + "_int8.bin")
    temp = quant_path.with_suffix(".tmp")
    write_quantized(quantize(arrays), temp)
    temp.replace(quant_path)
    manifest = dict(schema="zquoridor.student.v1", architecture=model.architecture,
                    features=FEATURES[model.architecture], hidden=model.hidden, value_hidden=32,
                    policy_out=209, qa=255, qb=64, qat=model.qat,
                    cpp_flags=[f"-DZQ_NNUE_RACE_FEATURES={int(model.architecture in ('race', 'multipath'))}",
                               f"-DZQ_NNUE_MULTIPATH_FEATURES={int(model.architecture == 'multipath')}",
                               f"-DZQ_NNUE_HIDDEN={model.hidden}"],
                    float_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    int8_sha256=hashlib.sha256(quant_path.read_bytes()).hexdigest())
    path.with_suffix(".architecture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
