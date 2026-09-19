"""Compact NNUE students with explicit architecture and portable exports."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from quantize_nnue import quantize, write_quantized

FEATURES = {"base": 354, "race": 456, "race_regime": 588}
LAYOUT = ("w1", "b1", "wv1_wl", "bv1_wl", "wv2_wl", "bv2_wl", "wp", "bp")


def encode_features(data, indices, architecture="base"):
    from train_teacher_policy import dense_features
    x = dense_features(data, indices)
    if architecture == "base":
        return x
    if architecture not in ("race", "race_regime"):
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
    xr = np.concatenate((x, extra), axis=1)
    if architecture == "race":
        return xr

    # Exact distance margin crossed with a coarse wall regime. This keeps
    # one additional active feature while separating "ahead with no walls"
    # from the same margin when both players still retain wall resources.
    regime = np.zeros(n, dtype=np.int64)
    regime[(ow == 0) & (pw > 0)] = 1
    regime[(ow > 0) & (pw == 0)] = 2
    regime[(ow == 0) & (pw == 0)] = 3
    regime_extra = np.zeros((n, 132), dtype=np.float32)
    margin = np.clip(own - opp, -16, 16) + 16
    regime_extra[rows, margin * 4 + regime] = 1
    return np.concatenate((xr, regime_extra), axis=1)


def _round_ste(x, scale):
    rounded = torch.round(x * scale) / scale
    return x + (rounded - x).detach()


class Student(nn.Module):
    def __init__(self, architecture="base", hidden=256, qat=False):
        super().__init__()
        if architecture not in FEATURES or hidden not in (128, 256, 384, 512):
            raise ValueError("architecture must be base/race/race_regime; hidden must be 128/256/384/512")
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
                    cpp_flags=[f"-DZQ_NNUE_RACE_FEATURES={int(model.architecture in ('race', 'race_regime'))}",
                               f"-DZQ_NNUE_RACE_REGIME_FEATURES={int(model.architecture == 'race_regime')}",
                               f"-DZQ_NNUE_HIDDEN={model.hidden}"],
                    float_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    int8_sha256=hashlib.sha256(quant_path.read_bytes()).hexdigest())
    path.with_suffix(".architecture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
