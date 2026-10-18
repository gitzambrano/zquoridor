"""Experimental models and state-only encoders for the architecture campaign.

All action indices use the canonical mover frame: 81 cells, 64 horizontal
walls, and 64 vertical walls. Encoders never read targets or search metadata.
"""
from collections import deque
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .student_model import Student, encode_features, _ORTH_NEIGHBORS, _EDGE_H_MASKS, _EDGE_V_MASKS
    from .student_model import export as export_native
except ImportError:
    from student_model import Student, encode_features, _ORTH_NEIGHBORS, _EDGE_H_MASKS, _EDGE_V_MASKS
    from student_model import export as export_native

BASE = "multipath_phase_contact_bucketed"
SPECS = {
    "race_joint": dict(extra=121, transfer="exact", description="Joint distance buckets near the goal"),
    "corridor": dict(extra=64, transfer="exact", description="Goal-directed exits and forced corridor length"),
    "wall_pairs": dict(extra=512, transfer="exact", description="Hashed wall pairs and boundary motifs relative to pawns"),
    "tactical_aux": dict(extra=8, transfer="exact", description="Training-only topology targets"),
    "context_gate": dict(extra=0, transfer="exact", description="Identity-initialized channel group modulation"),
    "action_scorer": dict(extra=0, transfer="exact", description="Shared nonlinear decoder for all 209 actions"),
    "graph": dict(extra=81 * 12 + 81 * 4, transfer="distillation", description="Four graph blocks with goal distances"),
    "transformer": dict(extra=24 * 8 + 24, transfer="distillation", description="Two attention blocks over walls, pawns, goals, and stock"),
}
SCHEMA = "zquoridor.architecture_campaign.model.v1"


@lru_cache(maxsize=8192)
def topology(walls_h, walls_v):
    """Cache open neighbors and complete distance maps for a wall topology."""
    neighbors = _ORTH_NEIGHBORS.astype(np.int64).copy()
    blocked = ((np.uint64(walls_h) & _EDGE_H_MASKS) != 0) | ((np.uint64(walls_v) & _EDGE_V_MASKS) != 0)
    neighbors[blocked] = -1
    maps = []
    for goal in (8, 0):
        distance = np.full(81, 81, dtype=np.int64)
        queue = deque(range(goal * 9, goal * 9 + 9))
        distance[list(queue)] = 0
        while queue:
            cell = queue.popleft()
            for nxt in neighbors[cell]:
                if nxt >= 0 and distance[nxt] > distance[cell] + 1:
                    distance[nxt] = distance[cell] + 1
                    queue.append(int(nxt))
        maps.append(distance)
    return neighbors, np.stack(maps)


def corridor_summary(neighbors, distances, pawn):
    """Measure local forward options and the forced shortest-path prefix."""
    legal = neighbors[pawn]
    useful = [int(n) for n in legal if n >= 0 and distances[n] < distances[pawn]]
    length, cell = 0, pawn
    while length < 15 and distances[cell] > 0:
        options = [int(n) for n in neighbors[cell] if n >= 0 and distances[n] == distances[cell] - 1]
        if len(options) != 1:
            break
        length += 1
        cell = options[0]
    return len(useful), length, int(np.count_nonzero(legal >= 0))


def pair_features(walls_h, walls_v, own, opp):
    """Use stable hash buckets for pair shape, board region, and pawn relation."""
    walls = [(slot // 8, slot % 8, orientation) for orientation, bits in enumerate((walls_h, walls_v))
             for slot in range(64) if (bits >> slot) & 1]
    out = np.zeros(512, dtype=np.float32)
    def insert(values):
        code = 2166136261
        for value in values:
            code = ((code ^ (int(value) + 32)) * 16777619) & 0xffffffff
        out[code % 512] = 1
    for i, (r, c, orientation) in enumerate(walls):
        for side, pawn in enumerate((own, opp)):
            pr, pc = divmod(pawn, 9)
            relative = (side, int(np.clip(r - pr, -3, 3)), int(np.clip(c - pc, -3, 3)))
            insert((0, orientation, min(r, 7-r), min(c, 7-c), *relative))
            for rr, cc, oo in walls[i+1:]:
                if abs(rr-r) + abs(cc-c) <= 4:
                    insert((1, orientation, oo, rr-r, cc-c, *relative))
    return out


def encode(data, indices, architecture):
    if architecture not in SPECS:
        return encode_features(data, indices, architecture)
    base = encode_features(data, indices, BASE)
    extra = np.zeros((len(indices), SPECS[architecture]["extra"]), dtype=np.float32)
    if architecture in ("context_gate", "action_scorer"):
        return base
    for j, row in enumerate(indices):
        own, opp = int(data["own_pawn"][row]), int(data["opp_pawn"][row])
        wh, wv = int(data["walls_h"][row]), int(data["walls_v"][row])
        if architecture == "race_joint":
            a, b = min(int(data["own_dist"][row]), 10), min(int(data["opp_dist"][row]), 10)
            extra[j, a * 11 + b] = 1
            continue
        if architecture == "wall_pairs":
            extra[j] = pair_features(wh, wv, own, opp)
            continue
        if architecture == "transformer":
            tokens = np.zeros((24, 8), dtype=np.float32)
            mask = np.zeros(24, dtype=np.float32)
            # Token fields: row, column, pawn, horizontal, vertical, goal, own stock, opponent stock.
            tokens[0, :3] = (own // 9 / 8, own % 9 / 8, 1)
            tokens[1, :3] = (opp // 9 / 8, opp % 9 / 8, -1)
            tokens[2, [0, 5]] = (1, 1)
            tokens[3, 5] = -1
            tokens[2:4, 6] = int(data["walls_left_own"][row]) / 10
            tokens[2:4, 7] = int(data["walls_left_opp"][row]) / 10
            mask[:4] = 1
            cursor = 4
            for orientation, bits in enumerate((wh, wv)):
                for slot in range(64):
                    if not (bits >> slot) & 1:
                        continue
                    if cursor >= 24:
                        raise ValueError("A position contains more than 20 walls.")
                    tokens[cursor, :2] = (slot // 8 / 8, slot % 8 / 8)
                    tokens[cursor, 3 + orientation] = 1
                    mask[cursor] = 1
                    cursor += 1
            extra[j] = np.r_[tokens.ravel(), mask]
            continue
        neighbors, distances = topology(wh, wv)
        summaries = [corridor_summary(neighbors, distances[p], pawn) for p, pawn in enumerate((own, opp))]
        if architecture == "corridor":
            for p, (options, length, exits) in enumerate(summaries):
                offset = p * 32
                extra[j, offset + min(options, 4)] = 1
                extra[j, offset + 5 + length] = 1
                extra[j, offset + 21 + exits] = 1
                extra[j, offset + 26 + min(int(distances[p, (own, opp)[p]]), 5)] = 1
        elif architecture == "tactical_aux":
            # These deterministic labels supervise an auxiliary head. They are not trunk inputs.
            for p, (options, length, exits) in enumerate(summaries):
                extra[j, p*4:p*4+4] = (options/4, length/15, exits/4, min(int(distances[p, (own, opp)[p]]), 40)/40)
        elif architecture == "graph":
            nodes = np.zeros((81, 12), dtype=np.float32)
            cells = np.arange(81)
            nodes[:, 0], nodes[:, 1] = cells // 9 / 8, cells % 9 / 8
            nodes[own, 2], nodes[opp, 3] = 1, 1
            nodes[:, 4:6] = np.minimum(distances.T, 40) / 40
            nodes[72:, 6], nodes[:9, 7] = 1, 1
            nodes[:, 8:12] = neighbors >= 0
            extra[j] = np.r_[nodes.ravel(), neighbors.ravel()]
    return np.concatenate((base, extra), axis=1)


class ExperimentalStudent(Student):
    def __init__(self, architecture, hidden=512, qat=True):
        super().__init__(BASE, hidden, qat=qat)
        self.architecture = architecture
        self.variant = architecture
        self.native_compatible = architecture == "tactical_aux"
        self.transfer = SPECS[architecture]["transfer"]
        if architecture in ("race_joint", "corridor", "wall_pairs"):
            self.fc1 = nn.Linear(858 + SPECS[architecture]["extra"], hidden)
        elif architecture == "context_gate":
            self.gate = nn.Sequential(nn.Linear(106, 16), nn.ReLU(), nn.Linear(16, 16))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        elif architecture == "action_scorer":
            self.action_context = nn.Linear(hidden, 32)
            self.action_embedding = nn.Embedding(209, 32)
            self.action_decoder = nn.Sequential(nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.action_decoder[-1].weight)
            nn.init.zeros_(self.action_decoder[-1].bias)
        elif architecture == "tactical_aux":
            self.topology_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 8))
        elif architecture == "graph":
            self.fc1 = nn.Identity()
            self.node_input = nn.Linear(12, 32)
            self.graph_blocks = nn.ModuleList([nn.Linear(64, 32) for _ in range(4)])
            self.graph_output = nn.Linear(32 * 4 + 106, hidden)
        elif architecture == "transformer":
            self.fc1 = nn.Identity()
            self.token_input = nn.Linear(8, 32)
            layer = nn.TransformerEncoderLayer(32, 2, dim_feedforward=64, dropout=0, batch_first=True)
            self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.attention_output = nn.Linear(32 + 106, hidden)

    def warm_start(self, old):
        if self.transfer != "exact":
            raise ValueError("This model requires distillation instead of a weight copy.")
        # The base helper zeros every parameter. Preserve nonzero adapter features.
        extra = {k: v.clone() for k, v in self.state_dict().items()
                 if k not in old.state_dict() and not k.startswith("fc1.")}
        super().warm_start(old)
        self.load_state_dict(extra, strict=False)

    def activation(self, x):
        if self.variant in ("tactical_aux", "action_scorer"):
            return super().activation(x[:, :858])
        if self.variant == "context_gate":
            a, _ = super().activation(x[:, :858])
            context = x[:, 290:396]
            gate = 1 + .25 * torch.tanh(self.gate(context))
            a = (a * gate.repeat_interleave(self.hidden // 16, dim=1)).clamp(0, 1)
            return self.quantized_activation(a)
        if self.variant == "graph":
            nodes = x[:, 858:858+81*12].reshape(-1, 81, 12)
            neighbors = x[:, 858+81*12:].reshape(-1, 81, 4).long()
            h = F.relu(self.node_input(nodes))
            batch = torch.arange(len(x), device=x.device)[:, None, None]
            mask = (neighbors >= 0).unsqueeze(-1)
            for block in self.graph_blocks:
                messages = h[batch, neighbors.clamp_min(0)] * mask
                mean = messages.sum(2) / mask.sum(2).clamp_min(1)
                h = h + F.relu(block(torch.cat((h, mean), -1)))
            own = (h * nodes[:, :, 2:3]).sum(1)
            opp = (h * nodes[:, :, 3:4]).sum(1)
            pooled = torch.cat((h.mean(1), h.amax(1), own, opp, x[:, 290:396]), 1)
            return self.quantized_activation(self.graph_output(pooled).sigmoid())
        if self.variant == "transformer":
            tokens = x[:, 858:858+24*8].reshape(-1, 24, 8)
            valid = x[:, 858+24*8:] > 0
            h = self.attention(self.token_input(tokens), src_key_padding_mask=~valid)
            pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            return self.quantized_activation(self.attention_output(torch.cat((pooled, x[:, 290:396]), 1)).sigmoid())
        return super().activation(x)

    def quantized_activation(self, activation):
        if not self.qat:
            return activation, None
        integer = activation * 255
        integer = integer + (integer.floor() - integer).detach()
        return integer / 255, integer

    def forward(self, x, **kwargs):
        result = super().forward(x, **kwargs)
        if self.variant == "action_scorer":
            a, _ = self.activation(x)
            features = self.action_context(a)[:, None, :] + self.action_embedding.weight[None, :, :]
            correction = self.action_decoder(features).squeeze(-1)
            result = (result[0], result[1] + correction, *result[2:])
        return result

    def additional_loss(self, x):
        if self.variant != "tactical_aux":
            return x.new_zeros(len(x))
        a, _ = self.activation(x)
        return F.smooth_l1_loss(self.topology_head(a).sigmoid(), x[:, 858:], reduction="none").mean(1)


def make_model(architecture, hidden=512, qat=True):
    if architecture in SPECS:
        return ExperimentalStudent(architecture, hidden, qat)
    return Student(architecture, hidden, qat)


def export(model, path):
    if not isinstance(model, ExperimentalStudent):
        return export_native(model, path)
    if model.native_compatible:
        architecture = model.architecture
        try:
            model.architecture = BASE
            result = export_native(model, path)
        finally:
            model.architecture = architecture
        result["training_variant"] = architecture
        Path(path).with_suffix(".architecture.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        return result
    path = Path(path).with_suffix(".pt")
    temporary = path.with_suffix(".tmp")
    torch.save(dict(schema=SCHEMA, architecture=model.architecture, hidden=model.hidden,
                    qat=model.qat, model={k: v for k, v in model.state_dict().items()
                                         if not k.startswith(("aux_policy.", "aux_value."))}), temporary)
    temporary.replace(path)
    result = dict(schema=SCHEMA, architecture=model.architecture, hidden=model.hidden,
                  features=858+SPECS[model.architecture]["extra"], policy_out=209,
                  runtime="pytorch", native_compatible=False, qat=model.qat,
                  quantization_scope="NNUE trunk and inherited heads; new adapters use floating point",
                  checkpoint=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".architecture.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def load_model(path):
    """Load a portable experimental checkpoint for inference or distillation."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA:
        raise ValueError("Unsupported experimental checkpoint schema.")
    model = make_model(state["architecture"], state["hidden"], state["qat"])
    model.load_state_dict(state["model"])
    return model.eval()
