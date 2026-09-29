#!/usr/bin/env python3
"""Build a conservative value-only recovery dataset for sidestep_flank.

Critical one-ply neighborhoods are labeled by Claustrophobia deep search.
Background anchors are prefixes from the normal and Center Rush books with
targets frozen to the deployed 2.10 NNUE, so value-only fine-tuning cannot
move an entire phase head without paying an anchor loss.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "training", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_teacher_soft import encode_states
from student_model import Student, encode_features

ARCH = "multipath_phase_bucketed"
HIDDEN = 512

def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def anchor_positions():
    seen = set()
    rows = []
    sources = [
        ROOT / "tools/external/openings_screen_v1.jsonl",
        ROOT / "tools/external/openings_center_rush_50pairs.jsonl",
    ]
    for source_no, path in enumerate(sources):
        for opening_no, row in enumerate(load_jsonl(path)):
            moves = list(row["moves"])
            for end in range(0, len(moves) + 1):
                hist = tuple(moves[:end])
                if hist in seen:
                    continue
                seen.add(hist)
                key = " ".join(hist)
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
                rows.append({
                    "schema": "zquoridor.position.v1",
                    "id": "a" + digest[:23],
                    "side_to_move": end & 1,
                    "opening_index": 10000 + source_no * 1000 + opening_no,
                    "split": "val" if int(digest[:8], 16) % 10 == 0 else "train",
                    "history": list(hist),
                    "metadata": {"role": "anchor", "source": path.name},
                })
    return rows

def infer_base(state, weights: Path):
    n = len(state["own_pawn"])
    data = {k: v for k, v in state.items() if k != "mover"}
    idx = np.arange(n)
    model = Student(ARCH, HIDDEN, qat=True)
    model.load_float(weights)
    model.eval()
    values = []
    policies = []
    with torch.no_grad():
        for start in range(0, n, 512):
            sl = idx[start:start+512]
            x = torch.from_numpy(encode_features(data, sl, ARCH))
            logit, policy = model(x)
            values.append((2.0 * logit.sigmoid() - 1.0).cpu().numpy().astype(np.float32))
            policies.append(torch.softmax(policy, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(values), np.concatenate(policies)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--base-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    critical = load_jsonl(args.positions)
    ids = [r["id"] for r in critical]
    with np.load(args.targets, allow_pickle=False) as t:
        tid = [x.decode("ascii") for x in t["id"]]
        by_id = {x:i for i,x in enumerate(tid)}
        order = np.asarray([by_id[x] for x in ids], dtype=np.int64)
        c_policy = t["policy"][order].astype(np.float32)
        c_value = t["value"][order].astype(np.float32)
        agreement = t["budget_agreement"][order].astype(np.float32)
    c_state = encode_states(critical, args.encoder)

    anchors = anchor_positions()
    critical_hist = {tuple(r["history"]) for r in critical}
    anchors = [r for r in anchors if tuple(r["history"]) not in critical_hist]
    a_state = encode_states(anchors, args.encoder)
    a_value, a_policy = infer_base(a_state, args.base_weights)

    # Deep teacher samples dominate locally, but anchors remain numerous and
    # preserve the deployed value surface away from the diagnosed states.
    c_weight = np.clip(4.0 + 6.0 * agreement, 4.0, 10.0).astype(np.float32)
    a_weight = np.full(len(anchors), 1.0, dtype=np.float32)

    def arr(rows, state, policy, value, weight, prefix):
        result = {k: v for k, v in state.items() if k != "mover"}
        result.update({
            "policy": policy.astype(np.float32),
            "value": value.astype(np.float32),
            "weight": weight.astype(np.float32),
            "is_val": np.asarray([r.get("split") == "val" for r in rows], dtype=np.bool_),
            "group_id": np.asarray([f"{prefix}_{r['id']}" for r in rows]),
            "opening_index": np.asarray([int(r.get("opening_index", -1)) for r in rows], dtype=np.int32),
        })
        return result

    cd = arr(critical, c_state, c_policy, c_value, c_weight, "critical")
    ad = arr(anchors, a_state, a_policy, a_value, a_weight, "anchor")
    keys = sorted(cd.keys())
    combined = {k: np.concatenate([cd[k], ad[k]]) for k in keys}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **combined)

    manifest = {
        "schema": "zquoridor.dataset.sidestep_value_recovery.v1",
        "critical_samples": len(critical),
        "critical_train": int((~cd["is_val"]).sum()),
        "critical_val": int(cd["is_val"].sum()),
        "anchor_samples": len(anchors),
        "anchor_train": int((~ad["is_val"]).sum()),
        "anchor_val": int(ad["is_val"].sum()),
        "teacher_weight_mean": float(c_weight.mean()),
        "teacher_budget_agreement_mean": float(agreement.mean()),
        "architecture": ARCH,
        "base_weights": str(args.base_weights),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
