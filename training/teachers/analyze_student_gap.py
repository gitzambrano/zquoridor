#!/usr/bin/env python3
"""Score student/teacher disagreement and select high-information positions.

Consumes the canonical soft-teacher dataset produced by ``build_teacher_soft.py``
and an existing ZQuoridor float-weight file. Per position it measures policy KL,
teacher/student argmax disagreement, value residual, teacher entropy, and a
combined priority score. Optionally writes a JSONL subset of the original
architecture-neutral positions for expensive search relabeling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

TRAINING = Path(__file__).resolve().parents[1]
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import train_nnue as base  # noqa: E402
from train_teacher_policy import dense_features  # noqa: E402


def load_positions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("schema") != "zquoridor.position.v1":
                raise ValueError(f"{path}:{lineno}: unsupported position schema")
            rows.append(obj)
    if not rows:
        raise ValueError("position corpus is empty")
    return rows


def score_gap(data, weights: Path, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    n = len(data["policy"])
    model = base.QuoridorNNUE().to(device).eval()
    base._load_into_model(model, base._load_raw_weights(str(weights)))

    kl = np.empty(n, dtype=np.float32)
    teacher_entropy = np.empty(n, dtype=np.float32)
    value_residual = np.empty(n, dtype=np.float32)
    teacher_top1 = np.empty(n, dtype=np.int16)
    student_top1 = np.empty(n, dtype=np.int16)
    student_value = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            idx = np.arange(start, stop, dtype=np.int64)
            x = torch.from_numpy(dense_features(data, idx)).to(device)
            target = torch.from_numpy(data["policy"][idx].astype(np.float32)).to(device)
            teacher_v = torch.from_numpy(data["value"][idx].astype(np.float32)).to(device)
            value_logits, logits = model(x)
            logp = F.log_softmax(logits, dim=1)
            entropy = -(target * torch.log(target.clamp_min(1e-12))).sum(dim=1)
            cross_entropy = -(target * logp).sum(dim=1)
            signed_value = 2.0 * torch.sigmoid(value_logits) - 1.0

            kl[start:stop] = (cross_entropy - entropy).clamp_min(0.0).cpu().numpy()
            teacher_entropy[start:stop] = entropy.cpu().numpy()
            value_residual[start:stop] = torch.abs(signed_value - teacher_v).cpu().numpy()
            teacher_top1[start:stop] = target.argmax(dim=1).cpu().numpy().astype(np.int16)
            student_top1[start:stop] = logits.argmax(dim=1).cpu().numpy().astype(np.int16)
            student_value[start:stop] = signed_value.cpu().numpy()

    mismatch = (teacher_top1 != student_top1).astype(np.float32)

    def rank01(x: np.ndarray) -> np.ndarray:
        if len(x) <= 1:
            return np.zeros_like(x, dtype=np.float32)
        order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        return order.astype(np.float32) / np.float32(len(x) - 1)

    priority = mismatch + rank01(kl) + rank01(value_residual)
    return {
        "policy_kl": kl,
        "teacher_entropy": teacher_entropy,
        "value_residual": value_residual,
        "teacher_top1": teacher_top1,
        "student_top1": student_top1,
        "student_value": student_value,
        "argmax_mismatch": mismatch,
        "priority": priority.astype(np.float32),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--student-weights", required=True, type=Path)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--selected-out", type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--min-kl", type=float, default=0.0)
    parser.add_argument("--require-argmax-mismatch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if not 0.0 < args.top_fraction <= 1.0:
        raise SystemExit("top-fraction must be in (0,1]")
    if args.batch_size <= 0 or args.min_kl < 0.0:
        raise SystemExit("batch-size must be positive and min-kl non-negative")
    if args.selected_out is not None and args.positions is None:
        raise SystemExit("--selected-out requires --positions")

    data = np.load(args.data, allow_pickle=False)
    required = {"id", "policy", "value", "own_pawn", "opp_pawn", "walls_h", "walls_v",
                "walls_left_own", "walls_left_opp", "own_dist", "opp_dist"}
    missing = sorted(required - set(data.files))
    if missing:
        raise SystemExit(f"dataset lacks fields: {missing}")
    n = len(data["policy"])
    if data["policy"].shape != (n, base.POLICY_OUT):
        raise SystemExit(f"invalid policy shape {data['policy'].shape}")

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    try:
        scored = score_gap(data, args.student_weights, args.batch_size, device)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    ids = data["id"]
    arrays = {"id": ids, **scored}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)

    eligible = scored["policy_kl"] >= args.min_kl
    if args.require_argmax_mismatch:
        eligible &= scored["argmax_mismatch"] > 0.5
    eligible_idx = np.flatnonzero(eligible)
    k = max(1, int(round(n * args.top_fraction))) if len(eligible_idx) else 0
    if k:
        ranked = eligible_idx[np.argsort(-scored["priority"][eligible_idx], kind="stable")]
        selected_idx = ranked[: min(k, len(ranked))]
    else:
        selected_idx = np.asarray([], dtype=np.int64)

    if args.selected_out is not None:
        positions = load_positions(args.positions)
        if len(positions) != n:
            raise SystemExit(f"positions/data length mismatch: {len(positions)} != {n}")
        position_ids = [str(row["id"]) for row in positions]
        data_ids = [x.decode("ascii") if isinstance(x, bytes) else str(x) for x in ids]
        if position_ids != data_ids:
            raise SystemExit("positions/data IDs are not aligned")
        args.selected_out.parent.mkdir(parents=True, exist_ok=True)
        with args.selected_out.open("w", encoding="utf-8") as fh:
            for idx in selected_idx:
                row = dict(positions[int(idx)])
                meta = dict(row.get("metadata") or {})
                meta["active_learning"] = {
                    "policy_kl": float(scored["policy_kl"][idx]),
                    "value_residual": float(scored["value_residual"][idx]),
                    "argmax_mismatch": bool(scored["argmax_mismatch"][idx] > 0.5),
                    "priority": float(scored["priority"][idx]),
                }
                row["metadata"] = meta
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "zquoridor.teacher.student_gap.v1",
        "samples": n,
        "student_weights": str(args.student_weights),
        "policy_kl_mean": float(scored["policy_kl"].mean()),
        "policy_kl_p90": float(np.quantile(scored["policy_kl"], 0.90)),
        "policy_kl_p99": float(np.quantile(scored["policy_kl"], 0.99)),
        "argmax_mismatch_fraction": float(scored["argmax_mismatch"].mean()),
        "value_residual_mean": float(scored["value_residual"].mean()),
        "value_residual_p90": float(np.quantile(scored["value_residual"], 0.90)),
        "teacher_entropy_mean": float(scored["teacher_entropy"].mean()),
        "selected": int(len(selected_idx)),
        "selected_fraction": float(len(selected_idx) / max(1, n)),
        "top_fraction_requested": args.top_fraction,
        "require_argmax_mismatch": args.require_argmax_mismatch,
        "min_kl": args.min_kl,
        "device": str(device),
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
