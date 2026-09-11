#!/usr/bin/env python3
"""Relabel ZQuoridor positions with a Claustrophobia TorchScript teacher.

The teacher's own Rust encoder is invoked through ``claustrophobia_encode_bridge``
so feature-plane and legal-action semantics stay owned by Claustrophobia. The
teacher logits are masked with Claustrophobia's exact legal-action mask before
softmax, matching its real evaluator. The resulting 209-way soft policy is then
converted from Claustrophobia's canonical action frame to the ZQuoridor
canonical frame before it is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from targets import POLICY_DIM, claustrophobia_policy_to_zq

PLANES = 20
BOARD = 9
TENSOR_LEN = PLANES * BOARD * BOARD


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_positions(path: Path) -> list[dict]:
    positions = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "zquoridor.position.v1":
                raise ValueError(f"{path}:{lineno}: unsupported position schema")
            history = row.get("history")
            if not isinstance(history, list) or not all(isinstance(x, str) for x in history):
                raise ValueError(f"{path}:{lineno}: invalid history")
            side = int(row.get("side_to_move", len(history) & 1))
            if side != (len(history) & 1):
                raise ValueError(f"{path}:{lineno}: side_to_move/history parity mismatch")
            positions.append(row)
    if not positions:
        raise ValueError("position corpus is empty")
    return positions


def encode_positions(positions: Sequence[dict], bridge: Path) -> tuple[np.ndarray, np.ndarray]:
    if not bridge.is_file():
        raise FileNotFoundError(f"Claustrophobia encoder bridge not found: {bridge}")
    with tempfile.TemporaryDirectory(prefix="zq_claustro_") as tmp:
        tmp = Path(tmp)
        inp = tmp / "positions.tsv"
        raw = tmp / "planes.f32"
        mask_raw = tmp / "legal.u8"
        with inp.open("w", encoding="utf-8") as fh:
            for row in positions:
                fh.write(str(row["id"]) + "\t" + " ".join(row["history"]) + "\n")
        subprocess.run([str(bridge), str(inp), str(raw), str(mask_raw)], check=True)
        flat = np.fromfile(raw, dtype="<f4")
        masks = np.fromfile(mask_raw, dtype=np.uint8)
    expected = len(positions) * TENSOR_LEN
    if flat.size != expected:
        raise ValueError(
            f"encoder returned {flat.size} floats; expected {expected} for {len(positions)} positions"
        )
    expected_masks = len(positions) * POLICY_DIM
    if masks.size != expected_masks:
        raise ValueError(
            f"encoder returned {masks.size} legal-mask bytes; expected {expected_masks}"
        )
    masks = masks.reshape(len(positions), POLICY_DIM).astype(bool)
    if not masks.any(axis=1).all():
        raise ValueError("encoder returned a position without legal actions")
    return flat.reshape(len(positions), PLANES, BOARD, BOARD), masks


def forward_teacher(planes: np.ndarray, legal_masks: np.ndarray, checkpoint: Path,
                    batch_size: int, device: torch.device,
                    store_logits: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    teacher = torch.jit.load(str(checkpoint), map_location=device).eval()
    n = len(planes)
    probs = np.empty((n, POLICY_DIM), dtype=np.float32)
    values = np.empty(n, dtype=np.float32)
    logits_out = np.empty((n, POLICY_DIM), dtype=np.float32) if store_logits else None
    illegal_mass_raw = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            xb = torch.from_numpy(planes[start:stop]).to(device)
            legal = torch.from_numpy(legal_masks[start:stop]).to(device=device, dtype=torch.bool)
            out = teacher(xb)
            logits = out[0].float()
            value = out[1].float().reshape(-1)
            if logits.ndim != 2 or logits.shape[1] != POLICY_DIM:
                raise ValueError(f"teacher returned policy shape {tuple(logits.shape)}")

            # Diagnostic only: quantify how wrong an unmasked direct softmax
            # would have been. Training targets below always use the evaluator-
            # faithful masked softmax.
            raw_probs = torch.softmax(logits, dim=1)
            illegal_mass_raw[start:stop] = (
                raw_probs.masked_fill(legal, 0.0).sum(dim=1).cpu().numpy()
            )

            masked_logits = logits.masked_fill(~legal, float("-inf"))
            probs[start:stop] = torch.softmax(masked_logits, dim=1).cpu().numpy()
            values[start:stop] = value.cpu().numpy()
            if logits_out is not None:
                # Store raw network logits. Legality is separately represented by
                # the policy target; preserving raw logits is useful for analysis.
                logits_out[start:stop] = logits.cpu().numpy()
    return probs, values, logits_out, illegal_mass_raw


def convert_policy_frames(values: np.ndarray, positions: Sequence[dict]) -> np.ndarray:
    out = np.empty_like(values)
    for index, row in enumerate(positions):
        out[index] = np.asarray(
            claustrophobia_policy_to_zq(values[index], int(row["side_to_move"])),
            dtype=np.float32,
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher-name", default="claustrophobia-v1.3.1")
    parser.add_argument("--teacher-commit", default="ae093653e62ad700e201706fa5ed767093d0d68e")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--store-logits", action="store_true")
    parser.add_argument("--float32-policy", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"teacher checkpoint not found: {args.checkpoint}")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)

    try:
        positions = load_positions(args.positions)
        planes, legal_masks = encode_positions(positions, args.bridge)
        probs, values, logits, illegal_mass_raw = forward_teacher(
            planes, legal_masks, args.checkpoint, args.batch_size, device, args.store_logits
        )
        probs = convert_policy_frames(probs, positions)
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    policy_dtype = np.float32 if args.float32_policy else np.float16
    arrays = {
        "id": np.asarray([row["id"] for row in positions], dtype="S24"),
        "side_to_move": np.asarray([row["side_to_move"] for row in positions], dtype=np.uint8),
        "policy": probs.astype(policy_dtype),
        "value": values.astype(np.float16),
        "raw_illegal_policy_mass": illegal_mass_raw.astype(np.float16),
    }
    if logits is not None:
        arrays["logits"] = convert_policy_frames(logits, positions).astype(policy_dtype)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)

    top1 = probs.argmax(axis=1)
    manifest = {
        "schema": "zquoridor.teacher.soft_targets.v1",
        "positions": str(args.positions),
        "out": str(args.out),
        "samples": len(positions),
        "teacher": args.teacher_name,
        "teacher_commit": args.teacher_commit,
        "teacher_checkpoint": str(args.checkpoint),
        "teacher_checkpoint_sha256": sha256(args.checkpoint),
        "mode": "network",
        "input_planes": PLANES,
        "policy_dim": POLICY_DIM,
        "policy_dtype": str(np.dtype(policy_dtype)),
        "value_dtype": "float16",
        "stored_logits": logits is not None,
        "device": str(device),
        "policy_frame": "zquoridor canonical mover frame",
        "teacher_encoder": "Claustrophobia encode20 + legal_mask_into via exact Rust bridge",
        "policy_normalization": "softmax after exact teacher legal-action mask",
        "raw_illegal_policy_mass_mean": float(illegal_mass_raw.mean()),
        "raw_illegal_policy_mass_p90": float(np.quantile(illegal_mass_raw, 0.90)),
        "raw_illegal_policy_mass_max": float(illegal_mass_raw.max()),
        "top1_unique_actions": int(len(np.unique(top1))),
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
    }
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
