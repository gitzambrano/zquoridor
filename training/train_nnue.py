#!/usr/bin/env python3
"""Execute tri-network training tactics and convex model soup unification.

This is the primary production training coordinator for Zquoridor NNUE.
It trains three specialized networks across diverse optimization basins
and unifies them via a convex model soup:

1. Arm 1 (Sprint & Tactical Sharpness):
   Softened auxiliary policy (T=2.0, beta=0.15), low LR with cosine decay.
   Excels at direct pawn sprints and sharp head-to-head tactical play.

2. Arm 2 (Surprise Weighting & Wall Defense):
   Policy surprise weighting (KL divergence prioritizing unexpected search discoveries)
   combined with auxiliary policy (T=2.0, beta=0.15). Excels at defensive wall placement.

3. Arm 3 (Deep Positional Regularization):
   High auxiliary policy weight (beta=0.25, lr-scale 5.0) with surprise weighting
   on the full dataset mixture. Yields minimal global loss and deep positional stability.

4. Unification (Convex Model Soup):
   Convex linear combination blending all three arms (33.3% Arm 1 + 33.3% Arm 2 + 33.4% Arm 3).
   Performs parameter clipping, quantizes to int8 matching C++ NNUE layout (731,172 bytes),
   and compiles native evaluation binaries.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

import run_experiment as r
from student_model import Student, export
from model_soup import create_soup
from compute_surprise_weights import compute_surprise_weights

CONFIG: Dict[str, Any] = {
    "mode": "all",  # Options: 'arm1', 'arm2', 'arm3', 'soup', 'all'
    "data": str(ROOT / "results" / "experiments" / "production-central-finetune-20261002" / "mixed_dataset"),
    "surprise_data": str(ROOT / "results" / "experiments" / "production-central-finetune-20261002" / "mixed_dataset_surprise_v2"),
    "out_root": str(ROOT / "results" / "experiments" / "production-3.01"),
    "init_from": str(ROOT / "data" / "nnue" / "nnue_weights.bin"),
    "init_architecture": "multipath_phase_bucketed",
    "init_hidden": 512,
    "architecture": "multipath_phase_bucketed",
    "hidden": 512,
    "epochs": 20,
    "batch_size": 1024,
    "lr": 1e-05,
    "min_lr": 1e-07,
    "trunk_lr_scale": 0.05,
    "warmup_epochs": 2,
    "weight_decay": 1e-05,
    "seed": 20261004,
    "qat": True,
    "mirror_h": True,
    "checkpoint_every": 5,
    "patience": 12,
    "device": "cuda",
    "cpu_threads": 4,
    # Arm 1 Hyperparameters
    "arm1_aux_temp": 2.0,
    "arm1_aux_weight": 0.15,
    "arm1_aux_lr_scale": 5.0,
    # Arm 2 Hyperparameters
    "arm2_aux_temp": 2.0,
    "arm2_aux_weight": 0.15,
    "arm2_aux_lr_scale": 5.0,
    "arm2_surprise_alpha": 0.5,
    "arm2_surprise_smax": 4.0,
    # Arm 3 Hyperparameters
    "arm3_aux_temp": 2.0,
    "arm3_aux_weight": 0.25,
    "arm3_aux_lr_scale": 5.0,
    # Model Soup Hyperparameters
    "soup_weights": [0.3333, 0.3333, 0.3334],
    "build": True,
}


def train_arm1(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Train Arm 1: Auxiliary Soft Policy for sprint and tactical sharpness."""
    out_dir = Path(cfg["out_root"]) / "arm1_sprint_aux"
    print(f"\n{'=' * 70}\n[TRAIN NNUE] STARTING ARM 1: Sprint & Tactical Sharpness\n{'=' * 70}")
    arm_cfg = dict(
        r.CONFIG,
        data=cfg["data"],
        out_dir=str(out_dir),
        architecture=cfg["architecture"],
        hidden=cfg["hidden"],
        init_from=cfg["init_from"],
        init_architecture=cfg["init_architecture"],
        init_hidden=cfg["init_hidden"],
        from_scratch=False,
        qat=cfg["qat"],
        mirror_h=cfg["mirror_h"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        min_lr=cfg["min_lr"],
        trunk_lr_scale=cfg["trunk_lr_scale"],
        warmup_epochs=cfg["warmup_epochs"],
        weight_decay=cfg["weight_decay"],
        weight_decay_schedule="constant",
        min_weight_decay=cfg["weight_decay"],
        weight_boosts=[],
        max_sample_weight=30.0,
        policy_weight=1.0,
        value_weight=1.0,
        aux_policy_enabled=True,
        aux_policy_temperature=cfg["arm1_aux_temp"],
        aux_policy_weight=cfg["arm1_aux_weight"],
        aux_policy_lr_scale=cfg["arm1_aux_lr_scale"],
        validation_mode="existing",
        train_scope="full",
        grad_clip=1.0,
        patience=cfg["patience"],
        seed=cfg["seed"],
        device=cfg["device"],
        cpu_threads=cfg["cpu_threads"],
        resume=True,
        checkpoint_every=cfg["checkpoint_every"],
        build=cfg["build"],
    )
    r.train(arm_cfg)
    if cfg["build"]:
        r.build_candidate(arm_cfg)
    return out_dir / "best.pt", out_dir / "student_int8.bin"


def ensure_surprise_data(cfg: Dict[str, Any]) -> str:
    """Ensure surprise-weighted dataset exists; generate if missing."""
    surprise_path = Path(cfg["surprise_data"])
    if (surprise_path / "dataset.manifest.json").is_file():
        return str(surprise_path)

    print(f"\n[TRAIN NNUE] Generating surprise weights for dataset: {cfg['data']}")
    surprise_path.mkdir(parents=True, exist_ok=True)
    compute_cfg = {
        "data": cfg["data"],
        "model": cfg["init_from"],
        "architecture": cfg["architecture"],
        "hidden": cfg["hidden"],
        "alpha": cfg["arm2_surprise_alpha"],
        "s_max": cfg["arm2_surprise_smax"],
        "batch_size": 16384,
        "device": cfg["device"],
        "output_dataset_dir": str(surprise_path),
        "output_weights": "",
        "output_kl": "",
        "normalize_mass": True,
    }
    compute_surprise_weights(compute_cfg)
    return str(surprise_path)


def train_arm2(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Train Arm 2: Policy Surprise Weighting for wall placement and defense."""
    out_dir = Path(cfg["out_root"]) / "arm2_surprise_aux"
    surprise_data = ensure_surprise_data(cfg)
    print(f"\n{'=' * 70}\n[TRAIN NNUE] STARTING ARM 2: Surprise Weighting & Wall Defense\n{'=' * 70}")
    arm_cfg = dict(
        r.CONFIG,
        data=surprise_data,
        out_dir=str(out_dir),
        architecture=cfg["architecture"],
        hidden=cfg["hidden"],
        init_from=cfg["init_from"],
        init_architecture=cfg["init_architecture"],
        init_hidden=cfg["init_hidden"],
        from_scratch=False,
        qat=cfg["qat"],
        mirror_h=cfg["mirror_h"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        min_lr=cfg["min_lr"],
        trunk_lr_scale=cfg["trunk_lr_scale"],
        warmup_epochs=cfg["warmup_epochs"],
        weight_decay=cfg["weight_decay"],
        weight_decay_schedule="constant",
        min_weight_decay=cfg["weight_decay"],
        weight_boosts=[],
        max_sample_weight=30.0,
        policy_weight=1.0,
        value_weight=1.0,
        aux_policy_enabled=True,
        aux_policy_temperature=cfg["arm2_aux_temp"],
        aux_policy_weight=cfg["arm2_aux_weight"],
        aux_policy_lr_scale=cfg["arm2_aux_lr_scale"],
        validation_mode="existing",
        train_scope="full",
        grad_clip=1.0,
        patience=cfg["patience"],
        seed=cfg["seed"] + 1,
        device=cfg["device"],
        cpu_threads=cfg["cpu_threads"],
        resume=True,
        checkpoint_every=cfg["checkpoint_every"],
        build=cfg["build"],
    )
    r.train(arm_cfg)
    if cfg["build"]:
        r.build_candidate(arm_cfg)
    return out_dir / "best.pt", out_dir / "student_int8.bin"


def train_arm3(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Train Arm 3: Deep Positional Regularization with elevated auxiliary weight."""
    out_dir = Path(cfg["out_root"]) / "arm3_deep_regularized"
    surprise_data = ensure_surprise_data(cfg)
    print(f"\n{'=' * 70}\n[TRAIN NNUE] STARTING ARM 3: Deep Positional Regularization\n{'=' * 70}")
    arm_cfg = dict(
        r.CONFIG,
        data=surprise_data,
        out_dir=str(out_dir),
        architecture=cfg["architecture"],
        hidden=cfg["hidden"],
        init_from=cfg["init_from"],
        init_architecture=cfg["init_architecture"],
        init_hidden=cfg["init_hidden"],
        from_scratch=False,
        qat=cfg["qat"],
        mirror_h=cfg["mirror_h"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        min_lr=cfg["min_lr"],
        trunk_lr_scale=cfg["trunk_lr_scale"],
        warmup_epochs=cfg["warmup_epochs"],
        weight_decay=cfg["weight_decay"],
        weight_decay_schedule="constant",
        min_weight_decay=cfg["weight_decay"],
        weight_boosts=[],
        max_sample_weight=30.0,
        policy_weight=1.0,
        value_weight=1.0,
        aux_policy_enabled=True,
        aux_policy_temperature=cfg["arm3_aux_temp"],
        aux_policy_weight=cfg["arm3_aux_weight"],
        aux_policy_lr_scale=cfg["arm3_aux_lr_scale"],
        validation_mode="existing",
        train_scope="full",
        grad_clip=1.0,
        patience=cfg["patience"],
        seed=cfg["seed"] + 2,
        device=cfg["device"],
        cpu_threads=cfg["cpu_threads"],
        resume=True,
        checkpoint_every=cfg["checkpoint_every"],
        build=cfg["build"],
    )
    r.train(arm_cfg)
    if cfg["build"]:
        r.build_candidate(arm_cfg)
    return out_dir / "best.pt", out_dir / "student_int8.bin"


def unify_soup(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Unify trained arms into a convex Model Soup champion candidate."""
    root = Path(cfg["out_root"])
    out_dir = root / "soup_tri_champion"
    arm1_pt = root / "arm1_sprint_aux" / "best.pt"
    arm2_pt = root / "arm2_surprise_aux" / "best.pt"
    arm3_pt = root / "arm3_deep_regularized" / "best.pt"

    for path in (arm1_pt, arm2_pt, arm3_pt):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot unify model soup; component missing: {path}")

    weights = cfg["soup_weights"]
    specs = [
        (arm1_pt, float(weights[0])),
        (arm2_pt, float(weights[1])),
        (arm3_pt, float(weights[2])),
    ]

    print(f"\n{'=' * 70}\n[TRAIN NNUE] UNIFYING MODEL SOUP: Convex Ensemble\n{'=' * 70}")
    int8_path, exe_path = create_soup(
        model_specs=specs,
        out_dir=out_dir,
        architecture=cfg["architecture"],
        hidden=cfg["hidden"],
        qat=cfg["qat"],
        build=cfg["build"],
    )
    print(f"\n[TRAIN NNUE] Model soup champion unified successfully in: {out_dir}")
    return int8_path, exe_path


def run_pipeline(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Execute complete 3-arm training pipeline followed by soup unification."""
    print(f"\n{'=' * 70}\n[TRAIN NNUE] EXECUTING COMPLETE 3-ARM TRAINING AND SOUP PIPELINE\n{'=' * 70}")
    train_arm1(cfg)
    train_arm2(cfg)
    train_arm3(cfg)
    return unify_soup(cfg)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Zquoridor NNUE Multi-Arm Training & Soup Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["arm1", "arm2", "arm3", "soup", "unify", "all", "pipeline"],
        default=CONFIG["mode"],
        help="Training mode: arm1 (sprint), arm2 (surprise), arm3 (deep), soup/unify (ensemble), or all/pipeline.",
    )
    parser.add_argument("--data", type=str, default=CONFIG["data"], help="Path to base training dataset directory")
    parser.add_argument("--surprise-data", type=str, default=CONFIG["surprise_data"], help="Path to surprise-weighted dataset")
    parser.add_argument("--out-root", type=str, default=CONFIG["out_root"], help="Root output directory for all arms and soup")
    parser.add_argument("--init-from", type=str, default=CONFIG["init_from"], help="Path to baseline float32 weights for warm start")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs per arm")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Training batch size")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=CONFIG["min_lr"], help="Minimum learning rate for cosine annealing")
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--no-build", action="store_true", help="Skip compilation of candidate binaries")
    args = parser.parse_args(argv)

    cfg = copy.deepcopy(CONFIG)
    cfg["mode"] = args.mode
    cfg["data"] = args.data
    cfg["surprise_data"] = args.surprise_data
    cfg["out_root"] = args.out_root
    cfg["init_from"] = args.init_from
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["lr"] = args.lr
    cfg["min_lr"] = args.min_lr
    cfg["device"] = args.device
    if args.no_build:
        cfg["build"] = False

    mode = cfg["mode"]
    if mode == "arm1":
        train_arm1(cfg)
    elif mode == "arm2":
        train_arm2(cfg)
    elif mode == "arm3":
        train_arm3(cfg)
    elif mode in ("soup", "unify"):
        unify_soup(cfg)
    elif mode in ("all", "pipeline"):
        run_pipeline(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return 0


# =============================================================================
# Legacy NNUE Architecture Constants & Compat Stubs (for train_teacher_policy & tests)
# =============================================================================
QA_DEFAULT = 255
QB_DEFAULT = 64
INT8_MAX = 127
INT16_MAX = 32767
DIST_BUCKETS = 21
WALLS_PER_PLAYER = 10
WALLS_LEFT_BUCKETS = 11
NUM_FEATURES = 354
HIDDEN = 256
POLICY_OUT = 209


def screlu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0).square()


def clipped_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


class QuoridorNNUE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(NUM_FEATURES, HIDDEN)
        self.value1_wl = torch.nn.Linear(HIDDEN, 32)
        self.value2_wl = torch.nn.Linear(32, 1)
        self.policy = torch.nn.Linear(HIDDEN, POLICY_OUT)

    def forward(self, x: torch.Tensor):
        acc = self.fc1(x)
        a = screlu(acc)
        h_wl = clipped_relu(self.value1_wl(a))
        value_wl = self.value2_wl(h_wl).squeeze(-1)
        policy_logits = self.policy(a)
        return value_wl, policy_logits


def _load_raw_weights(path):
    head_floats = HIDDEN * 32 + 32 + 32 + 1
    base_floats = NUM_FEATURES * HIDDEN + HIDDEN + head_floats
    tail_floats = POLICY_OUT * HIDDEN + POLICY_OUT
    expected_new = (base_floats + tail_floats) * 4
    expected_old = (base_floats + head_floats + tail_floats) * 4

    import os
    file_size = os.path.getsize(path)
    if file_size not in (expected_new, expected_old):
        raise ValueError(f"[init-from] '{path}' tem tamanho inesperado ({file_size} bytes)")
    is_old_format = (file_size == expected_old)

    with open(path, "rb") as f:
        w1 = np.fromfile(f, dtype="<f4", count=NUM_FEATURES * HIDDEN).reshape(NUM_FEATURES, HIDDEN)
        b1 = np.fromfile(f, dtype="<f4", count=HIDDEN)

        def read_head():
            wv1 = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
            bv1 = np.fromfile(f, dtype="<f4", count=32)
            wv2 = np.fromfile(f, dtype="<f4", count=32)
            bv2 = np.fromfile(f, dtype="<f4", count=1)
            return wv1, bv1, wv2, bv2

        wv1_wl, bv1_wl, wv2_wl, bv2_wl = read_head()
        if is_old_format:
            read_head()
        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
        bp = np.fromfile(f, dtype="<f4", count=POLICY_OUT)
    return dict(w1=w1, b1=b1, wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl, wp=wp, bp=bp)


def _load_into_model(model: QuoridorNNUE, raw: dict):
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(raw["w1"].T.copy()))
        model.fc1.bias.copy_(torch.from_numpy(raw["b1"]))
        model.value1_wl.weight.copy_(torch.from_numpy(raw["wv1_wl"].T.copy()))
        model.value1_wl.bias.copy_(torch.from_numpy(raw["bv1_wl"]))
        model.value2_wl.weight.copy_(torch.from_numpy(raw["wv2_wl"].reshape(1, -1)))
        model.value2_wl.bias.copy_(torch.from_numpy(raw["bv2_wl"]))
        model.policy.weight.copy_(torch.from_numpy(raw["wp"]))
        model.policy.bias.copy_(torch.from_numpy(raw["bp"]))


if __name__ == "__main__":
    raise SystemExit(main())

