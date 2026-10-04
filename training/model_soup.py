#!/usr/bin/env python3
"""Create model soups by convex combination of compatible NNUE student checkpoints.

Takes two or more model checkpoints (PyTorch .pt or exported float .bin) with
matching architecture, blends their weights according to specified weights,
clips parameters within safe numerical bounds, exports float and int8 representations,
and compiles a native evaluation binary.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from student_model import Student, export
from run_experiment import build_candidate, CONFIG as RUN_CONFIG

CONFIG: Dict[str, Any] = {
    "models": [],  # List of tuples: (path_str, float_weight)
    "out_dir": str(ROOT / "results" / "experiments" / "aux_policy_stage_a" / "Soup_Tri"),
    "architecture": "multipath_phase_bucketed",
    "hidden": 512,
    "qat": True,
    "build": True,
}


def load_model_weights(
    model_path: Path,
    architecture: str,
    hidden: int,
    qat: bool = True,
) -> Student:
    """Load student model from either .pt checkpoint or .bin float weights."""
    model = Student(architecture, hidden, qat=qat)
    if model_path.suffix == ".bin":
        model.load_float(model_path)
    elif model_path.suffix == ".pt":
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        # Filter out aux_policy if present and student doesn't have it, or enable it
        has_aux = any(k.startswith("aux_policy") for k in state_dict.keys())
        if has_aux:
            model.enable_aux_policy()
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")
    return model


def create_soup(
    model_specs: List[Tuple[Path, float]],
    out_dir: Path,
    architecture: str = "multipath_phase_bucketed",
    hidden: int = 512,
    qat: bool = True,
    build: bool = True,
) -> Tuple[Path, Path]:
    """Blend multiple student models into a single soup model and export it."""
    if not model_specs:
        raise ValueError("At least one model specification is required to create a soup.")

    total_weight = sum(w for _, w in model_specs)
    if total_weight <= 0:
        raise ValueError("Sum of model weights must be positive.")
    normalized_specs = [(p, w / total_weight) for p, w in model_specs]

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating model soup in: {out_dir}")
    print(f"Architecture: {architecture}, Hidden: {hidden}")
    print("Components:")
    for path, weight in normalized_specs:
        print(f"  - {path} (weight: {weight:.4f})")

    # Load first model to initialize blended state
    first_path, first_w = normalized_specs[0]
    first_model = load_model_weights(first_path, architecture, hidden, qat=qat)
    base_state = {k: v.clone() * first_w for k, v in first_model.state_dict().items()}

    # Accumulate remaining models
    for path, weight in normalized_specs[1:]:
        model = load_model_weights(path, architecture, hidden, qat=qat)
        model_state = model.state_dict()
        for k in base_state:
            if k in model_state:
                base_state[k] += model_state[k] * weight
            else:
                raise KeyError(f"Parameter '{k}' missing from model at {path}")

    # Build soup model
    soup_model = Student(architecture, hidden, qat=qat)
    if any(k.startswith("aux_policy") for k in base_state):
        soup_model.enable_aux_policy()
    soup_model.load_state_dict(base_state)
    soup_model.clip_weights()

    # Export float and int8
    out_student_bin = out_dir / "student.bin"
    manifest = export(soup_model, out_student_bin)
    out_int8_bin = out_dir / "student_int8.bin"

    manifest_file = out_dir / "soup_manifest.json"
    soup_metadata = {
        "schema": "zquoridor.soup.v1",
        "components": [{"path": str(p), "weight": w} for p, w in normalized_specs],
        "manifest": manifest,
    }
    manifest_file.write_text(json.dumps(soup_metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Exported float weights: {out_student_bin}")
    print(f"Exported int8 weights:  {out_int8_bin}")

    if build:
        build_cfg = dict(
            RUN_CONFIG,
            out_dir=str(out_dir),
            architecture=architecture,
            hidden=hidden,
        )
        build_candidate(build_cfg)
        print(f"Candidate binary built in: {out_dir / 'zquoridor.exe'}")

    return out_int8_bin, out_dir / "zquoridor.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create NNUE model soup.")
    parser.add_argument("--models", nargs="+", help="Pairs of <model_path> <weight>, e.g. path1 0.5 path2 0.5")
    parser.add_argument("--out-dir", type=str, default=CONFIG["out_dir"], help="Output directory")
    parser.add_argument("--architecture", type=str, default=CONFIG["architecture"], help="Architecture name")
    parser.add_argument("--hidden", type=int, default=CONFIG["hidden"], help="Hidden width")
    parser.add_argument("--no-build", action="store_true", help="Skip compilation of native candidate")
    args = parser.parse_args(argv)

    cfg = copy.deepcopy(CONFIG)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.architecture:
        cfg["architecture"] = args.architecture
    if args.hidden:
        cfg["hidden"] = args.hidden
    if args.no_build:
        cfg["build"] = False

    if args.models:
        if len(args.models) % 2 != 0:
            raise ValueError("--models must be provided in pairs: <path> <weight>")
        specs = []
        for i in range(0, len(args.models), 2):
            specs.append((Path(args.models[i]), float(args.models[i + 1])))
        cfg["models"] = specs
    elif not cfg["models"]:
        raise ValueError("No models specified in CONFIG or command-line.")

    model_specs = [(Path(p), float(w)) for p, w in cfg["models"]]
    create_soup(
        model_specs=model_specs,
        out_dir=Path(cfg["out_dir"]),
        architecture=cfg["architecture"],
        hidden=cfg["hidden"],
        qat=cfg["qat"],
        build=cfg["build"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
