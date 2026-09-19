#!/usr/bin/env python3
"""Run orthogonal race512-search10 fine-tune hyperparameter ablations."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/teaching/historical2m-search10-conservative/dataset.npz"
DEFAULT_INIT = ROOT / "results/experiments/race512-search10-ft-s20260917/student.bin"

VARIANTS = {
    "value05": {"value_weight": 0.5, "trunk_lr_scale": 0.1},
    "value20": {"value_weight": 2.0, "trunk_lr_scale": 0.1},
    "value40": {"value_weight": 4.0, "trunk_lr_scale": 0.1},
    "trunk03": {"value_weight": 1.0, "trunk_lr_scale": 0.3},
    "trunk10": {"value_weight": 1.0, "trunk_lr_scale": 1.0},
}


def command(label: str, data: Path, init: Path, device: str) -> list[str]:
    v = VARIANTS[label]
    out = ROOT / "results/experiments" / f"race512-search10-{label}-s20260919"
    return [
        sys.executable, str(ROOT / "training/run_experiment.py"),
        "--data", str(data),
        "--out-dir", str(out),
        "--architecture", "race",
        "--hidden", "512",
        "--init-from", str(init),
        "--init-architecture", "race",
        "--init-hidden", "512",
        "--no-from-scratch",
        "--qat",
        "--epochs", "40",
        "--batch-size", "4096",
        "--lr", "3e-5",
        "--schedule", "cosine",
        "--warmup-epochs", "2",
        "--min-lr", "3e-6",
        "--trunk-lr-scale", str(v["trunk_lr_scale"]),
        "--weight-decay", "1e-5",
        "--policy-weight", "1.0",
        "--value-weight", str(v["value_weight"]),
        "--train-scope", "full",
        "--grad-clip", "1.0",
        "--patience", "14",
        "--seed", "20260929",
        "--device", device,
        "--cpu-threads", "4",
        "--build",
        "--no-benchmark",
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--init", default=str(DEFAULT_INIT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", choices=sorted(VARIANTS))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    labels = [a.only] if a.only else list(VARIANTS)
    payload = []
    for label in labels:
        cmd = command(label, Path(a.data), Path(a.init), a.device)
        payload.append({"label": label, "settings": VARIANTS[label], "command": cmd})
        if not a.dry_run:
            if not Path(a.data).is_file():
                raise SystemExit(f"dataset missing: {a.data}")
            if not Path(a.init).is_file():
                raise SystemExit(f"initial weights missing: {a.init}")
            subprocess.run(cmd, cwd=ROOT, check=True)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
