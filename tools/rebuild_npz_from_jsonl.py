#!/usr/bin/env python3
"""Rebuild and synchronize dense NumPy target archives from canonical JSONL records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external.local_arena import Referee
from tools.generate_wall_drops import compute_shortest_paths
from training.teachers.targets import POLICY_DIM, normalize_policy

CONFIG = {
    "directory": "",
    "backup": True,
    "verify_only": False,
}


def rebuild_targets_npz(dataset_dir: Path | str, backup: bool = True, verify_only: bool = False) -> dict:
    """Reconstruct targets.npz from positions.jsonl and targets.jsonl with full validation."""
    dataset_path = Path(dataset_dir).resolve()
    positions_file = dataset_path / "positions.jsonl"
    targets_file = dataset_path / "targets.jsonl"
    npz_file = dataset_path / "targets.npz"
    manifest_file = dataset_path / "manifest.json"

    if not positions_file.is_file():
        raise FileNotFoundError(f"Missing positions file: {positions_file}")
    if not targets_file.is_file():
        raise FileNotFoundError(f"Missing targets file: {targets_file}")

    print(f"Reading records from {dataset_path}...", flush=True)

    positions: list[dict] = []
    labels: list[dict] = []
    policy_matrix: list[list[float]] = []
    value_list: list[float] = []
    aux_features: list[list[int]] = []
    sample_ids: list[str] = []

    with positions_file.open("r", encoding="utf-8") as f_p, targets_file.open("r", encoding="utf-8") as f_t:
        for line_idx, (lp, lt) in enumerate(zip(f_p, f_t), start=1):
            lp = lp.strip()
            lt = lt.strip()
            if not lp or not lt:
                continue

            try:
                p_rec = json.loads(lp)
                l_rec = json.loads(lt)
            except Exception as exc:
                print(f"Warning: line {line_idx} failed JSON parsing: {exc}", file=sys.stderr)
                continue

            # Verify sample ID consistency
            p_id = p_rec.get("id", "")
            l_id = l_rec.get("id", "")
            if p_id != l_id or not p_id:
                print(f"Warning: line {line_idx} ID mismatch: pos={p_id} vs tgt={l_id}", file=sys.stderr)
                continue

            target = l_rec.get("target", l_rec)
            raw_policy = target.get("policy")
            if not raw_policy or len(raw_policy) != POLICY_DIM:
                print(f"Warning: line {line_idx} invalid policy length: {len(raw_policy) if raw_policy else 0}", file=sys.stderr)
                continue

            try:
                policy = normalize_policy(raw_policy)
            except Exception as exc:
                print(f"Warning: line {line_idx} policy normalization failed: {exc}", file=sys.stderr)
                continue

            raw_val = target.get("value")
            if raw_val is None:
                print(f"Warning: line {line_idx} missing value", file=sys.stderr)
                continue
            value = float(raw_val)
            if not math.isfinite(value) or abs(value) > 1.0:
                print(f"Warning: line {line_idx} non-finite or out-of-range value: {value}", file=sys.stderr)
                continue

            aux_meta = l_rec.get("metadata", {}).get("aux")
            if not aux_meta or len(aux_meta) != 6:
                ref = Referee()
                for mv in p_rec.get("history", []):
                    ref.apply(mv)
                own_d, opp_d = compute_shortest_paths(ref, ref.side_to_move)
                aux_meta = [
                    own_d,
                    opp_d,
                    ref.walls_left[ref.side_to_move],
                    ref.walls_left[1 - ref.side_to_move],
                    len(p_rec.get("history", [])),
                    ref.side_to_move,
                ]

            positions.append(p_rec)
            labels.append(l_rec)
            policy_matrix.append(policy)
            value_list.append(value)
            aux_features.append(aux_meta)
            sample_ids.append(p_id)

    total_valid = len(positions)
    print(f"Parsed and validated {total_valid} synchronized records from JSONL.", flush=True)
    if total_valid == 0:
        raise ValueError(f"No valid synchronized records found in {dataset_path}")

    # Policy sanity check: ensure zero all-zero rows
    policy_arr = np.asarray(policy_matrix, dtype=np.float32)
    sums = policy_arr.sum(axis=1)
    zero_rows = int(np.sum(sums < 0.99))
    if zero_rows > 0:
        raise ValueError(f"CRITICAL: Found {zero_rows} rows with zero policy mass!")

    value_arr = np.asarray(value_list, dtype=np.float32)
    aux_arr = np.asarray(aux_features, dtype=np.int32)
    sample_ids_arr = np.asarray(sample_ids)

    stats = {
        "dataset_directory": str(dataset_path),
        "total_records": total_valid,
        "policy_shape": list(policy_arr.shape),
        "zero_policy_rows": zero_rows,
        "value_mean": float(np.mean(value_arr)),
        "value_min": float(np.min(value_arr)),
        "value_max": float(np.max(value_arr)),
        "aux_shape": list(aux_arr.shape),
    }

    if verify_only:
        print(f"Verification only mode: stats = {stats}", flush=True)
        return stats

    # Backup existing npz file if present
    if npz_file.exists() and backup:
        timestamp = int(time.time())
        backup_path = dataset_path / f"targets.npz.bak_{timestamp}"
        shutil.copy2(npz_file, backup_path)
        print(f"Created backup of existing archive at: {backup_path}", flush=True)
        stats["backup_file"] = str(backup_path)

    # Write atomic temporary archive then rename
    tmp_npz = dataset_path / f"targets_tmp_{int(time.time())}_{abs(hash(str(dataset_path)))}.npz"
    np.savez_compressed(
        tmp_npz,
        policy=policy_arr,
        value=value_arr,
        aux=aux_arr,
        sample_ids=sample_ids_arr,
    )
    tmp_npz.replace(npz_file)
    print(f"Successfully saved {total_valid} records to: {npz_file}", flush=True)

    # Update manifest
    manifest_data = {}
    if manifest_file.exists():
        try:
            with manifest_file.open("r", encoding="utf-8") as f_man:
                manifest_data = json.load(f_man)
        except Exception:
            pass
    manifest_data["generated"] = total_valid
    manifest_data["positions_file"] = str(positions_file)
    manifest_data["targets_file"] = str(targets_file)
    manifest_data["npz_file"] = str(npz_file)
    manifest_data["rebuilt_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with manifest_file.open("w", encoding="utf-8") as f_man:
        json.dump(manifest_data, f_man, indent=2)

    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build command line interface for targets.npz rebuilder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", dest="directory", type=str, required=True,
                        help="Path to dataset directory containing positions.jsonl and targets.jsonl")
    parser.add_argument("--no-backup", dest="backup", action="store_false", default=True,
                        help="Disable timestamped backup of existing targets.npz")
    parser.add_argument("--verify-only", action="store_true", default=False,
                        help="Only verify data integrity without writing new files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point with dictionary configuration and command line overrides."""
    parser = build_parser()
    args = parser.parse_args(argv)

    active_config = dict(CONFIG)
    for key, val in vars(args).items():
        if val is not None:
            active_config[key] = val

    stats = rebuild_targets_npz(
        dataset_dir=active_config["directory"],
        backup=active_config["backup"],
        verify_only=active_config["verify_only"],
    )
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
