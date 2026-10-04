#!/usr/bin/env python3
"""Execute Stage A evaluation of auxiliary policy head against frozen baseline.

Trains A0 (control) and A1 (auxiliary soft policy) from frozen production weights,
compiles candidate binaries, and screens both candidates against frozen V3.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

import run_experiment as r
from tools import match_finalists
from tools.run_complete_promotion_battery import compute_metrics_from_games

CONFIG: Dict[str, Any] = {
    "data": str(ROOT / "results" / "experiments" / "production-central-finetune-20261002" / "mixed_dataset"),
    "out_root": str(ROOT / "results" / "experiments" / "aux_policy_stage_a"),
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
    "warmup_epochs": 0,
    "weight_decay": 1e-05,
    "seed": 20261004,
    "qat": True,
    "mirror_h": True,
    "checkpoint_every": 5,
    "patience": 0,
    "device": "cuda",
    "cpu_threads": 4,
    "aux_policy_temperature": 2.0,
    "aux_policy_weight": 0.15,
    "aux_policy_lr_scale": 5.0,
    "screening_pairs": 20,
    "screening_move_time_ms": 200,
    "screening_workers": 4,
    "normal_openings": str(ROOT / "tools" / "external" / "openings_normal_screen_100.jsonl"),
    "centerrush_openings": str(ROOT / "tools" / "external" / "openings_center_rush_sound_5k.jsonl"),
    "baseline_exe": str(ROOT / "bin" / "zquoridor.exe"),
    "baseline_nnue": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "run_a0": True,
    "run_a1": True,
    "run_screening": True,
    "dry_run": False,
}


def parse_config(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse configuration with CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def _ensure_candidate_trained(cfg: Dict[str, Any], name: str, aux_enabled: bool) -> Tuple[Path, Path]:
    """Ensure that the candidate is trained and compiled, returning exe and int8 paths."""
    out_dir = Path(cfg["out_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "train_report.json"
    int8_file = out_dir / "student_int8.bin"
    suffix = ".exe" if os.name == "nt" else ""
    exe_file = out_dir / f"zquoridor{suffix}"

    train_needed = True
    if report_file.is_file() and int8_file.is_file():
        try:
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
            if report_data.get("epochs") == cfg["epochs"] and report_data.get("training_status") == "complete":
                train_needed = False
                print(f"[{name}] Training already complete with {cfg['epochs']} epochs.")
        except Exception:
            train_needed = True

    if train_needed:
        print(f"\n{'=' * 70}")
        print(f"STARTING TRAINING: {name} (aux_enabled={aux_enabled})")
        print(f"Epochs: {cfg['epochs']} | Batch size: {cfg['batch_size']} | LR: {cfg['lr']} -> {cfg['min_lr']}")
        print(f"{'=' * 70}\n", flush=True)

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
            aux_policy_enabled=aux_enabled,
            aux_policy_temperature=cfg["aux_policy_temperature"],
            aux_policy_weight=cfg["aux_policy_weight"],
            aux_policy_lr_scale=cfg["aux_policy_lr_scale"],
            validation_mode="existing",
            train_scope="full",
            grad_clip=1.0,
            patience=cfg["patience"],
            seed=cfg["seed"],
            device=cfg["device"],
            cpu_threads=cfg["cpu_threads"],
            resume=True,
            checkpoint_every=cfg["checkpoint_every"],
            build=True,
        )
        r.train(arm_cfg)
        r.build_candidate(arm_cfg)

    if not exe_file.is_file():
        arm_cfg = dict(
            r.CONFIG,
            out_dir=str(out_dir),
            architecture=cfg["architecture"],
            hidden=cfg["hidden"],
        )
        r.build_candidate(arm_cfg)

    return exe_file, int8_file


def _screen_candidate(
    cand_name: str,
    cand_exe: Path,
    cand_nnue: Path,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run screening matches against frozen V3 on normal and Center Rush openings."""
    out_dir = Path(cfg["out_root"]) / "screening" / cand_name
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_dir = out_dir / "normal"
    cr_dir = out_dir / "centerrush"

    print(f"\n--- Screening {cand_name} on Normal Openings ---", flush=True)
    normal_cfg = {
        "engine1_name": cand_name,
        "engine1_executable": str(cand_exe),
        "engine1_nnue": str(cand_nnue),
        "engine1_args": "",
        "engine2_name": "frozen_v3",
        "engine2_executable": cfg["baseline_exe"],
        "engine2_nnue": cfg["baseline_nnue"],
        "engine2_args": "",
        "pairs": cfg["screening_pairs"],
        "move_time_ms": cfg["screening_move_time_ms"],
        "workers": cfg["screening_workers"],
        "seed": cfg["seed"],
        "openings": cfg["normal_openings"],
        "output": str(normal_dir),
        "bootstrap": 20000,
    }
    normal_res = match_finalists.run(normal_cfg)
    normal_metrics = compute_metrics_from_games(normal_dir / "games.jsonl")

    print(f"\n--- Screening {cand_name} on Center Rush Openings ---", flush=True)
    cr_cfg = {
        "engine1_name": cand_name,
        "engine1_executable": str(cand_exe),
        "engine1_nnue": str(cand_nnue),
        "engine1_args": "",
        "engine2_name": "frozen_v3",
        "engine2_executable": cfg["baseline_exe"],
        "engine2_nnue": cfg["baseline_nnue"],
        "engine2_args": "",
        "pairs": cfg["screening_pairs"],
        "move_time_ms": cfg["screening_move_time_ms"],
        "workers": cfg["screening_workers"],
        "seed": cfg["seed"] + 1,
        "openings": cfg["centerrush_openings"],
        "output": str(cr_dir),
        "bootstrap": 20000,
    }
    cr_res = match_finalists.run(cr_cfg)
    cr_metrics = compute_metrics_from_games(cr_dir / "games.jsonl")

    # Combine games for overall score
    all_rows = []
    for gfile in (normal_dir / "games.jsonl", cr_dir / "games.jsonl"):
        if gfile.is_file():
            with gfile.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_rows.append(json.loads(line))

    from tools.external import local_arena
    overall_summary = local_arena.summarize_pairs(all_rows, bootstrap=20000, seed=cfg["seed"])

    cand_metrics = normal_metrics.get(cand_name, {})
    cr_cand_metrics = cr_metrics.get(cand_name, {})
    total_moves = cand_metrics.get("move_count", 0) + cr_cand_metrics.get("move_count", 0)
    avg_depth = 0.0
    avg_nps = 0.0
    if total_moves > 0:
        avg_depth = (
            cand_metrics.get("mean_depth", 0.0) * cand_metrics.get("move_count", 0)
            + cr_cand_metrics.get("mean_depth", 0.0) * cr_cand_metrics.get("move_count", 0)
        ) / total_moves
        avg_nps = (
            cand_metrics.get("mean_nps", 0.0) * cand_metrics.get("move_count", 0)
            + cr_cand_metrics.get("mean_nps", 0.0) * cr_cand_metrics.get("move_count", 0)
        ) / total_moves

    return {
        "candidate": cand_name,
        "normal": normal_res,
        "normal_metrics": normal_metrics,
        "centerrush": cr_res,
        "centerrush_metrics": cr_metrics,
        "overall": overall_summary,
        "avg_depth": avg_depth,
        "avg_nps": avg_nps,
    }


def main(argv: Optional[List[str]] = None) -> int:
    config = parse_config(argv)
    if config["dry_run"]:
        print(json.dumps(config, indent=2))
        return 0

    out_root = Path(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    candidates: Dict[str, Tuple[Path, Path]] = {}

    if config["run_a0"]:
        exe_a0, nnue_a0 = _ensure_candidate_trained(config, "A0_control", aux_enabled=False)
        candidates["A0_control"] = (exe_a0, nnue_a0)

    if config["run_a1"]:
        exe_a1, nnue_a1 = _ensure_candidate_trained(config, "A1_aux_t20_b15", aux_enabled=True)
        candidates["A1_aux_t20_b15"] = (exe_a1, nnue_a1)

    screening_results = {}
    if config["run_screening"]:
        for name, (exe, nnue) in candidates.items():
            res = _screen_candidate(name, exe, nnue, config)
            screening_results[name] = res

        summary_path = out_root / "stage_a_summary.json"
        summary_path.write_text(json.dumps(screening_results, indent=2) + "\n", encoding="utf-8")
        print(f"\nStage A summary written to: {summary_path}")

    # Print final markdown table
    print("\n" + "=" * 100)
    print("STAGE A SCREENING RESULTS TABLE")
    print("=" * 100)
    print("| Candidate | Validation mode | T | Aux weight | Epoch | Val loss | vs V3 | Normal | Center Rush | NPS | Depth | Decision |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for name in ("A0_control", "A1_aux_t20_b15"):
        rep_path = out_root / name / "train_report.json"
        val_loss_str = "-"
        if rep_path.is_file():
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            val_loss = rep.get("best_val_loss")
            val_loss_str = f"{val_loss:.5f}" if val_loss else "-"

        t_val = "2.0" if name.startswith("A1") else "-"
        w_val = "0.15" if name.startswith("A1") else "-"

        sc = screening_results.get(name)
        if sc:
            ov = sc["overall"]
            ov_score = ov.get("score_pct", 50.0)
            ov_elo = ov.get("elo", 0.0)
            ov_ci = ov.get("paired_bootstrap_95", {})
            ov_low = ov_ci.get("elo_low", 0.0)
            ov_high = ov_ci.get("elo_high", 0.0)
            vs_v3_str = f"{ov_score:.1f}% ({ov_elo:+.1f} [{ov_low:+.1f}, {ov_high:+.1f}])"

            n_score = sc["normal"].get("summary", {}).get("score_pct", 50.0)
            cr_score = sc["centerrush"].get("summary", {}).get("score_pct", 50.0)
            n_str = f"{n_score:.1f}%"
            cr_str = f"{cr_score:.1f}%"

            nps_str = f"{sc['avg_nps']:.0f}"
            depth_str = f"{sc['avg_depth']:.1f}"

            if name == "A0_control":
                decision = "Control baseline"
            else:
                decision = "Promising (proceed to Stage B)" if ov_score >= 50.0 else "Rejected (A1 worse)"
        else:
            vs_v3_str = "-"
            n_str = "-"
            cr_str = "-"
            nps_str = "-"
            depth_str = "-"
            decision = "-"

        print(f"| {name} | existing | {t_val} | {w_val} | {config['epochs']} | {val_loss_str} | {vs_v3_str} | {n_str} | {cr_str} | {nps_str} | {depth_str} | {decision} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
