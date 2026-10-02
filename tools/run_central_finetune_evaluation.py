#!/usr/bin/env python3
"""Run paired center-rush evaluation against frozen production and Claustrophobia.

The candidate and frozen main use the same frozen native executable so the
match isolates NNUE weights. Both opponents use the same book, seed, and fixed
200 ms move clock. Each ``pairs`` entry is one opening played with colors
swapped (two games).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import match_finalists, run_benchmark
from tools.external import local_arena


DEFAULT_FROZEN_DIR = ROOT / "results/experiments/frozen-production-20261002"
DEFAULT_BOOK = ROOT / "tools/external/openings_center_rush_sound_5k.jsonl"
DEFAULT_OUTPUT = ROOT / "results/benchmarks/central-finetune"

# Edit CONFIG for a standard local invocation; CLI flags override each value.
CONFIG = {
    "candidate_name": None,
    "candidate_nnue": None,
    "candidate_architecture": None,
    "frozen_dir": str(DEFAULT_FROZEN_DIR),
    "pairs": 200,
    "move_time_ms": 200,
    "workers": 1,
    "seed": 20261002,
    "openings": str(DEFAULT_BOOK),
    "output": str(DEFAULT_OUTPUT),
    "claustrophobia_device": "cpu",
    "claustrophobia_max_sims": 4096,
    "bootstrap": 20000,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not slug:
        raise ValueError("candidate_name must contain a letter or number")
    return slug[:64]


def _candidate_architecture_path(nnue: Path) -> Path:
    if nnue.name.endswith("_int8.bin"):
        return nnue.with_name(nnue.name[:-len("_int8.bin")] + ".architecture.json")
    return nnue.with_suffix(".architecture.json")


def _validate_architecture(metadata: dict, *, label: str, nnue_sha: str | None = None) -> None:
    expected = {"features": 504, "hidden": 512, "value_buckets": 6, "value_depth": 2}
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items()
                  if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"{label} architecture must be 504/512/6/2; got {mismatches}")
    if nnue_sha is not None and metadata.get("int8_sha256", "").lower() != nnue_sha.lower():
        raise ValueError(f"{label} architecture int8_sha256 does not match its NNUE file")


def _load_frozen_artifacts(frozen_dir: Path) -> tuple[Path, Path, dict]:
    manifest_path = frozen_dir / "frozen_baseline_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen baseline manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "zquoridor.frozen_finetune_baseline.v1":
        raise ValueError(f"unsupported frozen baseline manifest: {manifest_path}")
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, dict):
        raise ValueError("frozen baseline manifest has no artifacts map")
    paths = {}
    for name in ("zquoridor.exe", "production_nnue_int8.bin"):
        row = artifact_rows.get(name)
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError(f"frozen baseline manifest lacks {name}")
        path = (frozen_dir / row["path"]).resolve()
        try:
            path.relative_to(frozen_dir.resolve())
        except ValueError as error:
            raise ValueError(f"frozen artifact escapes frozen_dir: {name}") from error
        if not path.is_file():
            raise FileNotFoundError(f"frozen baseline artifact not found: {path}")
        if path.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError(f"frozen baseline size mismatch: {name}")
        if _sha256(path).lower() != str(row["sha256"]).lower():
            raise ValueError(f"frozen baseline SHA-256 mismatch: {name}")
        paths[name] = path
    _validate_architecture(
        manifest.get("architecture", {}), label="frozen baseline",
        nnue_sha=str(artifact_rows["production_nnue_int8.bin"]["sha256"]),
    )
    return paths["zquoridor.exe"], paths["production_nnue_int8.bin"], manifest


def _require_complete(summary: dict, *, label: str, pairs: int) -> None:
    failed = int(summary.get("failed_games", 0))
    complete = int(summary.get("complete_pairs", -1))
    if failed or complete != pairs:
        raise RuntimeError(
            f"{label} evaluation incomplete: failed_games={failed}, "
            f"complete_pairs={complete}, expected_pairs={pairs}; inspect its games.jsonl"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs: dict[str, object] = {"default": None}
        if isinstance(value, int):
            kwargs["type"] = int
        elif value is not None:
            kwargs["type"] = str
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    return parser


def resolve_config(argv: list[str] | None = None) -> dict:
    config = dict(CONFIG)
    config.update({key: value for key, value in vars(build_parser().parse_args(argv)).items()
                   if value is not None})
    if not config["candidate_name"] or not config["candidate_nnue"]:
        raise ValueError("--candidate-name and --candidate-nnue are required")
    if config["candidate_architecture"] is not None:
        path = Path(config["candidate_architecture"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"candidate_architecture not found: {path}")
        config["candidate_architecture"] = str(path)
    for key in ("candidate_nnue", "openings", "frozen_dir"):
        path = Path(config[key]).resolve()
        if key != "frozen_dir" and not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
        if key == "frozen_dir" and not path.is_dir():
            raise FileNotFoundError(f"frozen_dir not found: {path}")
        config[key] = str(path)
    if int(config["pairs"]) <= 0 or not 1 <= int(config["workers"]) <= 4:
        raise ValueError("pairs must be positive and workers must be between 1 and 4")
    if int(config["move_time_ms"]) != 200:
        raise ValueError("the central fine-tune gate is fixed at 200 ms per move")
    if int(config["bootstrap"]) < 1000:
        raise ValueError("bootstrap must be at least 1000")
    return config


def run(config: dict) -> dict:
    candidate_nnue = Path(config["candidate_nnue"]).resolve()
    frozen_dir = Path(config["frozen_dir"]).resolve()
    frozen_exe, frozen_nnue, frozen_manifest = _load_frozen_artifacts(frozen_dir)
    candidate_sha = _sha256(candidate_nnue)
    explicit_arch_path = config.get("candidate_architecture")
    candidate_architecture_path = (
        Path(explicit_arch_path).resolve() if explicit_arch_path else
        _candidate_architecture_path(candidate_nnue)
    )
    same_as_frozen = candidate_sha.lower() == _sha256(frozen_nnue).lower()
    if candidate_architecture_path.is_file():
        candidate_architecture = json.loads(candidate_architecture_path.read_text(encoding="utf-8"))
        _validate_architecture(
            candidate_architecture,
            label="candidate",
            nnue_sha=None if same_as_frozen else candidate_sha,
        )
    elif same_as_frozen:
        candidate_architecture = frozen_manifest["architecture"]
        candidate_architecture_path = Path(
            config.get("frozen_dir", str(frozen_dir))
        ).resolve() / "frozen_baseline_manifest.json"
    else:
        raise FileNotFoundError(
            f"candidate architecture manifest not found: {candidate_architecture_path}"
        )
    run_root = Path(config["output"]).resolve() / f"{_slug(config['candidate_name'])}-{candidate_sha[:12]}"
    book = Path(config["openings"]).resolve()
    seed = int(config["seed"])
    pairs = int(config["pairs"])
    workers = int(config["workers"])
    move_ms = int(config["move_time_ms"])

    # Same binary on both sides deliberately isolates candidate weights.
    main_config = {
        "engine1_name": str(config["candidate_name"]),
        "engine1_executable": str(frozen_exe),
        "engine1_nnue": str(candidate_nnue),
        "engine2_name": "frozen_production",
        "engine2_executable": str(frozen_exe),
        "engine2_nnue": str(frozen_nnue),
        "pairs": pairs,
        "move_time_ms": move_ms,
        "workers": workers,
        "seed": seed,
        "openings": str(book),
        "output": str(run_root / "vs-frozen-main"),
        "bootstrap": int(config["bootstrap"]),
    }
    main_summary = match_finalists.run(main_config)
    _require_complete(main_summary, label="frozen main", pairs=pairs)

    claustro_config = dict(run_benchmark.CONFIG)
    claustro_config.update({
        "opponents": ["claustrophobia"],
        "pairs": pairs,
        "workers": workers,
        "seed": seed,
        "openings": str(book),
        "output": str(run_root / "vs-claustrophobia"),
        "resume": True,
        "retry_failed": True,
        "auto_setup": False,
        "zq_executable": str(frozen_exe),
        "nnue": str(candidate_nnue),
        "zq_move_time_ms": move_ms,
        "claustrophobia_move_time_ms": move_ms,
        "claustrophobia_max_sims": int(config["claustrophobia_max_sims"]),
        "claustrophobia_device": str(config["claustrophobia_device"]),
        "bootstrap": int(config["bootstrap"]),
    })
    claustro_report = run_benchmark.run(claustro_config)
    claustro_summary = claustro_report.get("summaries", {}).get("claustrophobia", {})
    _require_complete(claustro_summary, label="Claustrophobia", pairs=pairs)

    report = {
        "schema": "zquoridor.central_finetune_evaluation.v1",
        "candidate_name": str(config["candidate_name"]),
        "candidate_architecture": str(candidate_architecture_path),
        "candidate_executable": str(frozen_exe),
        "candidate_nnue": str(candidate_nnue),
        "candidate_nnue_sha256": candidate_sha,
        "frozen_executable": str(frozen_exe),
        "frozen_executable_sha256": _sha256(frozen_exe),
        "frozen_nnue": str(frozen_nnue),
        "frozen_nnue_sha256": _sha256(frozen_nnue),
        "frozen_manifest": str(frozen_dir / "frozen_baseline_manifest.json"),
        "frozen_manifest_sha256": _sha256(frozen_dir / "frozen_baseline_manifest.json"),
        "openings": str(book),
        "openings_sha256": _sha256(book),
        "pairs_per_opponent": pairs,
        "games_per_opponent": 2 * pairs,
        "move_time_ms": move_ms,
        "workers": workers,
        "seed": seed,
        "vs_frozen_main": main_summary,
        "vs_claustrophobia": claustro_summary,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run(resolve_config(argv))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"central evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
