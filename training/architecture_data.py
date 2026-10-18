"""Inventory local corpora and prepare complete rich labels for shared training."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))
from mix_experimental_datasets import REQUIRED, _sha256, _atomic_json
from prepare_temporal_search import encode, STATE_KEYS

# These source declarations preserve the production curriculum. Composite
# ancestors remain in the inventory but do not add duplicate gradient mass.
ANCESTORS = {"multipath-master-11m", "multipath-weakness-boosted-11m", "multitier-final-dataset",
             "massive-background-10m", "mixed-gen1-500k-search20", "replay_clean_stored", "center-rush-200k-priority"}


def inspect_npz(path):
    """Read headers without materializing compressed training arrays."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
            result[name[:-4]] = dict(shape=list(shape), dtype=str(dtype))
    return result


def source_settings(name, family=None):
    """Declare provenance separately from target support or policy entropy."""
    settings = dict(pool="historical", policy_quality=.8, value_quality=1., aux_quality=.8,
                    weight_cap=30., head_quality_cap=12., provenance="historical_search_or_mixed")
    if family == "gumbel":
        settings.update(pool="gumbel", policy_quality=4., value_quality=2., aux_quality=3.,
                        provenance="dense_completed_gumbel_policy_with_partially_searched_action_values")
    elif family == "wall":
        settings.update(pool="wall", policy_quality=2., value_quality=1.5, aux_quality=1.5,
                        provenance="full_root_visits_on_synthetic_wall_positions")
    elif family == "temporal":
        settings.update(pool="temporal", policy_quality=1., value_quality=3., aux_quality=1.,
                        provenance="exact_ply_temporal_search_values_and_original_current_policy")
    elif "claustro" in name or "claustro75" in name:
        settings.update(pool="teacher", policy_quality=2., value_quality=2., aux_quality=1.5,
                        provenance="external_search_teacher")
    elif "critical" in name or "weakness" in name:
        settings.update(pool="teacher", policy_quality=1.5, value_quality=1.5, aux_quality=1.,
                        provenance="curated_search_teacher_mixture")
    elif "background" in name:
        settings.update(policy_quality=.5, value_quality=.5, aux_quality=.5, provenance="forward_distillation_anchor")
    elif "replay" in name:
        settings.update(pool="replay", provenance="stored_or_inference_replay")
    return settings


def inventory(roots):
    rows = []
    candidates = []
    for root in roots:
        path = Path(root)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            candidates.append(path)
        elif path.exists():
            candidates.extend(path.rglob("dataset.npz"))
            candidates.extend(p.parent for p in path.rglob("dataset.manifest.json") if not (p.parent/"dataset.npz").exists())
    master_present = any(p.parent.name == "multipath_unified_clean_15m" for p in candidates if p.is_file())
    seen = set()
    for path in sorted(set(p.resolve() for p in candidates)):
        name = path.parent.name if path.is_file() else path.name
        if name == "dataset":
            name = path.parent.name + "_dataset"
        row = dict(name=name, path=str(path), status="available", **source_settings(name))
        try:
            if path.is_file():
                fields = inspect_npz(path)
                row["samples"] = fields.get("value", {}).get("shape", [0])[0]
                row["fields"] = sorted(fields)
                if set(REQUIRED) - fields.keys():
                    row.update(status="excluded", reason="missing_canonical_fields")
                elif not {"group_id", "opening_index"}.intersection(fields):
                    row.update(status="excluded", reason="missing_trajectory_group_provenance")
                elif master_present and name in ANCESTORS:
                    row.update(status="covered", reason="included_in_multipath_unified_clean_15m")
            else:
                manifest = json.loads((path/"dataset.manifest.json").read_text(encoding="utf-8"))
                row["samples"] = manifest.get("samples", 0)
                if not manifest.get("complete"):
                    row.update(status="excluded", reason="incomplete_manifest")
                if "temporal" in manifest.get("schema", ""):
                    row.update(source_settings(name, "temporal"))
                if "surprise" in name or "surprise" in str(path.parent):
                    row.update(status="covered", reason="weight_variant_of_same_source")
            if name in seen and row["status"] == "available":
                row["name"] = name + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:8]
            seen.add(name)
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            row.update(status="excluded", reason=type(error).__name__)
        rows.append(row)
    return rows


def ensure_encoder():
    executable = ROOT / "bin" / ("teacher_encode_state.exe" if sys.platform == "win32" else "teacher_encode_state")
    source = ROOT/"tools/teacher/encode_state.cpp"
    dependencies = [source, *list((ROOT/"src").glob("*.hpp"))]
    if executable.exists() and executable.stat().st_mtime >= max(p.stat().st_mtime for p in dependencies):
        return executable
    compiler = shutil.which("g++") or ("C:/mingw64/bin/g++.exe" if Path("C:/mingw64/bin/g++.exe").exists() else None)
    if compiler is None:
        raise RuntimeError("A C++17 compiler is required for canonical state validation.")
    executable.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, "-O2", "-std=c++17", "-I"+str(ROOT/"src"), str(source), "-o", str(executable)], check=True)
    return executable


def jsonl(path):
    """Read complete records and omit only an incomplete final append."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n") and not stream.read(1):
                    return
                raise


def prepare_rich(positions_file, labels_file, output, seed=20261018, val_fraction=.05):
    """Join rich labels by ID, preserve full policies, and blend exact future values."""
    output = Path(output)
    identity = {"positions_sha256": _sha256(positions_file), "labels_sha256": _sha256(labels_file),
                "preparer_sha256": _sha256(Path(__file__)), "seed": seed, "val_fraction": val_fraction}
    manifest_file = output.with_suffix(".manifest.json")
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text(encoding="utf-8"))
        if old.get("identity") != identity or old.get("sha256") != _sha256(output):
            raise ValueError("Rich inputs changed. Select a new preparation directory.")
        return old
    positions = list(jsonl(positions_file))
    labels = {row["id"]: row for row in jsonl(labels_file)}
    # A Gumbel reset breaks the exact history prefix. One complete prefix chain
    # is one trajectory group. Repeated trajectories share the same group hash.
    chains, chain = [], []
    for position in positions:
        history = position["history"]
        if chain and (len(history) != len(chain[-1]["history"]) + 1 or history[:-1] != chain[-1]["history"]):
            chains.append(chain)
            chain = []
        chain.append(position)
    if chain:
        chains.append(chain)
    rows, groups, values, policies, effort, temporal = [], [], [], [], [], []
    missing = 0
    for chain in chains:
        is_gumbel = any("gumbel" in p.get("source", "") for p in chain)
        signature = chain[-1]["history"][:12] if is_gumbel else chain[-1]["history"]
        group = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        by_ply = {}
        for position in chain:
            label = labels.get(position["id"])
            if label and "value" in label.get("target", {}):
                by_ply[len(position["history"])] = (int(position["side_to_move"]), float(label["target"]["value"]))
        for position in chain:
            label = labels.get(position["id"])
            target = label.get("target", {}) if label else {}
            if "policy" not in target or "value" not in target:
                missing += 1
                continue
            policy = np.asarray(target["policy"], dtype=np.float32)
            value = float(target["value"])
            if policy.shape != (209,) or not np.isfinite(policy).all() or (policy < 0).any() or policy.sum() <= 0:
                raise ValueError("Invalid rich policy.")
            if not math.isfinite(value) or abs(value) > 1:
                raise ValueError("Invalid rich value.")
            ply, side = len(position["history"]), int(position["side_to_move"])
            terms = [(coefficient, by_ply[ply+offset]) for offset, coefficient in ((0,.5),(4,.25),(8,.125),(16,.0625))
                     if ply+offset in by_ply] if is_gumbel else [(1., (side, value))]
            value = sum(c * v * (1 if s == side else -1) for c, (s, v) in terms) / sum(c for c, _ in terms)
            rows.append(position)
            groups.append(group)
            values.append(value)
            policies.append(policy / policy.sum())
            effort.append(float(label.get("budget", {}).get("move_time_ms", 0)))
            temporal.append(len(terms) > 1)
    if not rows:
        raise ValueError("No complete rich labels are available.")
    states = encode([" ".join(position["history"]) for position in rows])
    keep = np.array([state is not None for state in states])
    if not keep.all():
        raise ValueError("The native encoder rejected a rich position.")
    arrays = {key: np.array([state[i] for state in states], dtype=np.uint64 if key in ("walls_h", "walls_v") else np.int64)
              for i, key in enumerate(STATE_KEYS)}
    arrays.update(value=np.asarray(values, np.float32), policy=np.asarray(policies, np.float32),
                  weight=np.ones(len(rows), np.float32), group_id=np.asarray(groups, "S64"),
                  is_val=np.array([int(hashlib.sha256((str(seed)+g).encode()).hexdigest()[:16],16)/2**64 < val_fraction for g in groups]),
                  search_budget_ms=np.asarray(effort, np.float32),
                  value_sample_weight=1 + np.asarray(temporal, np.float32),
                  temporal_future_available=np.asarray(temporal, bool))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, **arrays)
    output.with_suffix(".tmp").replace(output)
    report = dict(schema="zquoridor.rich_training.v1", complete=True, identity=identity, samples=len(rows),
                  rows_with_exact_future=int(sum(temporal)), missing_labels=missing,
                  value_semantics="signed_current_mover; available exact plies 0,4,8,16; no invented terminal value",
                  policy_semantics="original current-position distribution; no top-k truncation",
                  sha256=_sha256(output))
    _atomic_json(manifest_file, report)
    return report


def prepare_ledgers(ledgers, output, reference, seed=20261018):
    """Prepare rich JSONL pairs and trajectory datasets from materialized ledgers."""
    ensure_encoder()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for ledger in ledgers:
        files.extend(jsonl(ledger))
    lookup = {row["path"].replace("\\", "/"): row for row in files}
    sources, temporal_files = [], defaultdict(list)
    for key, row in sorted(lookup.items()):
        local = Path(row["local"])
        if not local.is_file() or row.get("sha256") and _sha256(local) != row["sha256"]:
            raise ValueError("A downloaded artifact is missing or changed: " + key)
        family = row.get("kind", "")
        if row["title"] == "positions.jsonl":
            label = lookup.get(key.rsplit("/", 1)[0]+"/targets.jsonl")
            if label is None:
                raise ValueError("Rich positions require the full targets.jsonl, including budget provenance.")
            snapshot_identity = key + _sha256(local) + _sha256(label["local"]) + _sha256(Path(__file__))
            name = family + "_" + hashlib.sha256(snapshot_identity.encode()).hexdigest()[:10]
            dataset = output/name/"dataset.npz"
            prepare_rich(local, label["local"], dataset, seed)
            sources.append(dict(name=name, path=str(dataset), status="available",
                                **source_settings(name, "gumbel" if "gumbel" in family else "wall"),
                                search_budget_ms_reference=20. if "gumbel" in family else 200.))
        elif row["title"].endswith((".bin", ".meta", "games.jsonl.gz")):
            canonical_kind = "claustro_search_games_v1" if "claustro" in family else "selfplay_858_v1"
            entry = dict(row, kind=canonical_kind)
            if entry["title"].endswith(".meta.bin"):
                entry["title"] = entry["title"][:-4]
                entry["path"] = entry["path"][:-4]
            temporal_files[canonical_kind].append(entry)
    for family, records in temporal_files.items():
        ledger = output/(family+".jsonl")
        ledger.write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
        dataset = output/family/"dataset"
        subprocess.run([sys.executable, str(ROOT/"training/prepare_temporal_search.py"),
                        "--downloads", str(ledger), "--output", str(dataset), "--reference", str(reference)], check=True, cwd=ROOT)
        sources.append(dict(name=family+"_temporal", path=str(dataset), status="available", **source_settings(family, "temporal")))
    return sources
