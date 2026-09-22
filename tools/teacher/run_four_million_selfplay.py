#!/usr/bin/env python3
"""Generate an exact, resumable four-million-state self-play corpus."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.teacher.build_selfplay_seed_schedule import REQUIRED_FAMILIES
from tools.teacher.selfplay_unique_store import (
    META_RECORD_SIZE,
    POLICY_DIM,
    QUALITY_MISSING_ROOT_VALUE,
    V3_DTYPE,
    V3_RECORD_SIZE,
    UniqueStateStore,
)

SCHEMA = "zquoridor.four_million_selfplay.v1"
DEFAULT_TOTAL_TARGET = 4_000_000
DEFAULT_CENTRAL_TARGET = 2_500_000
DEFAULT_BROAD_TARGET = 1_500_000
DEFAULT_FAMILY_FLOOR = 400_000
# Edit this block to run the controller without CLI options. CLI options override
# these values for one invocation. Paths are deliberately relative to ROOT.
CONFIG = {
    "out": "data/selfplay_canonical_v3/contact-4m-50ms",
    "exe": "bin/selfplay_phase_mcgs075.exe",
    "exe_arg": [],
    "weights": "results/experiments/multipath-phase512-searchboost-100ep/student_int8.bin",
    "central_schedule": "data/selfplay_canonical_v3/contact-4m-50ms/schedules/central_positions.jsonl",
    "broad_schedule": "data/selfplay_canonical_v3/contact-4m-50ms/schedules/broad_positions.jsonl",
    "weakness_corpus": [],
    "total_target": DEFAULT_TOTAL_TARGET, "central_target": DEFAULT_CENTRAL_TARGET,
    "broad_target": DEFAULT_BROAD_TARGET, "family_floor": DEFAULT_FAMILY_FLOOR,
    "games_per_shard": 512, "seed_rows_per_shard": 10_000, "time_ms": 50,
    "max_plies": 140, "seed": 20260921, "fine_tune_pid": None,
    "threads_during_fine_tune": 10, "threads_after_fine_tune": 12,
    "fine_tune_threads": 4, "cpu_thread_limit": 16, "reliability": 2.0,
    "dry_run": True,
}
META_DTYPE = np.dtype([
    ("game", "<u8"), ("root", "<f4"), ("plies", "<u2"), ("length", "<u2"),
    ("source", "u1"), ("flags", "u1"), ("reserved", "<u2"),
])


@dataclass
class TargetCounts:
    """Hold admitted unique counts for the corpus targets."""

    total: int
    central: int
    broad: int
    family: dict[str, int]


@dataclass(frozen=True)
class ResumeState:
    """Hold the accepted shard list and the next accepted shard index."""

    next_shard: int
    counts: TargetCounts
    accepted_shards: tuple[dict, ...]


def counts_for(*, total: int, central: int, broad: int,
               family: Mapping[str, int], **_counters: int) -> TargetCounts:
    """Build target counts from explicit counters."""
    return TargetCounts(total, central, broad, dict(family))


def targets_met(counts: TargetCounts, *, total_target: int = DEFAULT_TOTAL_TARGET,
                central_target: int = DEFAULT_CENTRAL_TARGET,
                broad_target: int = DEFAULT_BROAD_TARGET,
                family_floor: int = DEFAULT_FAMILY_FLOOR) -> bool:
    """Return true only when every corpus target is satisfied."""
    return (
        counts.total >= total_target
        and counts.central >= central_target
        and counts.broad >= broad_target
        and all(counts.family.get(name, 0) >= family_floor for name in REQUIRED_FAMILIES)
    )


def family_target(central_target: int, family_floor: int) -> int:
    """Return the balanced central target for one opening family."""
    return max(family_floor, math.ceil(central_target / len(REQUIRED_FAMILIES)))


def _unclassified_central(counts: TargetCounts) -> int:
    """Return central states that do not belong to a required family."""
    assigned = sum(max(0, counts.family.get(name, 0)) for name in REQUIRED_FAMILIES)
    return max(0, counts.central - assigned)


def _family_target_for_counts(counts: TargetCounts, central_target: int,
                              family_floor: int) -> int:
    """Return a balanced family target after imported central states."""
    available = max(0, central_target - _unclassified_central(counts))
    return max(family_floor, math.ceil(available / len(REQUIRED_FAMILIES)))


def _family_target_for_name(counts: TargetCounts, name: str,
                            central_target: int, family_floor: int) -> int:
    """Return the exact balanced target for one named family."""
    available = max(0, central_target - _unclassified_central(counts))
    base = max(family_floor, available // len(REQUIRED_FAMILIES))
    remainder = max(0, available - base * len(REQUIRED_FAMILIES))
    rank = REQUIRED_FAMILIES.index(name)
    return base + (1 if rank < remainder else 0)


def proportional_deficits(
    counts: TargetCounts,
    *,
    central_target: int = DEFAULT_CENTRAL_TARGET,
    broad_target: int = DEFAULT_BROAD_TARGET,
    family_floor: int = DEFAULT_FAMILY_FLOOR,
    total_target: int | None = None,
) -> dict[tuple[str, str | None], float]:
    """Return nonnegative proportional deficits for selectable pools."""
    total_target = central_target + broad_target if total_target is None else total_target
    result: dict[tuple[str, str | None], float] = {}
    if _capacity(counts, "broad", None, total_target=total_target,
                 central_target=central_target, broad_target=broad_target,
                 family_floor=family_floor) > 0:
        result[("broad", None)] = max(0.0, (broad_target - counts.broad) / max(1, broad_target))
    for name in REQUIRED_FAMILIES:
        if _capacity(counts, "central", name, total_target=total_target,
                     central_target=central_target, broad_target=broad_target,
                     family_floor=family_floor) > 0:
            family_target_for_name = _family_target_for_name(
                counts, name, central_target, family_floor
            )
            result[("central", name)] = max(
                0.0,
                (family_target_for_name - counts.family.get(name, 0))
                / max(1, family_target_for_name),
            )
    return result


def select_next_pool(
    counts: TargetCounts,
    *,
    central_target: int = DEFAULT_CENTRAL_TARGET,
    broad_target: int = DEFAULT_BROAD_TARGET,
    family_floor: int = DEFAULT_FAMILY_FLOOR,
    total_target: int | None = None,
) -> tuple[str, str | None]:
    """Select the pool that has the largest proportional deficit."""
    deficits = proportional_deficits(
        counts,
        central_target=central_target,
        broad_target=broad_target,
        family_floor=family_floor,
        total_target=total_target,
    )
    if not deficits:
        raise RuntimeError("no self-play pool has remaining capacity")
    return min(deficits, key=lambda item: (-deficits[item], item[0], item[1] or ""))


def select_thread_count(
    fine_tune_active: bool,
    *,
    during_fine_tune: int = 10,
    after_fine_tune: int = 12,
    fine_tune_threads: int = 4,
    cpu_thread_limit: int = 16,
) -> int:
    """Return a self-play thread count that respects the local CPU limit."""
    if not 1 <= cpu_thread_limit <= 16:
        raise ValueError("the CPU thread limit must be from 1 to 16")
    if during_fine_tune < 1 or after_fine_tune < 1:
        raise ValueError("the self-play thread counts must be positive")
    if fine_tune_threads < 0:
        raise ValueError("the fine-tune thread count must not be negative")
    if during_fine_tune + fine_tune_threads > cpu_thread_limit:
        raise ValueError("the fine-tune and self-play thread counts exceed the CPU limit")
    if after_fine_tune > cpu_thread_limit:
        raise ValueError("the self-play thread count exceeds the CPU limit")
    requested = during_fine_tune if fine_tune_active else after_fine_tune
    used = fine_tune_threads if fine_tune_active else 0
    available = cpu_thread_limit - used
    if available < 1:
        raise ValueError("the CPU thread limit leaves no self-play thread")
    return min(requested, available)


class OutputLock:
    """Hold one nonblocking lock for a controller output directory."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / "controller.lock"
        self._handle = None

    def __enter__(self) -> "OutputLock":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.path.open("a+b")
        except OSError as error:
            raise RuntimeError(f"controller lock is already held: {self.path}") from error
        try:
            self._handle.seek(0)
            if self._handle.read(1) == b"":
                self._handle.seek(0)
                self._handle.write(b"0")
                self._handle.flush()
        except OSError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"controller lock is already held: {self.path}") from error
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"controller lock is already held: {self.path}") from error
        self._handle.seek(0)
        self._handle.truncate(0)
        self._handle.write(f"{os.getpid()}\n".encode("ascii"))
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_manifest(out_dir: Path) -> dict:
    path = out_dir / "manifest.json"
    if not path.exists():
        return {"schema": SCHEMA, "accepted_shards": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"invalid controller manifest: {path}")
    accepted = value.get("accepted_shards")
    if not isinstance(accepted, list):
        raise ValueError(f"invalid accepted shard list: {path}")
    return value


def _safe_shard_path(out_dir: Path, name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("the shard manifest path is invalid")
    path = (out_dir / name).resolve()
    if out_dir.resolve() not in path.parents:
        raise ValueError("the shard manifest path leaves the output directory")
    return path


def _validate_pair(v3_path: Path, meta_path: Path) -> list[bytes]:
    if not v3_path.exists() or not meta_path.exists():
        raise ValueError("the shard pair is incomplete")
    if v3_path.stat().st_size % V3_RECORD_SIZE:
        raise ValueError("the V3 shard is not 64-byte aligned")
    if meta_path.stat().st_size % META_RECORD_SIZE:
        raise ValueError("the metadata sidecar is not 20-byte aligned")
    record_count = v3_path.stat().st_size // V3_RECORD_SIZE
    if meta_path.stat().st_size // META_RECORD_SIZE != record_count:
        raise ValueError("the V3 shard and metadata sidecar differ in record count")
    v3 = np.memmap(v3_path, dtype=V3_DTYPE, mode="r")
    metadata = np.memmap(meta_path, dtype=META_DTYPE, mode="r")
    try:
        keys: list[bytes] = []
        for row_index, (row, meta) in enumerate(zip(v3, metadata)):
            action = int(row["policy_target"])
            if action >= POLICY_DIM:
                raise ValueError(f"the shard has an illegal policy target at record {row_index}")
            for index, probability in zip(row["policy_top_idx"], row["policy_top_prob"]):
                if int(probability) and int(index) >= POLICY_DIM:
                    raise ValueError(f"the shard has an illegal policy index at record {row_index}")
            if not (int(meta["flags"]) & QUALITY_MISSING_ROOT_VALUE) and not np.isfinite(meta["root"]):
                raise ValueError(f"the metadata root value is not finite at record {row_index}")
            if int(meta["length"]) and int(meta["plies"]) > int(meta["length"]):
                raise ValueError(f"the metadata terminal distance exceeds the game length at record {row_index}")
            keys.append(UniqueStateStore.canonical_key(row))
        return keys
    finally:
        # Windows keeps the mapped file open until the memmap object is
        # released. Release both views before quarantine moves the shard.
        # NumPy memmap owns a Windows file mapping.  ``del`` alone is not
        # sufficient when validation raises inside the loop: the mapping can
        # survive until the next GC cycle and make quarantine fail with
        # WinError 32.  Close the underlying mmap explicitly, then collect
        # any wrapper objects before the caller moves the files.
        for mapped in (metadata, v3):
            mmap_obj = getattr(mapped, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        del metadata
        del v3
        gc.collect()


def _quarantine(out_dir: Path, paths: Iterable[Path]) -> None:
    quarantine = out_dir / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            continue
        destination = quarantine / path.name
        suffix = 1
        while destination.exists():
            destination = quarantine / f"{path.stem}.{suffix}{path.suffix}"
            suffix += 1
        last_error: OSError | None = None
        for attempt in range(8):
            try:
                shutil.move(str(path), str(destination))
                last_error = None
                break
            except PermissionError as error:
                last_error = error
                gc.collect()
                time.sleep(0.10 * (attempt + 1))
        if last_error is not None:
            raise last_error


def _manifest_counts(accepted_shards: Iterable[Mapping[str, object]]) -> TargetCounts:
    total = central = broad = 0
    family: dict[str, int] = {}
    for entry in accepted_shards:
        records = int(entry.get("unique_records", entry.get("records", 0)))
        total += records
        source = entry.get("source")
        if source == "central":
            central += records
            name = entry.get("family")
            if isinstance(name, str):
                family[name] = family.get(name, 0) + records
        elif source == "broad":
            broad += records
    return TargetCounts(total, central, broad, family)


def discover_resume_state(out_dir: str | Path) -> ResumeState:
    """Validate accepted artifacts and quarantine every loose shard pair."""
    root = Path(out_dir)
    manifest = _read_manifest(root)
    launch_intent = manifest.get("launch_intent")
    if launch_intent is not None:
        if not isinstance(launch_intent, dict):
            raise ValueError("the launch intent manifest entry is invalid")
        stale = dict(launch_intent)
        stale["stale_reason"] = "unbound_launch_intent"
        manifest.setdefault("stale_launch_intents", []).append(stale)
        _atomic_json(root / "manifest.json", manifest)
        raise RuntimeError(
            "unbound launch intent remains in the manifest; inspect the child before resuming"
        )
    active = manifest.get("active_shard")
    if active is not None:
        if not isinstance(active, dict) or not isinstance(active.get("pid"), int):
            raise ValueError("the active child manifest entry is invalid")
        pid = active["pid"]
        if _process_exists(pid):
            identity_status = _active_child_identity_status(active)
            if identity_status == "match":
                raise RuntimeError(f"active child {pid} owns the staging shard")
            if identity_status == "unknown":
                raise RuntimeError(f"active child {pid} identity cannot be verified")
            active = dict(active)
            active["stale_reason"] = "child_identity_mismatch"
        else:
            active = dict(active)
            active["stale_reason"] = "child_not_found"
        paths = []
        for name in (active.get("v3"), active.get("meta")):
            try:
                paths.append(_safe_shard_path(root, name))
            except ValueError:
                pass
        _quarantine(root, paths)
        orphaned = manifest.setdefault("orphaned_shards", [])
        if isinstance(orphaned, list):
            orphaned.append(active)
        manifest.pop("active_shard", None)
        _atomic_json(root / "manifest.json", manifest)
    valid: list[dict] = []
    accepted_paths: set[Path] = set()
    changed = False
    for raw in manifest["accepted_shards"]:
        if not isinstance(raw, dict):
            changed = True
            continue
        try:
            v3 = _safe_shard_path(root, raw.get("v3"))
            meta = _safe_shard_path(root, raw.get("meta"))
            _validate_pair(v3, meta)
            valid.append(raw)
            accepted_paths.update((v3, meta))
        except (OSError, ValueError):
            changed = True
            paths = []
            for name in (raw.get("v3"), raw.get("meta")):
                try:
                    paths.append(_safe_shard_path(root, name))
                except ValueError:
                    pass
            _quarantine(root, paths)
    loose = []
    for path in root.glob("shard_*.bin"):
        if path.resolve() not in accepted_paths:
            loose.extend((path, path.with_suffix(".meta")))
    for path in root.glob("shard_*.meta"):
        if path.resolve() not in accepted_paths:
            loose.extend((path.with_suffix(".bin"), path))
    _quarantine(root, loose)
    staging = root / "staging"
    if staging.exists():
        staged = []
        for path in staging.glob("shard_*.bin"):
            staged.extend((path, path.with_suffix(".meta")))
        for path in staging.glob("shard_*.meta"):
            staged.extend((path.with_suffix(".bin"), path))
        _quarantine(root, staged)
    if changed:
        manifest["accepted_shards"] = valid
        _atomic_json(root / "manifest.json", manifest)
    indices = [int(item["index"]) for item in valid if isinstance(item.get("index"), int)]
    return ResumeState(max(indices, default=-1) + 1, _manifest_counts(valid), tuple(valid))


def _counts_from_store(store: UniqueStateStore) -> TargetCounts:
    central = int(store.db.execute(
        "SELECT COUNT(*) FROM states WHERE source='central' OR source LIKE 'weakness-%'"
    ).fetchone()[0])
    broad = int(store.db.execute("SELECT COUNT(*) FROM states WHERE source='broad'").fetchone()[0])
    family = {
        str(name): int(count)
        for name, count in store.db.execute(
            "SELECT family, COUNT(*) FROM states WHERE source='central' GROUP BY family"
        )
        if name is not None
    }
    return TargetCounts(store.unique_count, central, broad, family)


def _store_counters(store: UniqueStateStore) -> dict[str, int]:
    """Return durable store counters, including derived duplicates."""
    counters = store.counts()
    raw = int(counters.get("raw", 0))
    unique = int(counters.get("unique", 0))
    return {
        "raw": raw,
        "unique": unique,
        "duplicate": max(0, raw - unique),
        "rejected": int(counters.get("rejected", 0)),
        "replaced": int(counters.get("replaced", 0)),
        "equal": int(counters.get("equal", 0)),
    }


def reconcile_store(out_dir: str | Path, accepted_shards: Iterable[Mapping[str, object]], *,
                    weakness_corpora: Iterable[str | Path] = ()) -> UniqueStateStore:
    """Rebuild the SQLite index from accepted shards before a resumed run."""
    root = Path(out_dir)
    final_path = root / "states.sqlite"
    temporary = root / "states.reconcile.sqlite"
    if temporary.exists():
        temporary.unlink()
    rebuilt = UniqueStateStore(temporary)
    try:
        for entry in accepted_shards:
            v3 = _safe_shard_path(root, entry["v3"])
            meta = _safe_shard_path(root, entry["meta"])
            rebuilt.admit_shard(
                v3,
                meta,
                source=str(entry.get("source", "unknown")),
                family=entry.get("family") if isinstance(entry.get("family"), str) else None,
            )
        for corpus in weakness_corpora:
            rebuilt.import_weakness_loss_100ms(corpus)
        rebuilt.close()
        os.replace(temporary, final_path)
    except Exception:
        rebuilt.close()
        raise
    return UniqueStateStore(final_path)


def _read_schedule(path: Path, family: str | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"the schedule row is not an object: {path}:{line_number}")
            if family is None or value.get("opening_category") == family:
                rows.append(value)
    if not rows:
        label = family if family is not None else "broad"
        raise ValueError(f"the {label} schedule has no usable rows")
    return rows


def _write_rotated_seeds(out_dir: Path, rows: Sequence[dict], index: int,
                         attempt: int, rows_per_shard: int) -> Path:
    if rows_per_shard < 1:
        raise ValueError("the seed row count must be positive")
    start = ((index + attempt) * rows_per_shard) % len(rows)
    selected = [rows[(start + offset) % len(rows)] for offset in range(rows_per_shard)]
    seed_dir = out_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"shard_{index:06d}_{attempt:04d}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return path


def _stage_paths(out_dir: Path, index: int, attempt: int) -> tuple[Path, Path]:
    stage = out_dir / "staging"
    stage.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{index:06d}_{attempt:04d}"
    return stage / f"{stem}.bin", stage / f"{stem}.meta"


def _accepted_paths(out_dir: Path, index: int) -> tuple[Path, Path]:
    return out_dir / f"shard_{index:06d}.bin", out_dir / f"shard_{index:06d}.meta"


def _write_new_records(source_v3: Path, source_meta: Path, keys: Sequence[bytes],
                       store: UniqueStateStore, capacity: int,
                       final_v3: Path, final_meta: Path) -> list[bytes]:
    if capacity < 1:
        return []
    v3 = np.memmap(source_v3, dtype=V3_DTYPE, mode="r")
    metadata = np.memmap(source_meta, dtype=META_DTYPE, mode="r")
    selected: list[int] = []
    selected_keys: list[bytes] = []
    seen: set[bytes] = set()
    new_count = 0
    for offset, key in enumerate(keys):
        if key in seen or store.lookup(key) is not None:
            selected.append(offset)
            selected_keys.append(key)
            continue
        if new_count >= capacity:
            continue
        seen.add(key)
        new_count += 1
        selected.append(offset)
        selected_keys.append(key)
    if not selected:
        return []
    np.asarray(v3[selected], dtype=V3_DTYPE).tofile(final_v3)
    np.asarray(metadata[selected], dtype=metadata.dtype).tofile(final_meta)
    return selected_keys


def _process_exists(pid: int | None) -> bool:
    if pid is None or pid < 1:
        return False
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return False
            return any(token == str(pid) for token in completed.stdout.split())
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _command_hash(command: Sequence[str]) -> str:
    """Return a stable identity for one child command."""
    tokens = []
    for item in command:
        token = str(item)
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        tokens.append(token)
    payload = json.dumps(tokens, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _process_identity(pid: int) -> dict[str, str] | None:
    """Read the command identity for a running process."""
    try:
        if os.name == "nt":
            query = (
                "(Get-CimInstance Win32_Process -Filter 'ProcessId = "
                f"{int(pid)}').CommandLine"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                capture_output=True,
                text=True,
                check=False,
            )
            command_line = completed.stdout.strip()
        else:
            command_line = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(
                b"\x00", b" "
            ).decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return None
    if not command_line:
        return None
    try:
        tokens = shlex.split(command_line, posix=os.name != "nt")
    except ValueError:
        tokens = [command_line]
    return {"command_hash": _command_hash(tokens)}


def _active_child_identity_status(active: Mapping[str, object]) -> str:
    """Return match, mismatch, or unknown for an active child entry."""
    expected = active.get("process_identity")
    if not isinstance(expected, Mapping):
        return "unknown"
    current = _process_identity(int(active["pid"]))
    if current is None:
        return "unknown"
    return "match" if dict(current) == dict(expected) else "mismatch"


def _source_class(source: str, family: str | None) -> int:
    if source == "broad":
        return 0
    assert family is not None
    return REQUIRED_FAMILIES.index(family) + 1


def _capacity(counts: TargetCounts, source: str, family: str | None, *,
              total_target: int, central_target: int, broad_target: int,
              family_floor: int) -> int:
    remaining_total = total_target - counts.total
    if source == "broad":
        return max(0, min(remaining_total, broad_target - counts.broad))
    assert family is not None
    reserved_for_other_families = sum(
        max(0, family_floor - counts.family.get(name, 0))
        for name in REQUIRED_FAMILIES if name != family
    )
    return max(0, min(
        remaining_total,
        central_target - counts.central - reserved_for_other_families,
        _family_target_for_name(counts, family, central_target, family_floor)
        - counts.family.get(family, 0),
    ))


def _command(args: argparse.Namespace, stage_v3: Path, stage_meta: Path,
             seeds: Path, index: int, attempt: int, source: str,
             family: str | None, threads: int) -> list[str]:
    return [
        str(args.exe),
        *args.exe_arg,
        "--games", str(args.games_per_shard),
        "--chunk-games", str(args.games_per_shard),
        "--threads", str(threads),
        "--time-ms", str(args.time_ms),
        "--max-plies", str(args.max_plies),
        "--positions", str(seeds),
        "--nnue-weights", str(args.weights),
        "--seed", str(args.seed + index * 999_983 + attempt * 7_919),
        "--start-shard", str(index),
        "--out", str(stage_v3),
        "--meta-out", str(stage_meta),
        "--meta-source-class", str(_source_class(source, family)),
    ]


def _counter_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    """Return one shard's store-counter changes."""
    names = ("raw", "unique", "rejected", "replaced", "equal")
    delta = {name: after.get(name, 0) - before.get(name, 0) for name in names}
    delta["duplicate"] = max(0, delta["raw"] - delta["unique"])
    return delta


def _game_entries(v3_path: Path, meta_path: Path, seed_id: int,
                  max_plies: int = 140) -> list[dict]:
    """Build game entries from the accepted rows and aligned metadata."""
    v3 = np.memmap(v3_path, dtype=V3_DTYPE, mode="r")
    metadata = np.memmap(meta_path, dtype=META_DTYPE, mode="r")
    by_game: dict[int, dict] = {}
    for offset, (row, meta) in enumerate(zip(v3, metadata)):
        game_id = int(meta["game"])
        entry = by_game.setdefault(game_id, {
            "game_id": game_id,
            "seed_id": seed_id,
            "offsets": [],
            "mover_results": [],
            "winners": set(),
            "lengths": [],
        })
        entry["offsets"].append(offset)
        game_result = int(row["game_result"])
        mover = int(row["mover"])
        entry["mover_results"].append(game_result)
        entry["lengths"].append(int(meta["length"]))
        if game_result and mover in (0, 1):
            entry["winners"].add(mover if game_result > 0 else 1 - mover)
    for entry in by_game.values():
        entry["mover_results"] = sorted(set(entry["mover_results"]))
        winners = entry.pop("winners")
        lengths = entry.pop("lengths")
        if len(winners) == 1:
            winner = next(iter(winners))
            result = 1.0 if winner == 0 else 0.0
            termination = "goal"
            basis = "V3 game_result identifies the winner."
        elif len(winners) > 1:
            winner = None
            result = None
            termination = "inconsistent_result"
            basis = "V3 game_result identifies conflicting winners."
        else:
            winner = -1
            result = 0.5
            if max_plies > 0 and max(lengths, default=0) >= max_plies:
                termination = "max_plies"
                basis = "The metadata game length reached max_plies."
            else:
                termination = "repetition"
                basis = "All V3 game_result values are zero before max_plies."
        entry.update({
            "winner": winner,
            "result": result,
            "termination": termination,
            "termination_basis": basis,
        })
    return [by_game[game_id] for game_id in sorted(by_game)]


def _validate_args(args: argparse.Namespace) -> None:
    """Validate controller targets and CPU reservations before generation."""
    if args.time_ms != 50:
        raise ValueError("the four-million controller requires 50 ms per move")
    if args.total_target < 1 or args.central_target < 1 or args.broad_target < 1:
        raise ValueError("the corpus targets must be positive")
    if args.family_floor < 0 or args.games_per_shard < 1 or args.seed_rows_per_shard < 1:
        raise ValueError("the shard and family parameters are invalid")
    if args.threads_during_fine_tune < 1 or args.threads_after_fine_tune < 1:
        raise ValueError("the self-play thread counts must be positive")
    if args.fine_tune_threads < 0:
        raise ValueError("the fine-tune thread count must not be negative")
    if not 1 <= args.cpu_thread_limit <= 16:
        raise ValueError("the CPU thread limit must be from 1 to 16")
    if args.threads_during_fine_tune + args.fine_tune_threads > args.cpu_thread_limit:
        raise ValueError("the fine-tune and self-play thread counts exceed the CPU limit")
    if args.threads_after_fine_tune > args.cpu_thread_limit:
        raise ValueError("the self-play thread count exceeds the CPU limit")
    if args.total_target != args.central_target + args.broad_target:
        raise ValueError("the total target must equal the central target plus the broad target")
    if args.family_floor * len(REQUIRED_FAMILIES) > args.central_target:
        raise ValueError("the family floors exceed the central target")


def run_controller(args: argparse.Namespace) -> int:
    """Run self-play until the complete unique corpus reaches every target."""
    _validate_args(args)
    for path in (args.exe, args.weights, args.central_schedule, args.broad_schedule, *args.weakness_corpus):
        if not path.exists():
            raise FileNotFoundError(path)
    out_dir = args.out.resolve()
    central_rows = {name: _read_schedule(args.central_schedule, name) for name in REQUIRED_FAMILIES}
    broad_rows = _read_schedule(args.broad_schedule)
    with OutputLock(out_dir):
        resume = discover_resume_state(out_dir)
        manifest = _read_manifest(out_dir)
        previous_corpora = manifest.get("weakness_corpora", [])
        if not isinstance(previous_corpora, list) or not all(isinstance(path, str) for path in previous_corpora):
            raise ValueError("the weakness corpus manifest list is invalid")
        corpora = {str(Path(path).resolve()) for path in previous_corpora}
        corpora.update(str(path.resolve()) for path in args.weakness_corpus)
        store = reconcile_store(out_dir, resume.accepted_shards, weakness_corpora=sorted(corpora))
        try:
            counts = _counts_from_store(store)
            store_counters = _store_counters(store)
            if (counts.total > args.total_target
                    or counts.central > args.central_target
                    or counts.broad > args.broad_target):
                raise ValueError("durable counts exceed the configured corpus targets")
            if _unclassified_central(counts) + args.family_floor * len(REQUIRED_FAMILIES) > args.central_target:
                raise ValueError(
                    "imported unclassified central states leave no feasible family composition"
                )
            game_manifest: list[object] = []
            for shard in resume.accepted_shards:
                shard_games = shard.get("games", [])
                if isinstance(shard_games, list):
                    game_manifest.extend(shard_games)
            manifest.update({
                "schema": SCHEMA,
                "targets": {
                    "total": args.total_target,
                    "central": args.central_target,
                    "broad": args.broad_target,
                    "family_floor": args.family_floor,
                },
                "schedules": {
                    "central": str(args.central_schedule.resolve()),
                    "broad": str(args.broad_schedule.resolve()),
                },
                "controller_pid": os.getpid(),
                "accepted_shards": list(resume.accepted_shards),
                "games": game_manifest,
                "weakness_corpora": sorted(corpora),
                "counters": dict(store_counters),
            })
            _atomic_json(out_dir / "manifest.json", manifest)
            index = resume.next_shard
            attempt = int(_read_progress(out_dir).get("next_attempt", 0))
            while not targets_met(
                counts,
                total_target=args.total_target,
                central_target=args.central_target,
                broad_target=args.broad_target,
                family_floor=args.family_floor,
            ):
                source, family = select_next_pool(
                    counts,
                    total_target=args.total_target,
                    central_target=args.central_target,
                    broad_target=args.broad_target,
                    family_floor=args.family_floor,
                )
                capacity = _capacity(
                    counts,
                    source,
                    family,
                    total_target=args.total_target,
                    central_target=args.central_target,
                    broad_target=args.broad_target,
                    family_floor=args.family_floor,
                )
                if capacity < 1:
                    raise RuntimeError("the selected pool has no remaining capacity")
                rows = broad_rows if source == "broad" else central_rows[family]
                seeds = _write_rotated_seeds(out_dir, rows, index, attempt, args.seed_rows_per_shard)
                fine_tune_active = _process_exists(args.fine_tune_pid)
                threads = select_thread_count(
                    fine_tune_active,
                    during_fine_tune=args.threads_during_fine_tune,
                    after_fine_tune=args.threads_after_fine_tune,
                    fine_tune_threads=args.fine_tune_threads,
                    cpu_thread_limit=args.cpu_thread_limit,
                )
                stage_v3, stage_meta = _stage_paths(out_dir, index, attempt)
                command = _command(args, stage_v3, stage_meta, seeds, index, attempt, source, family, threads)
                launch_intent = {
                    "launch_id": uuid.uuid4().hex,
                    "index": index,
                    "attempt": attempt,
                    "command": command,
                    "v3": str(stage_v3.relative_to(out_dir)),
                    "meta": str(stage_meta.relative_to(out_dir)),
                    "seed_file": str(seeds.relative_to(out_dir)),
                }
                manifest["launch_intent"] = launch_intent
                _atomic_json(out_dir / "manifest.json", manifest)
                started = time.monotonic()
                try:
                    child = subprocess.Popen(command, cwd=ROOT)
                except Exception:
                    manifest.pop("launch_intent", None)
                    _atomic_json(out_dir / "manifest.json", manifest)
                    _quarantine(out_dir, (stage_v3, stage_meta))
                    raise
                active = dict(launch_intent)
                active.update({
                    "pid": child.pid,
                    "process_identity": {"command_hash": _command_hash(command)},
                })
                manifest.pop("launch_intent", None)
                manifest["active_shard"] = active
                _atomic_json(out_dir / "manifest.json", manifest)
                _write_progress(out_dir, counts, index, attempt, "running", active_shard=active,
                                counters=store_counters)
                return_code = child.wait()
                manifest.pop("active_shard", None)
                _atomic_json(out_dir / "manifest.json", manifest)
                if return_code:
                    _quarantine(out_dir, (stage_v3, stage_meta))
                    raise RuntimeError(f"self-play shard {index} failed with exit code {return_code}")
                try:
                    keys = _validate_pair(stage_v3, stage_meta)
                except (OSError, ValueError) as error:
                    _quarantine(out_dir, (stage_v3, stage_meta))
                    manifest.setdefault("rejected_shards", []).append({
                        "index": index,
                        "attempt": attempt,
                        "reason": str(error),
                        "v3": stage_v3.name,
                        "meta": stage_meta.name,
                    })
                    _atomic_json(out_dir / "manifest.json", manifest)
                    attempt += 1
                    _write_progress(out_dir, counts, index, attempt, "running",
                                    counters=store_counters)
                    continue
                final_v3, final_meta = _accepted_paths(out_dir, index)
                selected_keys = _write_new_records(stage_v3, stage_meta, keys, store, capacity, final_v3, final_meta)
                if not selected_keys:
                    _quarantine(out_dir, (stage_v3, stage_meta))
                    attempt += 1
                    _write_progress(out_dir, counts, index, attempt, "running",
                                    counters=store_counters)
                    continue
                counters_before = store.counts()
                store.admit_shard(final_v3, final_meta, keys=selected_keys,
                                  reliability=args.reliability, source=source, family=family)
                counters = _counter_delta(counters_before, store.counts())
                counts = _counts_from_store(store)
                store_counters = _store_counters(store)
                manifest["counters"] = dict(store_counters)
                seed_id = args.seed + index * 999_983 + attempt * 7_919
                games = _game_entries(final_v3, final_meta, seed_id, args.max_plies)
                accepted_entry = {
                    "index": index,
                    "v3": final_v3.name,
                    "meta": final_meta.name,
                    "source": source,
                    "family": family,
                    "records": len(selected_keys),
                    "accepted_records": len(selected_keys),
                    "raw_records": counters["raw"],
                    "unique_records": counters["unique"],
                    "duplicate_records": counters["duplicate"],
                    "rejected_records": counters["rejected"],
                    "replaced_records": counters["replaced"],
                    "equal_records": counters["equal"],
                    "seed_file": str(seeds.relative_to(out_dir)),
                    "threads": threads,
                    "time_ms": args.time_ms,
                    "seconds": time.monotonic() - started,
                    "games": games,
                }
                manifest["accepted_shards"].append(accepted_entry)
                manifest.setdefault("games", []).extend(games)
                _atomic_json(out_dir / "manifest.json", manifest)
                _write_progress(out_dir, counts, index + 1, 0, "running",
                                last_shard=manifest["accepted_shards"][-1],
                                counters=store_counters)
                stage_v3.unlink(missing_ok=True)
                stage_meta.unlink(missing_ok=True)
                index += 1
                attempt = 0
            _write_progress(out_dir, counts, index, attempt, "complete",
                            counters=store_counters)
            return 0
        finally:
            store.close()


def _read_progress(out_dir: Path) -> dict:
    path = out_dir / "progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_progress(out_dir: Path, counts: TargetCounts, next_shard: int,
                    next_attempt: int, status: str, *, last_shard: Mapping[str, object] | None = None,
                    active_shard: Mapping[str, object] | None = None,
                    counters: Mapping[str, int] | None = None) -> None:
    counters = counters or {}
    payload = {
        "schema": SCHEMA,
        "status": status,
        "counts": {
            "total": counts.total,
            "central": counts.central,
            "broad": counts.broad,
            "family": counts.family,
        },
        "counters": {
            "raw": int(counters.get("raw", 0)),
            "unique": int(counters.get("unique", 0)),
            "duplicate": int(counters.get("duplicate", 0)),
            "rejected": int(counters.get("rejected", 0)),
            "replaced": int(counters.get("replaced", 0)),
            "equal": int(counters.get("equal", 0)),
        },
        "next_shard": next_shard,
        "next_attempt": next_attempt,
    }
    if last_shard is not None:
        payload["last_shard"] = dict(last_shard)
    if active_shard is not None:
        payload["active_shard"] = dict(active_shard)
    _atomic_json(out_dir / "progress.json", payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse controller arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CONFIG["out"], help="Output directory for accepted shards.")
    parser.add_argument("--exe", type=Path, default=CONFIG["exe"], help="Architecture-matched self-play executable.")
    parser.add_argument("--exe-arg", action="append", default=CONFIG["exe_arg"],
                        help="Argument that follows the self-play executable. Repeat this option.")
    parser.add_argument("--weights", type=Path, default=CONFIG["weights"], help="Architecture-matched quantized NNUE weights.")
    parser.add_argument("--central-schedule", type=Path, default=CONFIG["central_schedule"], help="Central schedule JSONL file.")
    parser.add_argument("--broad-schedule", type=Path, default=CONFIG["broad_schedule"], help="Broad schedule JSONL file.")
    parser.add_argument("--weakness-corpus", type=Path, action="append", default=CONFIG["weakness_corpus"],
                        help="Completed 100 ms weakness corpus. Repeat this option.")
    parser.add_argument("--total-target", type=int, default=CONFIG["total_target"])
    parser.add_argument("--central-target", type=int, default=CONFIG["central_target"])
    parser.add_argument("--broad-target", type=int, default=CONFIG["broad_target"])
    parser.add_argument("--family-floor", type=int, default=CONFIG["family_floor"])
    parser.add_argument("--games-per-shard", type=int, default=CONFIG["games_per_shard"])
    parser.add_argument("--seed-rows-per-shard", type=int, default=CONFIG["seed_rows_per_shard"])
    parser.add_argument("--time-ms", type=int, default=CONFIG["time_ms"])
    parser.add_argument("--max-plies", type=int, default=CONFIG["max_plies"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--fine-tune-pid", type=int, default=CONFIG["fine_tune_pid"])
    parser.add_argument("--threads-during-fine-tune", type=int, default=CONFIG["threads_during_fine_tune"])
    parser.add_argument("--threads-after-fine-tune", type=int, default=CONFIG["threads_after_fine_tune"])
    parser.add_argument("--fine-tune-threads", type=int, default=CONFIG["fine_tune_threads"])
    parser.add_argument("--cpu-thread-limit", type=int, default=CONFIG["cpu_thread_limit"])
    parser.add_argument("--reliability", type=float, default=CONFIG["reliability"],
                        help="Reliability for generated records.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction,
                        default=CONFIG["dry_run"], help="Print the resolved configuration and exit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line controller."""
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str), flush=True)
        return 0
    return run_controller(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[four-million-selfplay] ERROR: {error}", file=sys.stderr, flush=True)
        raise
