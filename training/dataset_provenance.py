"""Read, write, and validate dataset provenance records.

A provenance record states the origin and the contents of one dataset. The
record is a JSON file beside the data:

* A dataset directory holds ``provenance.json``.
* A single dataset file ``<file>`` has ``<file>.provenance.json``.

The record travels with the data when an operator copies the dataset. Paths in
the record are relative to the repository root and use forward slashes.
``tools/dataset_inventory.py`` reads every record and builds the inventory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "zquoridor.dataset_provenance.v1"
FILENAME = "provenance.json"
SUFFIX = ".provenance.json"

KINDS = ("selfplay", "teaching", "positions", "targets")
STATUSES = ("active", "archived", "partial", "scratch")
REQUIRED = ("schema", "name", "kind", "status", "description", "generator", "sources", "contents")


def record_path(dataset: Path) -> Path:
    """Return the provenance file path for a dataset directory or file."""
    dataset = Path(dataset)
    return dataset / FILENAME if dataset.is_dir() else dataset.with_name(dataset.name + SUFFIX)


def relative(path, root: Path = ROOT) -> str:
    """Return ``path`` relative to ``root`` with forward slashes, if possible."""
    text = str(path).replace("\\", "/")
    try:
        return Path(text).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return text


def git_commit(root: Path = ROOT) -> str | None:
    """Return the current commit hash, or ``None`` outside a Git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def validate(record: dict) -> list[str]:
    """Return the problems in one record. An empty list means a valid record."""
    problems = [f"missing field '{key}'" for key in REQUIRED if key not in record]
    if record.get("schema") not in (None, SCHEMA):
        problems.append(f"unknown schema '{record.get('schema')}'")
    if "kind" in record and record["kind"] not in KINDS:
        problems.append(f"kind must be one of {list(KINDS)}")
    if "status" in record and record["status"] not in STATUSES:
        problems.append(f"status must be one of {list(STATUSES)}")
    if not isinstance(record.get("sources", []), list):
        problems.append("sources must be a list")
    if not isinstance(record.get("contents", {}), dict):
        problems.append("contents must be an object")
    for source in record.get("sources", []) if isinstance(record.get("sources"), list) else []:
        path = source.get("path", "") if isinstance(source, dict) else ""
        if not path:
            problems.append("each source needs a 'path'")
        elif ":" in path[:3] or path.startswith("/"):
            problems.append(f"source path is absolute: {path}")
    return problems


def read(dataset: Path) -> dict | None:
    """Return the record for ``dataset``, or ``None`` if the file is absent."""
    path = record_path(dataset)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write(dataset: Path, *, kind: str, description: str, sources: list, contents: dict,
          parameters: dict | None = None, status: str = "active", generator: str | None = None,
          notes: str = "", root: Path = ROOT) -> Path:
    """Write the provenance record of a dataset that a generator just produced.

    ``sources`` holds one object per input with a ``path`` and an optional
    ``role``. ``generator`` defaults to the running script. The function adds
    the command line, the commit, and the UTC time.
    """
    dataset = Path(dataset)
    script = generator or relative(Path(sys.argv[0]).resolve(), root)
    record = {
        "schema": SCHEMA,
        "name": dataset.name,
        "kind": kind,
        "status": status,
        "description": description,
        "generator": {
            "script": script,
            "command": [script, *sys.argv[1:]],
            "commit": git_commit(root),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "sources": [dict(source, path=relative(source["path"], root)) for source in sources],
        "contents": contents,
        "parameters": parameters or {},
        "notes": notes,
    }
    problems = validate(record)
    if problems:
        raise ValueError(f"invalid provenance for {dataset}: {problems}")
    path = record_path(dataset)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
