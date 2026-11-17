#!/usr/bin/env python3
"""Build the dataset inventory from the provenance records on disk.

Input:
    Every dataset directory and dataset file under the ``ROOTS`` directories.
    Each dataset has a provenance record. ``training/dataset_provenance.py``
    defines the record format.

Output:
    * ``docs/datasets.md``: the local inventory. The script replaces only the
      text between the two inventory markers. Git ignores this file.
    * ``data/dataset_index.json``: a compact index for Git. The index holds no
      timestamps, so it changes only when a dataset changes.

The script reports datasets without a valid record, sources that are absent
from disk, and historical catalog entries that are absent from disk. With
``--check``, the script exits with status 1 if a dataset has no valid record.
The script does not modify or delete datasets. The top-level ``CONFIG``
supplies defaults and CLI options override them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import dataset_provenance as provenance  # noqa: E402

CONFIG = {
    "roots": ["data/selfplay", "data/teaching"],
    "markdown": "docs/datasets.md",
    "index": "data/dataset_index.json",
    "historical_catalog": "data/selfplay/manifest.json",
    "check": False,
    "write": True,
}

BEGIN = "<!-- dataset-inventory:begin -->"
END = "<!-- dataset-inventory:end -->"
IGNORED_FILES = {"README.md", "manifest.json"}
METADATA_SUFFIXES = (".manifest.json", provenance.SUFFIX, ".sha256", ".tmp")
SAMPLE_KEYS = ("samples", "total_samples", "positions", "total_positions", "records")


def _human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"


def _tree_size(path: Path) -> tuple[int, int]:
    if path.is_file():
        return path.stat().st_size, 1
    total = count = 0
    for folder, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(folder) / name).stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def _is_metadata(name: str) -> bool:
    return name in IGNORED_FILES or name == provenance.FILENAME or name.endswith(METADATA_SUFFIXES)


def discover(root: Path) -> list[Path]:
    """Return the dataset directories and dataset files directly under ``root``."""
    if not root.is_dir():
        return []
    entries = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and not child.name.startswith("."):
            entries.append(child)
        elif child.is_file() and not _is_metadata(child.name):
            entries.append(child)
        elif child.name.endswith(".manifest.json"):
            data_file = child.with_name(child.name[: -len(".manifest.json")])
            if not data_file.exists():
                entries.append(data_file)
    return entries


def _manifest_samples(dataset: Path) -> int | None:
    candidates = [dataset.with_name(dataset.name + ".manifest.json")]
    if dataset.is_dir():
        candidates = sorted(dataset.glob("*manifest*.json"))
    for path in candidates:
        if not path.is_file() or path.stat().st_size > 8_000_000:
            continue
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in SAMPLE_KEYS:
            if isinstance(values.get(key), int):
                return values[key]
    return None


def describe(dataset: Path, root_name: str) -> dict:
    """Return the inventory entry for one dataset."""
    entry = {"root": root_name, "name": dataset.name, "present": dataset.exists(), "problems": []}
    entry["bytes"], entry["files"] = _tree_size(dataset) if dataset.exists() else (0, 0)
    record_file = provenance.record_path(dataset)
    try:
        record = provenance.read(dataset)
    except ValueError as error:
        record, entry["problems"] = None, [f"provenance file is not valid JSON: {error}"]
    if not entry["present"]:
        entry["problems"].append("manifest exists, but the data file is absent")
    if record is None:
        entry["problems"].append(f"no provenance record ({provenance.relative(record_file)})")
        entry.update(status="unknown", kind="", description="", generator="", sources=[],
                     samples=_manifest_samples(dataset), provenance_sha256=None)
        return entry
    entry["record_problems"] = provenance.validate(record)
    entry["problems"] += entry["record_problems"]
    contents = record.get("contents", {})
    generator = record.get("generator", {})
    entry.update(
        status=record.get("status", "unknown"),
        kind=record.get("kind", ""),
        description=record.get("description", ""),
        generator=generator.get("script", "") if isinstance(generator, dict) else str(generator),
        sources=[s.get("path", "") for s in record.get("sources", []) if isinstance(s, dict)],
        samples=contents.get("samples") if isinstance(contents.get("samples"), int) else _manifest_samples(dataset),
        provenance_sha256=hashlib.sha256(record_file.read_bytes()).hexdigest(),
    )
    entry["absent_sources"] = [s for s in entry["sources"] if s and not (ROOT / s).exists()]
    return entry


def historical_absent(catalog: Path, present: set[str]) -> list[dict]:
    """Return the historical catalog entries that have no directory on disk."""
    if not catalog.is_file():
        return []
    values = json.loads(catalog.read_text(encoding="utf-8")).get("datasets", {})
    return [{"name": name, "records": info.get("records"), "description": info.get("description", "")}
            for name, info in sorted(values.items()) if name not in present]


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(entries: list[dict], absent_history: list[dict], roots: list[str]) -> str:
    """Return the generated Markdown region."""
    lines = [BEGIN, "", "This section is generated by `python tools/dataset_inventory.py`.",
             "Do not edit it by hand. Edit the provenance record of the dataset instead.", ""]
    for root_name in roots:
        rows = [e for e in entries if e["root"] == root_name]
        lines += [f"## `{root_name}/`", ""]
        if not rows:
            lines += ["The directory contains no datasets.", ""]
            continue
        lines += ["| Dataset | Status | Kind | Samples | Size | Generator | Sources | Description |",
                  "| :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- |"]
        for e in rows:
            samples = f"{e['samples']:,}" if isinstance(e.get("samples"), int) else ""
            sources = "<br>".join(f"`{s}`" + (" (absent)" if s in e.get("absent_sources", []) else "")
                                  for s in e["sources"])
            generator = f"`{e['generator']}`" if e["generator"] else ""
            lines.append(f"| `{e['name']}` | {e['status']} | {e['kind']} | {samples} | "
                         f"{_human_bytes(e['bytes'])} | {generator} | {sources} | {_cell(e['description'])} |")
        lines.append("")
    if absent_history:
        lines += ["## Historical catalog entries absent from disk", "",
                  "`data/selfplay/manifest.json` lists these generations. Their directories are absent.", "",
                  "| Generation | Records | Description |", "| :--- | ---: | :--- |"]
        for item in absent_history:
            records = f"{item['records']:,}" if isinstance(item.get("records"), int) else ""
            lines.append(f"| `{item['name']}` | {records} | {_cell(item['description'])} |")
        lines.append("")
    problems = [(e, p) for e in entries for p in e["problems"]]
    lines += ["## Problems", ""]
    lines += [f"- `{e['root']}/{e['name']}`: {p}." for e, p in problems] or ["No dataset has a problem."]
    lines += ["", END]
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, region: str) -> None:
    """Replace the generated region of ``path``, or create the file."""
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            tail = rest.split(END, 1)[1].lstrip("\n")
            path.write_text(head + region + ("\n" + tail if tail else ""), encoding="utf-8", newline="\n")
            return
        path.write_text(text.rstrip("\n") + "\n\n" + region, encoding="utf-8", newline="\n")
        return
    path.write_text("# Local dataset inventory\n\n" + region, encoding="utf-8", newline="\n")


def build_index(entries: list[dict], absent_history: list[dict]) -> dict:
    """Return the compact index that Git tracks."""
    keep = ("root", "name", "status", "kind", "samples", "generator", "sources",
            "description", "provenance_sha256")
    return {
        "schema": "zquoridor.dataset_index.v1",
        "generator": "tools/dataset_inventory.py",
        "datasets": [{key: e.get(key) for key in keep} for e in entries if e["present"]],
        "absent_historical_generations": [item["name"] for item in absent_history],
    }


def run(config: dict) -> int:
    entries = [describe(path, root_name) for root_name in config["roots"]
               for path in discover(ROOT / root_name)]
    present = {e["name"] for e in entries if e["present"]}
    absent_history = historical_absent(ROOT / config["historical_catalog"], present)
    if config["write"]:
        write_markdown(ROOT / config["markdown"], render(entries, absent_history, config["roots"]))
        index = ROOT / config["index"]
        index.write_text(json.dumps(build_index(entries, absent_history), indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")
    failing = [e for e in entries if e["problems"]]
    print(f"Datasets: {len(entries)} | with problems: {len(failing)} | "
          f"absent historical generations: {len(absent_history)}")
    for e in failing:
        for problem in e["problems"]:
            print(f"  {e['root']}/{e['name']}: {problem}")
    invalid = [e for e in entries if e.get("provenance_sha256") is None or e.get("record_problems")]
    return 1 if config["check"] and invalid else 0


def parse_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=json.loads, default=argparse.SUPPRESS,
                        help='JSON list of dataset roots, for example ["data/teaching"]')
    for key in ("markdown", "index", "historical_catalog"):
        parser.add_argument("--" + key.replace("_", "-"), default=argparse.SUPPRESS)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--write", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    config = dict(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def main(argv=None) -> int:
    return run(parse_config(argv))


if __name__ == "__main__":
    sys.exit(main())
