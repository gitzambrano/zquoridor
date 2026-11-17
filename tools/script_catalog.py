#!/usr/bin/env python3
"""Generate the script index of ``docs/scripts.md`` from the tracked sources.

Input:
    Every Python file that Git tracks under ``tools/``, ``training/``, and
    ``colab/``. Tests, package files, and frozen snapshots are excluded.

Output:
    The text between the two script-index markers in ``docs/scripts.md``.
    Each row gives the path, the role, a link to a detailed section if one
    exists, and the first sentence of the module docstring.

Roles:
    * ``runner``: a module with a top-level ``CONFIG`` and a ``main`` function.
    * ``campaign``: a script for one specific dataset or campaign.
    * ``diagnostic``: a historical inspection or repair script.
    * ``module``: a library or a helper that other scripts import.

With ``--check``, the script writes nothing. It exits with status 1 if the
index is out of date. The top-level ``CONFIG`` supplies defaults and CLI
options override them.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "doc": "docs/scripts.md",
    "roots": ["tools", "training", "colab"],
    "check": False,
}

BEGIN = "<!-- script-index:begin -->"
END = "<!-- script-index:end -->"
CAMPAIGN_DIRS = ("tools/campaigns/",)
CAMPAIGN_FILES = ("tools/release_410.py", "tools/run_v3_search_screening.py",
                  "tools/run_v3_search_confirmation.py")
DIAGNOSTIC_DIRS = ("colab/playwright/diagnostics/",)
EXCLUDED_DIRS = ("experimental/", "tools/path_clash_bot_arena/")


def tracked_scripts(roots: list[str]) -> list[str]:
    """Return the tracked Python sources under ``roots`` in path order."""
    out = subprocess.run(["git", "ls-files", "--", *[f"{r}/*.py" for r in roots]],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    keep = []
    for path in out:
        name = path.rsplit("/", 1)[-1]
        if name == "__init__.py" or name.startswith("test_") or "/tests/" in path:
            continue
        if path.startswith(EXCLUDED_DIRS) or not (ROOT / path).is_file():
            continue
        keep.append(path)
    return sorted(keep)


def summary(tree: ast.Module) -> str:
    """Return the first sentence of the module docstring."""
    doc = ast.get_docstring(tree)
    if not doc:
        return "No module docstring."
    paragraph = " ".join(doc.strip().split("\n\n", 1)[0].split())
    match = re.match(r"(.+?[.!?])(\s|$)", paragraph)
    return (match.group(1) if match else paragraph).replace("|", "\\|")


def role(path: str, tree: ast.Module) -> str:
    """Return the role of one script."""
    if path.startswith(CAMPAIGN_DIRS) or path in CAMPAIGN_FILES:
        return "campaign"
    if path.startswith(DIAGNOSTIC_DIRS):
        return "diagnostic"
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(getattr(t, "id", None) for t in node.targets)
        elif isinstance(node, ast.AnnAssign):
            names.add(getattr(node.target, "id", None))
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return "runner" if {"CONFIG", "main"} <= names else "module"


def _anchor(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.lower())
    return text.strip().replace(" ", "-")


def detail_headings(text: str) -> dict[str, str]:
    """Return the detailed ``### `path``` sections outside the generated index."""
    if BEGIN in text:
        text = text.split(BEGIN, 1)[0] + text.split(END, 1)[-1]
    found = {}
    for line in text.splitlines():
        match = re.match(r"###\s+`([^`]+\.py)`\s*$", line)
        if match:
            found[match.group(1)] = _anchor(line.lstrip("#").strip())
    return found


def render(paths: list[str], details: dict[str, str]) -> str:
    """Return the generated index region."""
    lines = [BEGIN, "", "## Script index", "",
             "`python tools/script_catalog.py` generates this section. Do not edit it by hand.",
             "Edit the module docstring of the script instead.", ""]
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(path.rsplit("/", 1)[0], []).append(path)
    counts: dict[str, int] = {}
    for folder in sorted(groups):
        lines += [f"### `{folder}/`", "", "| Script | Role | Details | Summary |",
                  "| :--- | :--- | :--- | :--- |"]
        for path in groups[folder]:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
            kind = role(path, tree)
            counts[kind] = counts.get(kind, 0) + 1
            link = f"[section](#{details[path]})" if path in details else ""
            lines.append(f"| `{path.rsplit('/', 1)[1]}` | {kind} | {link} | {summary(tree)} |")
        lines.append("")
    totals = ", ".join(f"{counts[k]} {k}" for k in sorted(counts))
    lines[7:7] = [f"The index lists {len(paths)} scripts: {totals}.", ""]
    lines.append(END)
    return "\n".join(lines) + "\n"


def updated_text(text: str, region: str) -> str:
    """Return ``text`` with the generated region replaced or appended."""
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1].lstrip("\n")
        return head + region + ("\n" + tail if tail else "")
    return text.rstrip("\n") + "\n\n" + region


def run(config: dict) -> int:
    doc = ROOT / config["doc"]
    text = doc.read_text(encoding="utf-8")
    paths = tracked_scripts(config["roots"])
    new_text = updated_text(text, render(paths, detail_headings(text)))
    if config["check"]:
        if new_text != text:
            print(f"{config['doc']} script index is out of date. Run python tools/script_catalog.py.")
            return 1
        print(f"{config['doc']} script index is current ({len(paths)} scripts).")
        return 0
    doc.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"Wrote the script index for {len(paths)} scripts to {config['doc']}.")
    return 0


def parse_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", default=argparse.SUPPRESS)
    parser.add_argument("--roots", nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    config = dict(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def main(argv=None) -> int:
    return run(parse_config(argv))


if __name__ == "__main__":
    sys.exit(main())
