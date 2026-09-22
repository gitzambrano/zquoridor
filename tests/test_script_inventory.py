from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _source_files(*folders: str):
    for folder in folders:
        yield from (ROOT / folder).rglob("*")


def test_catalog_points_to_existing_public_runners():
    text = (ROOT / "docs" / "scripts.md").read_text(encoding="utf-8")
    for path in (
        "tools/run_benchmark.py",
        "tools/run_full_candidate_suite.py",
        "training/run_experiment.py",
        "training/run_teaching.py",
    ):
        assert path in text
        assert (ROOT / path).exists()


def test_deleted_status_is_not_referenced():
    for file in _source_files("docs", "src", "tools", "training", "tests", "benchmarks", "gui_web"):
        if "docs" in file.parts and "superpowers" in file.parts:
            continue
        if file.name == "test_script_inventory.py":
            continue
        if file.is_file() and file.suffix in {".md", ".hpp", ".cpp", ".py", ".bat", ".sh", ".ps1"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            assert "docs/status.md" not in text


def test_local_dataset_catalog_is_ignored():
    dataset_catalog = ROOT / "docs" / "datasets.md"
    assert dataset_catalog.exists()
    result = __import__("subprocess").run(
        ["git", "check-ignore", "--quiet", str(dataset_catalog)],
        cwd=ROOT,
    )
    assert result.returncode == 0
