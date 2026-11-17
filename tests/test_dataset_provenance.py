import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import dataset_provenance as dp  # noqa: E402
from tools import dataset_inventory as inv  # noqa: E402


def _write(tmp_path, folder, **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset.npz").write_bytes(b"x" * 10)
    values = dict(kind="teaching", description="Test dataset.", sources=[{"path": tmp_path / "src.npz", "role": "input"}],
                  contents={"samples": 10}, generator="tools/example.py", root=tmp_path)
    values.update(overrides)
    return dp.write(folder, **values)


def test_write_records_relative_sources_and_validates(tmp_path):
    path = _write(tmp_path, tmp_path / "data" / "teaching" / "set_a")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "provenance.json"
    assert record["sources"] == [{"path": "src.npz", "role": "input"}]
    assert record["generator"]["script"] == "tools/example.py"
    assert dp.validate(record) == []


def test_validate_rejects_absolute_source_and_bad_status():
    record = {"schema": dp.SCHEMA, "name": "x", "kind": "teaching", "status": "done", "description": "",
              "generator": {}, "sources": [{"path": "C:/data/x.npz"}], "contents": {}}
    problems = dp.validate(record)
    assert any("status" in p for p in problems)
    assert any("absolute" in p for p in problems)


def test_file_dataset_uses_sibling_record(tmp_path):
    data_file = tmp_path / "positions.jsonl"
    data_file.write_text("{}\n", encoding="utf-8")
    assert dp.record_path(data_file).name == "positions.jsonl.provenance.json"


def test_inventory_reports_missing_records_and_absent_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "ROOT", tmp_path)
    teaching = tmp_path / "data" / "teaching"
    _write(tmp_path, teaching / "with_record")
    (teaching / "without_record").mkdir()
    (teaching / "orphan.npz.manifest.json").write_text('{"samples": 3}', encoding="utf-8")

    config = dict(inv.CONFIG, roots=["data/teaching"], markdown="datasets.md", index="index.json",
                  historical_catalog="none.json", check=True)
    assert inv.run(config) == 1

    text = (tmp_path / "datasets.md").read_text(encoding="utf-8")
    assert inv.BEGIN in text and inv.END in text
    assert "`src.npz` (absent)" in text
    assert "without_record" in text and "no provenance record" in text
    assert "data file is absent" in text

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    names = {d["name"] for d in index["datasets"]}
    assert names == {"with_record", "without_record"}


def test_inventory_keeps_hand_written_text(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "ROOT", tmp_path)
    _write(tmp_path, tmp_path / "data" / "teaching" / "only")
    md = tmp_path / "datasets.md"
    md.write_text(f"# Title\n\nHand text.\n\n{inv.BEGIN}\nold\n{inv.END}\n\nTail text.\n", encoding="utf-8")
    config = dict(inv.CONFIG, roots=["data/teaching"], markdown="datasets.md", index="index.json",
                  historical_catalog="none.json")
    assert inv.run(config) == 0
    text = md.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nHand text.")
    assert "Tail text." in text and "\nold\n" not in text


def test_repository_datasets_have_valid_records_when_present():
    if not (ROOT / "data" / "teaching").is_dir():
        pytest.skip("local datasets are absent")
    entries = [inv.describe(p, r) for r in inv.CONFIG["roots"] for p in inv.discover(ROOT / r)]
    missing = [e["name"] for e in entries if any("no provenance record" in p for p in e["problems"])]
    assert missing == []
