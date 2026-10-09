import json
from pathlib import Path

import pytest

from tools import run_network_revalidation as revalidation


def test_expected_samples_match_paired_game_counts():
    assert revalidation._book_counts("h2h", "200ms") == {"center": 133, "normal": 67}
    assert revalidation._book_counts("external", "200ms") == {"center": 100, "normal": 50}
    assert revalidation._book_counts("h2h", "3plus2") == {"center": 100, "normal": 100}
    assert revalidation._book_counts("external", "3plus2") == {"center": 100, "normal": 100}


def test_three_plus_two_shards_partition_each_category(monkeypatch):
    def openings(_path, pairs, _seed):
        return [(index, [f"move-{index}"]) for index in range(pairs)]

    monkeypatch.setattr(revalidation.run_benchmark, "_read_openings", openings)
    shards = [revalidation._selected_books("h2h", "3plus2", index, 4) for index in range(4)]

    for category in ("center", "normal"):
        assert [len(shard[category]) for shard in shards] == [25, 25, 25, 25]
        flattened = [index for shard in shards for index, _moves in shard[category]]
        assert flattened == list(range(100))
    assert all(sum(len(openings) for openings in shard.values()) == 50 for shard in shards)


def test_wdl_and_paired_confidence_are_reported():
    rows = []
    for index, results in ((2, (1.0, 0.5)), (100003, (0.0, 1.0))):
        category = "center" if index < 100000 else "normal"
        for side, result in enumerate(results):
            rows.append({"opponent": "pre300", "opening_index": index, "zq_player": side,
                "status": "ok", "result": result, "plies": 20, "termination": "goal",
                "repeated_states": 0, "category": category})

    center = revalidation._rows_summary(rows, "center", expected_pairs=100)
    assert center["wdl"] == {"wins": 1, "draws": 1, "losses": 0}
    assert center["wdl_score_pct"] == 75.0
    assert center["complete_pairs"] == 1
    assert center["expected_pairs"] == 100
    assert center["sample_complete"] is False
    assert center["paired_bootstrap_95"] is not None
    assert center["strength_claim_ready"] is False


def test_53_percent_interval_crossing_50_is_inconclusive(monkeypatch):
    rows = []
    for index in range(2):
        for side in (0, 1):
            rows.append({"opening_index": index, "zq_player": side, "category": "center",
                "status": "ok", "result": 0.5, "plies": 20, "termination": "goal",
                "repeated_states": 0})

    def summary(_rows, *, bootstrap, seed):
        return {"failed_games": 0, "complete_pairs": 2,
            "paired_bootstrap_95": {"iterations": bootstrap, "seed": seed,
                "score_low_pct": 42.0, "score_high_pct": 64.0}}

    monkeypatch.setattr(revalidation.local_arena, "summarize_pairs", summary)
    stats = revalidation._rows_summary(rows, "center", expected_pairs=2)

    assert stats["sample_complete"] is True
    assert stats["evidence_status"] == "inconclusive"
    assert stats["strength_claim_ready"] is False


def test_external_summaries_never_mark_promotion_ready():
    rows = []
    categories = {"center": [(0, [])], "normal": [(0, [])]}
    for category, index in (("center", 0), ("normal", 100000)):
        for side in (0, 1):
            rows.append({"opponent": "pre300", "opening_index": index, "zq_player": side,
                "status": "ok", "result": 1.0, "plies": 20, "termination": "goal",
                "repeated_states": 0, "category": category})

    report = revalidation._summarize_run(rows, categories,
        {"mode": "external", "weight": "v300"})

    assert report["promotion_ready"] is False


def test_all_paired_delta_keeps_disjoint_center_and_normal_opening_ids():
    baseline = []
    candidate = []
    for index, category in ((4, "center"), (100006, "normal")):
        for side, old_result, new_result in ((0, 0.5, 1.0), (1, 0.5, 0.5)):
            base = {"opening_index": index, "zq_player": side, "category": category, "result": old_result}
            newer = {**base, "result": new_result}
            baseline.append(base)
            candidate.append(newer)

    delta = revalidation._paired_delta(candidate, baseline, None, seed=9)

    assert delta["complete_pairs"] == 2
    assert delta["delta_score_pp"] == 25.0
    assert delta["paired_bootstrap_95"]["zero_in_interval"] is False


def test_main_load_probe_uses_one_executable_for_every_weight(monkeypatch, tmp_path):
    captured = []

    class FakePlayer:
        def __init__(self, argv, name):
            captured.append((argv, name))

        def close(self):
            return None

    monkeypatch.setattr(revalidation.local_arena, "UciPlayer", FakePlayer)
    executable = tmp_path / "main.exe"
    weights = {name: {"path": tmp_path / f"{name}.bin", "sha256": name}
        for name in ("pre300", "v300", "v301")}

    assert revalidation.validate_main_weight_loads(executable, weights) == {
        name: name for name in weights
    }
    assert all(row[0][0] == str(executable) for row in captured)
    assert [row[1] for row in captured] == ["pre300", "v300", "v301"]


def test_manifest_validation_rejects_changed_identity(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "wrong", "weights": {}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unknown schema"):
        revalidation.validate_weight_manifest(path)


def test_shards_are_limited_to_four_three_plus_two_parts():
    revalidation._validate_shard("3plus2", 0, 4)
    revalidation._validate_shard("3plus2", 3, 4)
    with pytest.raises(ValueError):
        revalidation._validate_shard("200ms", 0, 4)
    with pytest.raises(ValueError):
        revalidation._validate_shard("3plus2", 4, 4)


def test_cli_has_no_low_sample_campaign_override():
    with pytest.raises(SystemExit):
        revalidation.parser().parse_args(["h2h", "--left", "v300", "--right", "pre300", "--pairs", "1"])
