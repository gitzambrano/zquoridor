import json

from tools import report_edge_acc_campaign_clock_v2 as report


def test_one_color_does_not_complete_an_opening_pair():
    row = {"status": "ok", "result": 1, "category": "center",
           "opening_index": 1, "zq_player": 0}
    assert report.category_game_summary([row], 100)["complete_pairs"] == 0
    assert report.category_game_summary(
        [row, {**row, "zq_player": 1, "result": 0}], 100)["complete_pairs"] == 1


def test_off_completion_and_candidate_scores_use_separate_samples(monkeypatch, tmp_path):
    overrides = {"overlay_root": tmp_path, "campaign_root": tmp_path,
                 "networks": ["504"], "variants": ["off", "delta_dense_only"],
                 "opponents": ["titanium"], "three_plus_two_pairs": {"center": 4, "normal": 4},
                 "three_plus_two_shards": 4, "bootstrap_iterations": 40}
    for key, value in overrides.items():
        monkeypatch.setitem(report.CONFIG, key, value)
    for variant, participant, result in (("off", "off", 0),
                                         ("delta_dense_only", "candidate", 1)):
        for shard in range(4):
            directory = tmp_path / "3plus2" / "504" / variant / "titanium" / f"shard_{shard:03d}_of_004"
            directory.mkdir(parents=True)
            rows = [{"status": "ok", "result": result, "participant": participant,
                     "category": category, "opening_index": shard, "zq_player": side}
                    for category in ("center", "normal") for side in (0, 1)]
            (directory / "games.jsonl").write_text("\n".join(map(json.dumps, rows)))
    inventory = report.three_plus_two_inventory()
    assert inventory["504/off/titanium"]["rankable"]
    candidate = inventory["504/delta_dense_only/titanium"]
    assert candidate["rankable"]
    assert candidate["results"]["center"]["score_pct"] == 100
    assert candidate["results"]["center"]["recorded_games"] == 8
