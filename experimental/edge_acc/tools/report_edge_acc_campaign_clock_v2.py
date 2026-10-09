#!/usr/bin/env python3
"""Write a consolidated report from stored EdgeAcc campaign results."""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "benchmarks" / "edge_acc_campaign_clock_v2"
OVERLAY = ROOT / "results" / "benchmarks" / "edge_acc_campaign_clock_v3"
CONFIG = {
    "campaign_root": CAMPAIGN,
    "overlay_root": OVERLAY,
    "output_markdown": OVERLAY / "STATUS_TABLE.md",
    "output_json": OVERLAY / "MATRIX.json",
    "networks": ["504", "858"],
    "variants": ["off", "delta_dense_only", "v3_node_dense_bfs", "v3"],
    "opponents": ["claustrophobia", "titanium"],
    "h2h_pairs": {"all": 200, "center": 133, "normal": 67},
    "external_pairs_200ms": {"center": 100, "normal": 50},
    "three_plus_two_pairs": {"center": 100, "normal": 100},
    "three_plus_two_shards": 4,
    "bootstrap_iterations": 20000,
    "bootstrap_seed": 20261006,
    "parity_paths": [
        ROOT / "results" / "edge-parity-local" / "parity_manifest.json",
        ROOT / "results" / "benchmarks" / "edge-parity-local" / "parity_manifest.json",
    ],
    "speed_paths": [
        ROOT / "results" / "edge-parity-local" / "isolated_speed.json",
        ROOT / "results" / "benchmarks" / "edge-parity-local" / "isolated_speed.json",
    ],
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def elo(score):
    if score is None or score <= 0 or score >= 100:
        return None
    return 400.0 * math.log10(score / (100.0 - score))


def compact_summary(summary):
    if not isinstance(summary, dict):
        return {"status": "missing", "recorded_games": 0, "failed_games": 0,
                "complete_pairs": 0, "expected_pairs": None, "score_pct": None,
                "elo": None, "ci95_score_pct": None, "ci95_elo": None}
    ci = summary.get("paired_bootstrap_95") or {}
    return {
        "status": "complete" if summary.get("failed_games") == 0 else "incomplete",
        "recorded_games": summary.get("recorded_games", 0),
        "failed_games": summary.get("failed_games", 0),
        "complete_pairs": summary.get("complete_pairs", 0),
        "expected_pairs": None,
        "score_pct": summary.get("score_pct"), "elo": summary.get("elo"),
        "ci95_score_pct": [ci.get("score_low_pct"), ci.get("score_high_pct")]
        if ci.get("score_low_pct") is not None else None,
        "ci95_elo": [ci.get("elo_low"), ci.get("elo_high")]
        if ci.get("elo_low") is not None else None,
    }


def theoretical_h2h_reference():
    return {"status": "theoretical_reference", "theoretical_reference": True,
            "recorded_games": None, "failed_games": 0, "complete_pairs": None,
            "expected_pairs": None, "score_pct": 50.0, "elo": 0.0,
            "ci95_score_pct": None, "ci95_elo": None,
            "sample_note": "No observed self-play sample"}


def load_games(paths):
    rows = []
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    return rows


def latest_game_rows(rows, include_participant=False):
    latest = {}
    for row in rows:
        key = [row.get("category"), row.get("opening_index"), row.get("zq_player")]
        if include_participant:
            key.append(row.get("participant"))
        latest[tuple(key)] = row
    return list(latest.values())


def paired_ci(values):
    if not values:
        return None
    rng = random.Random(CONFIG["bootstrap_seed"])
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(CONFIG["bootstrap_iterations"]))
    return [means[int(0.025 * len(means))], means[min(len(means) - 1, int(0.975 * len(means)))]]


def category_game_summary(rows, expected, paired_delta=None):
    valid = [r for r in rows if r.get("status") == "ok" and r.get("result") is not None]
    outcomes = [float(r["result"]) for r in valid]
    opening_scores = {}
    for row in valid:
        opening_scores.setdefault((row.get("category"), row.get("opening_index")), []).append(float(row["result"]))
    opening_values = [sum(scores) / len(scores) for scores in opening_scores.values() if len(scores) >= 2]
    failures = sum(r.get("status") != "ok" for r in rows)
    crashes = sum(r.get("termination") == "engine_error" for r in rows)
    time_losses = sum(r.get("termination") == "time" for r in rows)
    ci = paired_ci(opening_values) if len(opening_values) == expected else None
    return {"recorded_games": len(rows), "failed_games": failures,
            "complete_pairs": len(opening_values), "expected_pairs": expected,
            "score_pct": (100 * sum(outcomes) / len(outcomes)) if outcomes else None,
            "elo": elo(100 * sum(outcomes) / len(outcomes)) if outcomes else None,
            "ci95_score_pct": [100 * x for x in ci] if ci else None,
            "ci95_elo": [elo(100 * x) for x in ci] if ci else None,
            "crash_or_forfeit_games": crashes, "time_loss_games": time_losses}


def paired_external_delta(rows, category, expected):
    buckets = {}
    for row in rows:
        if row.get("category") != category or row.get("status") != "ok" or row.get("result") is None:
            continue
        key = (row.get("opening_index"), row.get("zq_player"))
        buckets.setdefault(key, {})[row.get("participant")] = float(row["result"])
    paired = {}
    for (opening, _side), participants in buckets.items():
        if "candidate" in participants and "off" in participants:
            paired.setdefault(opening, []).append(participants["candidate"] - participants["off"])
    deltas = [sum(values) / len(values) for values in paired.values() if len(values) >= 2]
    ci = paired_ci(deltas) if len(deltas) == expected else None
    candidate_scores = [float(r["result"]) for r in rows if r.get("category") == category and
                        r.get("participant") == "candidate" and r.get("status") == "ok" and r.get("result") is not None]
    off_scores = [float(r["result"]) for r in rows if r.get("category") == category and
                  r.get("participant") == "off" and r.get("status") == "ok" and r.get("result") is not None]
    candidate_pct = 100 * sum(candidate_scores) / len(candidate_scores) if candidate_scores else None
    off_pct = 100 * sum(off_scores) / len(off_scores) if off_scores else None
    return {"complete_paired_openings": len(deltas), "expected_pairs": expected,
            "delta_score_pct": 100 * sum(deltas) / len(deltas) if deltas else None,
            "candidate_score_pct": candidate_pct, "baseline_score_pct": off_pct,
            "delta_elo_estimate": (elo(candidate_pct) - elo(off_pct)
                                   if candidate_pct is not None and off_pct is not None and
                                   elo(candidate_pct) is not None and elo(off_pct) is not None else None),
            "paired_bootstrap_95": [100 * x for x in ci] if ci else None}


def paired_field_delta(candidate_rows, baseline_rows, category, expected, field="weight",
                       candidate_label=None, baseline_label=None):
    outcomes = {}
    for label, rows in ((candidate_label, candidate_rows), (baseline_label, baseline_rows)):
        for row in rows:
            if row.get("category") != category or row.get("status") != "ok" or row.get("result") is None:
                continue
            key = (row.get("opening_index"), row.get("zq_player"))
            outcomes.setdefault(key, {})[label] = float(row["result"])
    paired = {}
    for (opening, _side), row_outcomes in outcomes.items():
        if candidate_label in row_outcomes and baseline_label in row_outcomes:
            paired.setdefault(opening, []).append(row_outcomes[candidate_label] - row_outcomes[baseline_label])
    deltas = [sum(values) / len(values) for values in paired.values() if len(values) >= 2]
    ci = paired_ci(deltas) if len(deltas) == expected else None
    candidate_scores = [float(r["result"]) for r in candidate_rows if r.get("category") == category and
                        r.get("status") == "ok" and r.get("result") is not None]
    baseline_scores = [float(r["result"]) for r in baseline_rows if r.get("category") == category and
                       r.get("status") == "ok" and r.get("result") is not None]
    candidate_pct = 100 * sum(candidate_scores) / len(candidate_scores) if candidate_scores else None
    baseline_pct = 100 * sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
    return {"complete_paired_openings": len(deltas), "expected_pairs": expected,
            "delta_score_pct": 100 * sum(deltas) / len(deltas) if deltas else None,
            "candidate_score_pct": candidate_pct, "baseline_score_pct": baseline_pct,
            "delta_elo_estimate": (elo(candidate_pct) - elo(baseline_pct)
                                   if candidate_pct is not None and baseline_pct is not None and
                                   elo(candidate_pct) is not None and elo(baseline_pct) is not None else None),
            "paired_bootstrap_95": [100 * x for x in ci] if ci else None}


def three_plus_two_inventory():
    base = Path(CONFIG["overlay_root"]) / "3plus2"
    inventory = {}
    for network in CONFIG["networks"]:
        for variant in CONFIG["variants"]:
            for opponent in ["h2h", *CONFIG["opponents"]]:
                if opponent == "h2h":
                    dirs = [base / network / variant / f"h2h_shard_{i:03d}_of_004"
                            for i in range(CONFIG["three_plus_two_shards"])]
                else:
                    dirs = [base / network / variant / opponent / f"shard_{i:03d}_of_004"
                            for i in range(CONFIG["three_plus_two_shards"])]
                present = [d for d in dirs if (d / "games.jsonl").exists()]
                off_dirs = []
                off_present = []
                if opponent != "h2h" and variant != "off":
                    off_base = base / network / "off" / opponent
                    off_dirs = [off_base / f"shard_{i:03d}_of_004"
                                for i in range(CONFIG["three_plus_two_shards"])]
                    off_present = [d for d in off_dirs if (d / "games.jsonl").exists()]
                raw_paths = [d / "games.jsonl" for d in present + off_present]
                raw_rows = load_games(raw_paths)
                rows = latest_game_rows(raw_rows, include_participant=opponent != "h2h")
                failures = sum(1 for row in rows if row.get("status") != "ok")
                by_category = {}
                categories = ("center", "normal")
                for category in categories:
                    expected = CONFIG["three_plus_two_pairs"][category]
                    group = [r for r in rows if r.get("category") == category]
                    primary = ("off" if variant == "off" else "candidate")
                    primary_rows = (group if opponent == "h2h" else
                                    [r for r in group if r.get("participant") == primary])
                    summary = category_game_summary(primary_rows, expected)
                    if opponent != "h2h":
                        summary["participants"] = {
                            participant: category_game_summary(
                                [r for r in group if r.get("participant") == participant], expected)
                            for participant in ("off", "candidate")}
                        summary["paired_delta_candidate_minus_off"] = paired_external_delta(rows, category, expected)
                    by_category[category] = summary
                if opponent == "h2h":
                    by_category["all"] = category_game_summary(
                        rows, sum(CONFIG["three_plus_two_pairs"].values()))
                crash_count = sum(r.get("termination") == "engine_error" for r in rows)
                time_count = sum(r.get("termination") == "time" for r in rows)
                complete = (len(present) == CONFIG["three_plus_two_shards"] and
                            (opponent == "h2h" or variant == "off" or len(off_present) == CONFIG["three_plus_two_shards"]) and
                            failures == 0 and crash_count == 0 and
                            all(v["complete_pairs"] == v["expected_pairs"] for v in by_category.values()))
                if opponent != "h2h":
                    required_sides = ("off",) if variant == "off" else ("off", "candidate")
                    complete = complete and all(
                        all(by_category[cat]["participants"][side]["complete_pairs"] == by_category[cat]["expected_pairs"] and
                            by_category[cat]["participants"][side]["failed_games"] == 0
                            for side in required_sides) and
                        (variant == "off" or
                         by_category[cat]["paired_delta_candidate_minus_off"]["complete_paired_openings"] == by_category[cat]["expected_pairs"])
                        for cat in categories)
                inventory[f"{network}/{variant}/{opponent}"] = {
                    "status": "complete" if complete else ("incomplete" if present else "missing"),
                    "shards_present": len(present), "shards_expected": CONFIG["three_plus_two_shards"],
                    "off_shards_present": len(off_present),
                    "raw_attempts": len(raw_rows), "latest_unique_games": len(rows),
                    "raw_failed_attempts": sum(1 for row in raw_rows if row.get("status") != "ok"),
                    "failed_games": failures, "crash_or_forfeit_games": crash_count,
                    "time_loss_games": time_count, "results": by_category,
                    "rankable": complete,
                }
    # Retain the v2 failed-shard audit, with raw attempts and latest unique outcomes.
    net_dir = Path(CONFIG["campaign_root"]) / "network_revalidation" / "h2h" / "3plus2" / "v300_vs_pre300"
    raw_net_rows = load_games([net_dir / f"shard_{i:03d}_of_004" / "games.jsonl"
                               for i in range(CONFIG["three_plus_two_shards"])])
    net_rows = latest_game_rows(raw_net_rows)
    failures = [r for r in net_rows if r.get("status") != "ok"]
    inventory["network_revalidation/v300_vs_pre300/h2h"] = {
        "status": "incomplete" if any((net_dir / f"shard_{i:03d}_of_004" / "games.jsonl").exists()
                                       for i in range(CONFIG["three_plus_two_shards"])) else "missing",
        "shards_present": sum((net_dir / f"shard_{i:03d}_of_004" / "games.jsonl").exists()
                               for i in range(CONFIG["three_plus_two_shards"])),
        "shards_expected": CONFIG["three_plus_two_shards"],
        "raw_attempts": len(raw_net_rows), "raw_failed_attempts": sum(r.get("status") != "ok" for r in raw_net_rows),
        "latest_unique_games": len(net_rows), "failed_games": len(failures),
        "error_rows": [{k: row.get(k) for k in ("category", "opening_index", "left", "right", "status", "termination", "error", "zq_player")}
                       for row in failures],
        "rankable": False,
    }
    cross_base = Path(CONFIG["overlay_root"]) / "cross_network_h2h_3plus2" / "hybrid858_vs_main504_off"
    cross_dirs = [cross_base / f"shard_{i:03d}_of_004" for i in range(CONFIG["three_plus_two_shards"])]
    cross_present = [d for d in cross_dirs if (d / "games.jsonl").exists()]
    cross_raw = load_games([d / "games.jsonl" for d in cross_present])
    cross_rows = latest_game_rows(cross_raw)
    cross_result = {cat: category_game_summary([r for r in cross_rows if r.get("category") == cat],
                                                CONFIG["three_plus_two_pairs"][cat])
                    for cat in ("center", "normal")}
    cross_crashes = sum(r.get("termination") == "engine_error" for r in cross_rows)
    inventory["cross_network/hybrid858_vs_main504_off"] = {
        "status": "complete" if len(cross_present) == CONFIG["three_plus_two_shards"] and
                  all(cross_result[c]["complete_pairs"] == cross_result[c]["expected_pairs"] and
                      cross_result[c]["failed_games"] == 0 for c in cross_result) and cross_crashes == 0
                  else ("incomplete" if cross_present else "missing"),
        "shards_present": len(cross_present), "shards_expected": CONFIG["three_plus_two_shards"],
        "raw_attempts": len(cross_raw), "raw_failed_attempts": sum(r.get("status") != "ok" for r in cross_raw),
        "latest_unique_games": len(cross_rows),
        "crash_or_forfeit_games": cross_crashes, "results": cross_result,
        "rankable": False,
    }
    return inventory


def parity_and_speed():
    parity = next((read_json(p) for p in CONFIG["parity_paths"] if read_json(p)), None)
    speed = next((read_json(p) for p in CONFIG["speed_paths"] if read_json(p)), None)
    return parity, speed


def build_matrix():
    matrix = {"campaign": "edge_acc_campaign_clock_v2_plus_v3_overlay", "networks": {}, "three_plus_two": three_plus_two_inventory(),
              "network_revalidation": {}}
    for network in CONFIG["networks"]:
        matrix["networks"][network] = {}
        for variant in CONFIG["variants"]:
            root = Path(CONFIG["campaign_root"]) / "200ms" / network / variant
            h2h_source = read_json(Path(CONFIG["campaign_root"]) / f"h2h_{network}_{variant}" / "summary.json")
            h2h = theoretical_h2h_reference() if variant == "off" else compact_summary(h2h_source)
            if variant != "off":
                h2h["expected_pairs"] = CONFIG["h2h_pairs"]["all"]
            h2h_books = {}
            for category in ("center", "normal"):
                source = (read_json(Path(CONFIG["campaign_root"]) / f"h2h_{network}_{variant}" / "summary.json") or {})
                if variant == "off":
                    value = theoretical_h2h_reference()
                else:
                    value = source.get(category)
                h2h_books[category] = compact_summary(value)
                if variant == "off":
                    h2h_books[category] = theoretical_h2h_reference()
                else:
                    h2h_books[category]["expected_pairs"] = CONFIG["h2h_pairs"][category]
            ext = {}
            for opponent in CONFIG["opponents"]:
                summary = read_json(root / opponent / "summary.json")
                off_summary = read_json(Path(CONFIG["campaign_root"]) / "200ms" / network / "off" / opponent / "summary.json")
                delta_source = (summary or {}).get("paired_delta_candidate_minus_off") or {}
                item = {"opponent": opponent, "time_control": "200ms", "participants": {},
                        "paired_delta_candidate_minus_off": delta_source,
                        "off_absolute": {}}
                for participant, data in (summary or {}).get("participants", {}).items():
                    item["participants"][participant] = {}
                    for category in ("center", "normal"):
                        s = compact_summary(data.get(category))
                        s["expected_pairs"] = CONFIG["external_pairs_200ms"][category]
                        item["participants"][participant][category] = s
                for participant, data in (off_summary or {}).get("participants", {}).items():
                    if participant != "off":
                        continue
                    item["off_absolute"] = {}
                    for category in ("center", "normal"):
                        s = compact_summary(data.get(category))
                        s["expected_pairs"] = CONFIG["external_pairs_200ms"][category]
                        item["off_absolute"][category] = s
                for category, delta in item["paired_delta_candidate_minus_off"].items():
                    candidate_score = item.get("participants", {}).get("candidate", {}).get(category, {}).get("score_pct")
                    baseline_score = item.get("off_absolute", {}).get(category, {}).get("score_pct")
                    delta["candidate_score_pct"] = candidate_score
                    delta["baseline_score_pct"] = baseline_score
                    delta["delta_elo_estimate"] = (elo(candidate_score) - elo(baseline_score)
                                                   if candidate_score is not None and baseline_score is not None and
                                                   elo(candidate_score) is not None and elo(baseline_score) is not None else None)
                ext[opponent] = item
            matrix["networks"][network][variant] = {"h2h_200ms_vs_off": h2h,
                                                     "h2h_200ms_by_category": h2h_books,
                                                     "external_200ms": ext}
    matrix["cross_network_h2h_200ms"] = read_json(
        Path(CONFIG["campaign_root"]) / "cross_network_h2h_200ms" / "hybrid858_vs_main504_off" / "summary.json")
    for weight in ("pre300", "v300", "v301"):
        h2h = read_json(Path(CONFIG["campaign_root"]) / "network_revalidation" / "h2h" / "200ms" / f"{weight}_vs_pre300" / "summary.json")
        ext = {}
        for opponent in CONFIG["opponents"]:
            summary = read_json(Path(CONFIG["campaign_root"]) / "network_revalidation" / "external" / "200ms" / weight / opponent / "summary.json")
            ext[opponent] = summary
        matrix["network_revalidation"][weight] = {"h2h_200ms_vs_pre300": h2h, "external_200ms": ext}
    matrix["network_revalidation_h2h_200ms"] = {}
    for pair in ("v300_vs_pre300", "v301_vs_pre300", "v301_vs_v300"):
        matrix["network_revalidation_h2h_200ms"][pair] = read_json(
            Path(CONFIG["campaign_root"]) / "network_revalidation" / "h2h" / "200ms" / pair / "summary.json")
    matrix["network_revalidation_h2h_3plus2"] = {}
    for pair in ("v300_vs_pre300", "v301_vs_pre300", "v301_vs_v300"):
        dirs = [Path(CONFIG["overlay_root"]) / "network_revalidation" / "h2h" / "3plus2" / pair / f"shard_{i:03d}_of_004"
                for i in range(CONFIG["three_plus_two_shards"])]
        present = [d for d in dirs if (d / "games.jsonl").exists()]
        raw = load_games([d / "games.jsonl" for d in present])
        rows = latest_game_rows(raw)
        by_category = {cat: category_game_summary([r for r in rows if r.get("category") == cat],
                                                   CONFIG["three_plus_two_pairs"][cat])
                       for cat in ("center", "normal")}
        matrix["network_revalidation_h2h_3plus2"][pair] = {
            "status": "complete" if len(present) == CONFIG["three_plus_two_shards"] and
                      all(by_category[c]["complete_pairs"] == by_category[c]["expected_pairs"] and
                          by_category[c]["failed_games"] == 0 and by_category[c]["crash_or_forfeit_games"] == 0
                          for c in by_category) else ("incomplete" if present else "missing"),
            "shards_present": len(present), "shards_expected": CONFIG["three_plus_two_shards"],
            "raw_attempts": len(raw), "raw_failed_attempts": sum(r.get("status") != "ok" for r in raw),
            "latest_unique_games": len(rows),
            "results": by_category, "rankable": False}
    matrix["network_revalidation_external_3plus2"] = {}
    for weight in ("pre300", "v300", "v301"):
        matrix["network_revalidation_external_3plus2"][weight] = {}
        for opponent in CONFIG["opponents"]:
            dirs = [Path(CONFIG["overlay_root"]) / "network_revalidation" / "external" / "3plus2" / weight / opponent /
                    f"shard_{i:03d}_of_004" for i in range(CONFIG["three_plus_two_shards"])]
            present = [d for d in dirs if (d / "games.jsonl").exists()]
            raw = load_games([d / "games.jsonl" for d in present])
            rows = latest_game_rows(raw)
            categories = {}
            for category in ("center", "normal"):
                categories[category] = category_game_summary(
                    [r for r in rows if r.get("category") == category], CONFIG["three_plus_two_pairs"][category])
                if weight != "pre300":
                    baseline_path = Path(CONFIG["overlay_root"]) / "network_revalidation" / "external" / "3plus2" / "pre300" / opponent
                    baseline_dirs = [baseline_path / f"shard_{i:03d}_of_004" for i in range(CONFIG["three_plus_two_shards"])]
                    baseline_rows = latest_game_rows(load_games([d / "games.jsonl" for d in baseline_dirs if (d / "games.jsonl").exists()]))
                    categories[category]["paired_delta_candidate_minus_pre300"] = paired_field_delta(
                        rows, baseline_rows, category, CONFIG["three_plus_two_pairs"][category],
                        candidate_label=weight, baseline_label="pre300")
            crash_count = sum(r.get("termination") == "engine_error" for r in rows)
            fail_count = sum(r.get("status") != "ok" for r in rows)
            complete = (len(present) == CONFIG["three_plus_two_shards"] and fail_count == 0 and crash_count == 0 and
                        all(x["complete_pairs"] == x["expected_pairs"] for x in categories.values()))
            matrix["network_revalidation_external_3plus2"][weight][opponent] = {
                "status": "complete" if complete else ("incomplete" if present else "missing"),
                "shards_present": len(present), "shards_expected": CONFIG["three_plus_two_shards"],
                "raw_attempts": len(raw), "latest_unique_games": len(rows),
                "failed_games": fail_count, "crash_or_forfeit_games": crash_count,
                "time_loss_games": sum(r.get("termination") == "time" for r in rows),
                "results": categories, "rankable": False}
    parity, speed = parity_and_speed()
    matrix["parity_manifest"] = {"present": bool(parity), "finished_local": (parity or {}).get("finished_local"),
                                 "source": "results/benchmarks/edge-parity-local/parity_manifest.json" if parity else None}
    matrix["speed_manifest"] = {"present": bool(speed), "finished_utc": (speed or {}).get("finished_utc"),
                                "source": "results/benchmarks/edge-parity-local/isolated_speed.json" if speed else None}
    aliases = {"504": "n504", "858": "contact858"}
    for network, variants in matrix["networks"].items():
        alias = aliases[network]
        parity_data = parity or {}
        speed_data = speed or {}
        samples = [r for r in speed_data.get("samples", []) if r.get("network") in (network, alias)]
        for variant, data in variants.items():
            key = f"{alias}/{variant}"
            large = parity_data.get("large_tree", {}).get(key, {})
            acc = parity_data.get("accumulator", {}).get(key, {})
            fixed = parity_data.get("fixed_node", {}).get(key, {})
            if variant == "off":
                gates = [large.get("status") == "RAN", True, acc.get("status") == "PASS",
                         fixed.get("positions") == 80]
            else:
                gates = [large.get("status") == "RAN",
                         large.get("comparison_to_off", {}).get("status") == "PASS",
                         acc.get("status") == "PASS", fixed.get("comparison_to_off", {}).get("status") == "PASS"]
            speed_values = {}
            for v in ("off", variant):
                values = [float(r["nps"]) for r in samples if r.get("variant") == v and r.get("nps")]
                speed_values[v] = statistics.median(values) if values else None
            data["parity_status"] = "PASS" if all(gates) else "PENDING"
            data["parity_gates"] = {"large_tree": gates[0], "large_tree_vs_off": gates[1],
                                    "accumulator": gates[2], "fixed_node_vs_off": gates[3]}
            data["isolated_speed_nps"] = {"candidate_median": speed_values[variant],
                                          "off_median": speed_values["off"],
                                          "ratio_vs_off": (speed_values[variant] / speed_values["off"]
                                                           if speed_values[variant] and speed_values["off"] else None),
                                          "sample_count": sum(1 for r in samples if r.get("variant") == variant)}
    queue = read_json(Path(CONFIG["overlay_root"]) / "queue" / "status.json") or {}
    current = queue.get("current") or {}
    matrix["queue_status"] = {"state": queue.get("state"),
                              "completed_steps": [step.get("name") for step in queue.get("completed_steps", [])],
                              "current": {"name": current.get("name"), "workers": current.get("workers"),
                                          "attempt": current.get("attempt")},
                              "resource_wait": queue.get("resource_wait"),
                              "memory_policy": (queue.get("memory_policy") or queue.get("resource_policy") or
                                                queue.get("memory_budget")),
                              "ram": queue.get("ram"),
                              "resource_limits": queue.get("resource_limits"),
                              "updated_local": queue.get("updated_local"),
                              "error": queue.get("error")}
    matrix["promotion_recommendations"] = promotion_recommendations(matrix)
    return matrix


def strict_score_win(summary):
    expected = summary.get("expected_pairs")
    ci = summary.get("ci95_score_pct")
    complete = (expected is not None and summary.get("complete_pairs") == expected and
                summary.get("failed_games", 0) == 0 and summary.get("crash_or_forfeit_games", 0) == 0 and
                summary.get("score_pct") is not None)
    return complete, bool(complete and ci and ci[0] > 50.0)


def strict_delta_win(delta, expected=None):
    expected = expected if expected is not None else delta.get("expected_pairs")
    ci = delta.get("paired_bootstrap_95")
    complete = expected is not None and delta.get("complete_paired_openings") == expected and delta.get("delta_score_pct") is not None
    return complete, bool(complete and ci and ci[0] > 0.0)


def promotion_recommendations(matrix):
    output = {}
    for network, variants in matrix["networks"].items():
        for variant, data in variants.items():
            if variant == "off":
                continue
            evidence = []
            for dimension in ("all", "center", "normal"):
                summary = (data["h2h_200ms_vs_off"] if dimension == "all" else
                           data["h2h_200ms_by_category"][dimension])
                evidence.append((f"h2h_200ms_{dimension}", *strict_score_win(summary)))
            for opponent in CONFIG["opponents"]:
                item = data["external_200ms"][opponent]
                for category in ("center", "normal"):
                    candidate = item.get("participants", {}).get("candidate", {}).get(category, {})
                    off = item.get("off_absolute", {}).get(category, {})
                    complete = (candidate.get("complete_pairs") == candidate.get("expected_pairs") and
                                candidate.get("failed_games", 0) == 0 and
                                off.get("complete_pairs") == off.get("expected_pairs") and
                                off.get("failed_games", 0) == 0)
                    delta_complete, won = strict_delta_win(
                        item.get("paired_delta_candidate_minus_off", {}).get(category, {}),
                        CONFIG["external_pairs_200ms"][category])
                    evidence.append((f"{opponent}_200ms_{category}", complete and delta_complete,
                                     bool(complete and delta_complete and won)))
            for opponent in ("h2h", *CONFIG["opponents"]):
                shard = matrix["three_plus_two"].get(f"{network}/{variant}/{opponent}", {})
                categories = ("all", "center", "normal") if opponent == "h2h" else ("center", "normal")
                for category in categories:
                    result = shard.get("results", {}).get(category, {})
                    if opponent == "h2h":
                        complete, won = strict_score_win(result)
                        complete = complete and shard.get("status") == "complete"
                        won = won and complete
                    else:
                        complete, won = strict_delta_win(result.get("paired_delta_candidate_minus_off", {}))
                        complete = complete and shard.get("status") == "complete"
                        won = won and complete
                    if shard.get("crash_or_forfeit_games", 0):
                        complete, won = False, False
                    evidence.append((f"{opponent}_3plus2_{category}", complete, won))
            all_complete = all(item[1] for item in evidence)
            all_wins = all(item[2] for item in evidence)
            crash_count = sum(matrix["three_plus_two"].get(f"{network}/{variant}/{opp}", {}).get("crash_or_forfeit_games", 0)
                              for opp in ("h2h", *CONFIG["opponents"]))
            if crash_count:
                recommendation = "retain_main_crash_instability"
            elif not all_complete:
                recommendation = "retain_main_incomplete"
            elif all_wins:
                recommendation = "strict_win_review_only"
            else:
                recommendation = "retain_main_no_strict_win"
            output[f"{network}/{variant}"] = {"recommendation": recommendation,
                                               "automatic_promotion": False,
                                               "crash_or_forfeit_games_3plus2": crash_count,
                                               "all_dimensions_complete": all_complete,
                                               "all_dimensions_strict_wins": all_wins,
                                               "dimensions": [{"name": name, "complete": complete, "strict_win": won}
                                                              for name, complete, won in evidence]}
    return output


def fmt_score(s):
    if s.get("theoretical_reference"):
        return "50.0% theoretical reference; 0 Elo reference; N/A observed games, pairs, and CI"
    expected = s.get("expected_pairs")
    if (s.get("failed_games", 0) > 0 or s.get("crash_or_forfeit_games", 0) > 0 or
            (expected is not None and s.get("complete_pairs", 0) != expected)):
        return (f"incomplete ({s.get('complete_pairs', 0)}/{expected} pairs, "
                f"{s.get('failed_games', 0)} failed)")
    if s.get("score_pct") is None:
        return f"incomplete ({s.get('complete_pairs', 0)}/{s.get('expected_pairs')} pairs, {s.get('failed_games', 0)} failed)"
    ci = s.get("ci95_score_pct")
    ci_text = f"; 95% CI {ci[0]:.1f}–{ci[1]:.1f}%" if ci and None not in ci else ""
    elo_value = s.get("elo")
    elo_text = f"; {elo_value:+.1f} Elo" if elo_value is not None else ""
    return f"{s['score_pct']:.1f}%{elo_text}{ci_text}; {s.get('complete_pairs', 0)}/{s.get('expected_pairs')} pairs"


def render(matrix):
    lines = ["# Edge campaign status", "",
             "This report reads stored campaign summaries and game shards. Elo uses the standard score conversion. A result is complete only when the expected pairs exist and no game failed.", "",
             "## 200 ms matrix", "",
             "| Network | Variant | Parity | Isolated speed vs OFF | H2H vs OFF | Claustrophobia Center | Claustrophobia Normal | Titanium Center | Titanium Normal | 3+2 H2H | 3+2 Claustrophobia | 3+2 Titanium |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for network in CONFIG["networks"]:
        for variant in CONFIG["variants"]:
            data = matrix["networks"][network][variant]
            speed = data["isolated_speed_nps"]
            ratio = speed.get("ratio_vs_off")
            speed_text = f"{ratio:.3f}× ({speed.get('sample_count')} samples)" if ratio is not None else "unavailable"
            h = data["h2h_200ms_vs_off"]
            hs = read_json(Path(CONFIG["campaign_root"]) / f"h2h_{network}_{variant}" / "summary.json") or {}
            center_h = data["h2h_200ms_by_category"]["center"]
            normal_h = data["h2h_200ms_by_category"]["normal"]
            htext = f"All {fmt_score(h)}; Center {fmt_score(center_h)}; Normal {fmt_score(normal_h)}"
            ext_cells = []
            for opp in CONFIG["opponents"]:
                item = data["external_200ms"].get(opp, {})
                participant = "off" if variant == "off" else "candidate"
                candidate = (item.get("participants", {}).get(participant, {}) if variant == "off"
                             else item.get("participants", {}).get("candidate", {}))
                if variant == "off" and not candidate:
                    candidate = item.get("off_absolute", {})
                for cat in ("center", "normal"):
                    cell = fmt_score(candidate.get(cat, compact_summary(None)))
                    if variant != "off":
                        delta = item.get("paired_delta_candidate_minus_off", {}).get(cat, {})
                        value = delta.get("delta_score_pct")
                        ci = delta.get("paired_bootstrap_95")
                        n = delta.get("complete_paired_openings", 0)
                        if value is not None and isinstance(ci, list) and len(ci) == 2:
                            off_score = item.get("off_absolute", {}).get(cat, {}).get("score_pct")
                            candidate_score = candidate.get(cat, {}).get("score_pct")
                            delta_elo = ((elo(candidate_score) - elo(off_score))
                                         if candidate_score is not None and off_score is not None else None)
                            elo_text = f", Δ {delta_elo:+.1f} Elo" if delta_elo is not None else ""
                            cell += f"; paired Δ {value:+.1f}pp [{ci[0]:+.1f}, {ci[1]:+.1f}]{elo_text}, {n} pairs"
                        else:
                            cell += "; paired Δ incomplete"
                    ext_cells.append(cell)
            three = [matrix["three_plus_two"].get(f"{network}/{variant}/{opp}", {}) for opp in ("h2h", "claustrophobia", "titanium")]
            three_text = []
            for opponent, item in zip(("h2h", *CONFIG["opponents"]), three):
                category_text = []
                for cat in ("center", "normal"):
                    result = item.get("results", {}).get(cat, compact_summary(None))
                    if opponent != "h2h":
                        side = "off" if variant == "off" else "candidate"
                        result = result.get("participants", {}).get(side, compact_summary(None))
                    value = fmt_score(result)
                    if opponent != "h2h":
                        delta = item.get("results", {}).get(cat, {}).get("paired_delta_candidate_minus_off", {})
                        ci = delta.get("paired_bootstrap_95")
                        if delta.get("delta_score_pct") is not None and ci:
                            value += f"; paired Δ {delta['delta_score_pct']:+.1f}pp [{ci[0]:+.1f}, {ci[1]:+.1f}]"
                            if delta.get("delta_elo_estimate") is not None:
                                value += f"; Δ {delta['delta_elo_estimate']:+.1f} Elo"
                    if item.get("status") != "complete":
                        value = "PRELIM " + value
                    category_text.append(f"{cat.title()} {value}")
                three_text.append("; ".join(category_text) +
                                  f"; {item.get('shards_present',0)}/{item.get('shards_expected',4)} shards")
            lines.append("| " + " | ".join([network, variant, data["parity_status"], speed_text, htext, *ext_cells, *three_text]) + " |")
    lines += ["", "## Network weight revalidation at 200 ms", "",
              "| Weights | Claustrophobia Center | Claustrophobia Normal | Titanium Center | Titanium Normal |",
              "|---|---|---|---|---|"]
    for weight, data in matrix.get("network_revalidation", {}).items():
        cells = []
        for opponent in CONFIG["opponents"]:
            summary = data.get("external_200ms", {}).get(opponent) or {}
            for category in ("center", "normal"):
                found = summary.get("categories", {}).get(category)
                cell = compact_summary(found)
                cell["expected_pairs"] = CONFIG["external_pairs_200ms"][category]
                cells.append(fmt_score(cell))
        lines.append("| " + " | ".join([weight, *cells]) + " |")
    lines += ["", "H2H weight comparisons", "", "| Comparison | All | Center | Normal |", "|---|---|---|---|"]
    for pair, summary in matrix.get("network_revalidation_h2h_200ms", {}).items():
        cells = []
        for category in ("all", "center", "normal"):
            found = (summary or {}).get(category)
            item = compact_summary(found)
            expected = 200 if category == "all" else CONFIG["h2h_pairs"][category]
            item["expected_pairs"] = expected
            cells.append(fmt_score(item))
        lines.append("| " + " | ".join([pair, *cells]) + " |")
    lines += ["", "## Network weight revalidation at 3+2", "",
              "| Comparison | State | Center score or delta | Normal score or delta | Shards | Crashes or forfeits |",
              "|---|---|---|---|---:|---:|"]
    for pair, item in matrix.get("network_revalidation_h2h_3plus2", {}).items():
        values = [fmt_score(item.get("results", {}).get(cat, compact_summary(None))) for cat in ("center", "normal")]
        lines.append("| " + " | ".join([pair, item.get("status", "missing"), *values,
                                             f"{item.get('shards_present', 0)}/{item.get('shards_expected', 4)}",
                                             str(sum(x.get("crash_or_forfeit_games", 0) for x in item.get("results", {}).values()))]) + " |")
    for weight, opponents in matrix.get("network_revalidation_external_3plus2", {}).items():
        for opponent, item in opponents.items():
            values = []
            for cat in ("center", "normal"):
                result = item.get("results", {}).get(cat, {})
                score = fmt_score(result)
                delta = result.get("paired_delta_candidate_minus_pre300")
                if weight != "pre300" and delta and delta.get("delta_score_pct") is not None:
                    ci = delta.get("paired_bootstrap_95")
                    score += f"; paired Δ {delta['delta_score_pct']:+.1f}pp"
                    if ci:
                        score += f" [{ci[0]:+.1f}, {ci[1]:+.1f}]"
                    if delta.get("delta_elo_estimate") is not None:
                        score += f"; Δ {delta['delta_elo_estimate']:+.1f} Elo"
                values.append(score)
            lines.append("| " + " | ".join([f"{weight} vs {opponent}", item.get("status", "missing"), *values,
                                                 f"{item.get('shards_present', 0)}/{item.get('shards_expected', 4)}",
                                                 str(item.get("crash_or_forfeit_games", 0))]) + " |")
    lines += ["", "## Cross-network result", ""]
    cross = matrix.get("cross_network_h2h_200ms") or {}
    for category in ("all", "center", "normal"):
        s = compact_summary(cross.get(category))
        s["expected_pairs"] = CONFIG["h2h_pairs"].get(category, CONFIG["h2h_pairs"]["all"])
        lines.append(f"- Hybrid 858 versus Main 504 OFF, {category}: {fmt_score(s)}.")
    lines += ["", "## 3+2 shard status", "", "| Comparison | State | Shards | Raw attempts | Latest games | Raw failed attempts | Latest failed games | Crashes or forfeits | Time losses | Rankable |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, item in matrix["three_plus_two"].items():
        shard_cell = f"{item.get('shards_present',0)}/{item.get('shards_expected',4)}"
        if item.get("off_shards_present"):
            shard_cell += f" + OFF {item['off_shards_present']}/{item.get('shards_expected',4)}"
        lines.append(f"| {key} | {item.get('status')} | {shard_cell} | {item.get('raw_attempts',0)} | {item.get('latest_unique_games',0)} | {item.get('raw_failed_attempts',0)} | {item.get('failed_games',0)} | {item.get('crash_or_forfeit_games',0)} | {item.get('time_loss_games',0)} | {'yes' if item.get('rankable') else 'no'} |")
    for pair, item in matrix.get("network_revalidation_h2h_3plus2", {}).items():
        lines.append(f"| network_revalidation/{pair} | {item.get('status')} | {item.get('shards_present',0)}/{item.get('shards_expected',4)} | {item.get('raw_attempts',0)} | {item.get('latest_unique_games',0)} | {item.get('raw_failed_attempts',0)} | {sum(v.get('failed_games', 0) for v in item.get('results', {}).values())} | {sum(v.get('crash_or_forfeit_games', 0) for v in item.get('results', {}).values())} | {sum(v.get('time_loss_games', 0) for v in item.get('results', {}).values())} | no |")
    queue = matrix.get("queue_status") or {}
    current = queue.get("current") or {}
    v2_audit = matrix["three_plus_two"].get("network_revalidation/v300_vs_pre300/h2h", {})
    worker_count = current.get("workers") if current else None
    workers_text = str(worker_count) if worker_count is not None else "not reported"
    wait_detail = queue.get("resource_wait") or queue.get("memory_policy")
    ram = queue.get("ram") or {}
    ram_text = (f"{ram.get('free_mib')} MiB free against a {ram.get('reserve_mib')} MiB reserve"
                if ram.get("free_mib") is not None else "not reported")
    lines += ["", f"The queue state is `{queue.get('state') or 'unknown'}`. It records {len(queue.get('completed_steps', []))} completed steps. The current step is `{(current.get('name') if current else None) or 'none'}`. Status time: `{queue.get('updated_local') or 'not reported'}`.",
              f"The current step selects {workers_text} workers.",
              f"The resource status is `{json.dumps(wait_detail, ensure_ascii=False) if wait_detail is not None else 'none reported'}`.",
              f"The queue RAM reading is `{ram_text}`.",
              f"The queue error is `{queue.get('error', 'none')}`.",
              f"The v2 failed network shard contains {v2_audit.get('raw_failed_attempts', 0)} failed attempts and {v2_audit.get('failed_games', 0)} failures among {v2_audit.get('latest_unique_games', 0)} latest unique games. Failed rows do not count as wins.",
              "A result with `termination=time` is a clock loss and counts as a game result. A result with `termination=engine_error` is a crash or forfeit. Any crash or forfeit blocks promotion.",
              "No incomplete 3+2 comparison receives a rank.",
              "", "## Parity and speed", "", "The matrix gives parity gate status, manifest source paths, and isolated speed ratios when the manifests contain those measurements.", ""]
    lines += ["", "## Promotion recommendation", "", "No automatic promotion occurs. A candidate can enter human review only after complete 200 ms and 3+2 samples show a strict gain in aggregate H2H and every Center and Normal comparison against Main OFF, Claustrophobia, and Titanium. Any incomplete sample or crash keeps Main.", "", "| Candidate | Recommendation | Complete | Strict wins | 3+2 crashes or forfeits |", "|---|---|---|---|---:|"]
    for key, item in matrix.get("promotion_recommendations", {}).items():
        lines.append(f"| {key} | {item['recommendation']} | {'yes' if item['all_dimensions_complete'] else 'no'} | {'yes' if item['all_dimensions_strict_wins'] else 'no'} | {item['crash_or_forfeit_games_3plus2']} |")
    return "\n".join(lines)


def main():
    matrix = build_matrix()
    CONFIG["output_markdown"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["output_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["output_markdown"].write_text(render(matrix), encoding="utf-8")
    CONFIG["output_json"].write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {CONFIG['output_markdown']}")
    print(f"Wrote {CONFIG['output_json']}")


if __name__ == "__main__":
    main()
