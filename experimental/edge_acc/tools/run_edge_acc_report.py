#!/usr/bin/env python3
"""Build a parity, speed, and paired-game report for EdgeAcc variants."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmarks" / "edge_acc_campaign"
CONFIG = {
    "output_markdown": OUT / "edge_acc_report.md",
    "output_deltas": OUT / "edge_acc_category_deltas.json",
    "parity_manifest": ROOT / "results" / "benchmarks" / "edge-parity-local" / "parity_manifest.json",
    "alias_manifest": ROOT / "results" / "benchmarks" / "edge-parity-local" / "edge_acc_alias_tree_parity.json",
    "natural_manifest": ROOT / "results" / "benchmarks" / "edge-parity-local" / "edge_acc_natural_tree_parity.json",
    "archived_parent_manifest": ROOT / "results" / "benchmarks" / "edge-parity-local"
    / "archive" / "pre_distance_replay" / "manifest_node_context_before_replay.json",
    # The parity runner writes paired timing samples in this manifest.
    "speed_manifest": ROOT / "results" / "benchmarks" / "edge-parity-local" / "isolated_speed.json",
    "variants": ["off", "v1", "v2", "v3", "v3_parent", "v3_node", "hybrid_sparse", "hybrid_dense",
                 "v3_node_dense", "hybrid_dense_bfs", "v3_node_dense_bfs",
                 "delta_dense_only"],
    "networks": {"504": "n504", "858": "contact858"},
    "speedup_metric": "nps",
    "h2h_pairs": {"center": 133, "normal": 67},
    "external_pairs_200ms": {"center": 100, "normal": 50},
    "three_plus_two_pairs": {"center": 100, "normal": 100},
    "queue_screening_variants": ("v3", "v3_node_dense_bfs", "delta_dense_only"),
    "bootstrap": 20000,
    "seed": 20261006,
}

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
from tools import run_edge_acc_campaign as campaign
from tools.external import local_arena


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def latest_rows(paths: list[Path], key_fields: tuple[str, ...]) -> list[dict]:
    latest = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
                key = tuple(row.get(field) for field in key_fields)
            except (json.JSONDecodeError, TypeError):
                continue
            latest[key] = row
    return list(latest.values())


@lru_cache(maxsize=256)
def _summarize_cached(rows_json: str) -> dict:
    return local_arena.summarize_pairs(json.loads(rows_json), bootstrap=CONFIG["bootstrap"],
                                       seed=CONFIG["seed"])


def summarize(rows: list[dict]) -> dict:
    # Preserve row order because it determines insertion order of paired
    # outcomes and therefore the fixed-seed bootstrap stream. Move telemetry
    # is intentionally omitted; run_id remains as the campaign identity.
    fields = ("opponent", "opening_index", "zq_player", "status", "result",
              "termination", "repeated_states", "plies", "run_id")
    projected = [{field: row.get(field) for field in fields} for row in rows]
    return _summarize_cached(json.dumps(projected, sort_keys=True, separators=(",", ":")))


def complete(summary: dict | None, pairs: int) -> bool:
    return bool(summary and summary.get("failed_games") == 0
                and summary.get("complete_pairs") == pairs)


def parity_status(manifest: dict | None, network: str, variant: str) -> str:
    archived = variant == "v3_parent"
    if archived:
        manifest = read_json(Path(CONFIG["archived_parent_manifest"]))
    if not manifest or not manifest.get("finished_local"):
        return "PENDING"
    key = f"{CONFIG['networks'][network]}/{variant}"
    large = manifest.get("large_tree", {}).get(key, {})
    accumulator = manifest.get("accumulator", {}).get(key, {})
    fixed = manifest.get("fixed_node", {}).get(key, {})
    if variant == "off":
        reference_ok = (large.get("status") == "RAN"
                        and accumulator.get("status") == "PASS"
                        and "EDGE_ACC_PARITY_OK" in accumulator.get("stdout", "")
                        and fixed.get("positions") == 80)
        return "PASS (OFF ref)" if reference_ok else "PENDING"
    large_ok = (large.get("status") == "RAN"
                and large.get("comparison_to_off", {}).get("status") == "PASS"
                and large.get("comparison_to_off", {}).get("actual_off") == 10
                and large.get("comparison_to_off", {}).get("actual_candidate") == 10
                and not large.get("comparison_to_off", {}).get("mismatches"))
    acc_ok = accumulator.get("status") == "PASS" and "EDGE_ACC_PARITY_OK" in accumulator.get("stdout", "")
    fixed_ok = (fixed.get("comparison_to_off", {}).get("status") == "PASS"
                and fixed.get("comparison_to_off", {}).get("positions") == 80
                and not fixed.get("comparison_to_off", {}).get("mismatches"))
    if large_ok and acc_ok and fixed_ok:
        return "PASS (archived)" if archived else "PASS"
    if any(section.get("status") == "FAIL"
           for section in (large.get("comparison_to_off", {}), accumulator,
                          fixed.get("comparison_to_off", {}))):
        return "FAIL (archived)" if archived else "FAIL"
    return "PENDING"


def alias_status(manifest: dict | None, network: str, variant: str) -> str:
    if variant == "v3_parent":
        return "FAIL (archived)"
    if not manifest:
        return "PENDING alias"
    try:
        source_hashes = campaign.source_hash_map()
        weight_path = Path(campaign.CONFIG["networks"][network]["weights"]).resolve()
        campaign.validate_alias_entry(manifest, network, variant, source_hashes, weight_path)
    except RuntimeError as error:
        if "failed for" in str(error):
            return "FAIL alias"
        return "PENDING alias"
    return "PASS"


def natural_status(manifest: dict | None, network: str, variant: str) -> str:
    if variant not in campaign.CONFIG["parity"]["natural_variants"]:
        return "NOT REQUIRED"
    if not manifest:
        return "PENDING natural"
    try:
        source_hashes = campaign.source_hash_map()
        weight_path = Path(campaign.CONFIG["networks"][network]["weights"]).resolve()
        campaign.validate_natural_entry(manifest, network, variant, source_hashes, weight_path)
    except RuntimeError as error:
        if "failed for" in str(error) or "not both visited" in str(error):
            return "FAIL natural"
        return "PENDING natural"
    return "PASS"


def center_loss_ci(summary: dict | None, pairs: int) -> bool:
    if not complete(summary, pairs):
        return False
    interval = summary.get("paired_bootstrap_95") or {}
    upper = interval.get("score_high_pct")
    return upper is not None and upper < 50.0


def delta_loss_ci(summary: dict | None, pairs: int) -> bool:
    if not summary or summary.get("complete_paired_openings") != pairs:
        return False
    interval = summary.get("paired_bootstrap_95")
    return isinstance(interval, list) and len(interval) == 2 and interval[1] < 0.0


def timing_samples(manifest: dict | None) -> dict[tuple[str, str], list[float]]:
    output: dict[tuple[str, str], list[float]] = defaultdict(list)
    if not manifest:
        return output
    rows = manifest.get("samples", manifest.get("results", []))
    if isinstance(rows, dict):
        rows = rows.get("samples", [])
    for row in rows:
        try:
            network_value = str(row["network"])
            network = next((number for number, alias in CONFIG["networks"].items()
                            if network_value in (number, alias)), network_value)
            variant = str(row["variant"])
            value = float(row.get(CONFIG["speedup_metric"], row.get("nodes_per_second")))
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0 and math.isfinite(value):
            output[(network, variant)].append(value)
    return output


def speed_cell(samples: dict, network: str, variant: str, baseline: str) -> str:
    candidate = samples.get((network, variant), [])
    reference = samples.get((network, baseline), [])
    if not candidate or not reference:
        return "—"
    value = statistics.median(candidate) / statistics.median(reference)
    if variant == baseline:
        return "1.000×"
    return f"{value:.3f}×"


def speed_details(samples: dict, network: str, variant: str) -> dict:
    values = samples.get((network, variant), [])
    if not values:
        return {"sample_count": 0, "median_nps": None, "range_nps": None}
    return {"sample_count": len(values), "median_nps": statistics.median(values),
            "range_nps": [min(values), max(values)]}


def score_cell(summary: dict | None, pairs: int) -> str:
    if not complete(summary, pairs):
        return "—"
    ci = summary.get("paired_bootstrap_95") or {}
    low, high = ci.get("score_low_pct"), ci.get("score_high_pct")
    score = summary.get("score_pct")
    if score is None or low is None or high is None:
        return "—"
    return f"{score:.1f}% [{low:.1f}, {high:.1f}]"


def delta_cell(delta: dict | None, pairs: int) -> str:
    if not delta or delta.get("complete_paired_openings") != pairs:
        return "—"
    ci = delta.get("paired_bootstrap_95")
    value = delta.get("delta_score_pct")
    if not isinstance(ci, list) or value is None:
        return "—"
    return f"{value:+.1f}pp [{ci[0]:+.1f}, {ci[1]:+.1f}]"


def h2h_200(root: Path, network: str, variant: str) -> dict:
    folder = root / f"h2h_{network}_{variant}"
    summary = read_json(folder / "summary.json")
    if not summary:
        return {}
    return {category: summary.get(category) for category in ("normal", "center")}


def external_200(root: Path, network: str, variant: str, opponent: str) -> dict:
    paired_path = root / "200ms" / network / variant / opponent / "games.jsonl"
    baseline_path = root / "200ms" / network / "off" / opponent / "games.jsonl"
    paths = list(dict.fromkeys((paired_path, baseline_path)))
    rows = latest_rows(paths, ("participant", "category", "opening_index", "zq_player"))
    participants = {}
    for name in ("off", "candidate"):
        participants[name] = {
            category: summarize([row for row in rows if row.get("participant") == name
                                 and row.get("category") == category])
            for category in ("normal", "center")
        }
    delta = {category: campaign.paired_candidate_delta(rows, category)
             for category in ("normal", "center")}
    return {"participants": participants, "delta": delta}


def three_plus_two(root: Path, network: str, variant: str, opponent: str) -> tuple[dict, dict]:
    base = root / "3plus2" / network / variant
    if opponent == "h2h":
        folders = [base / f"{opponent}_shard_{i:03d}_of_004" for i in range(4)]
        raw_paths = [folder / "games.jsonl" for folder in folders]
        rows = latest_rows(raw_paths, ("opening_index", "zq_player"))
    else:
        candidate_paths = [base / opponent / f"shard_{i:03d}_of_004" / "games.jsonl"
                           for i in range(4)]
        baseline_paths = [root / "3plus2" / network / "off" / opponent /
                          f"shard_{i:03d}_of_004" / "games.jsonl" for i in range(4)]
        # Baseline-only OFF rows are authoritative when an older paired run
        # also contains OFF games for the same opening and side.
        raw_paths = list(dict.fromkeys(candidate_paths + baseline_paths))
        rows = latest_rows(raw_paths, ("participant", "category", "opening_index", "zq_player"))
    if opponent == "h2h":
        output = {}
        for category, expected in CONFIG["three_plus_two_pairs"].items():
            group = [row for row in rows if row.get("category") == category]
            output[category] = summarize(group)
            output[category]["expected_pairs"] = expected
        return output, {}
    participants = {}
    for name in ("off", "candidate"):
        participants[name] = {}
        for category, expected in CONFIG["three_plus_two_pairs"].items():
            group = [row for row in rows if row.get("participant") == name and row.get("category") == category]
            participants[name][category] = summarize(group)
            participants[name][category]["expected_pairs"] = expected
    deltas = {}
    for category, expected in CONFIG["three_plus_two_pairs"].items():
        paired = {}
        for row in rows:
            if row.get("category") != category or row.get("status") != "ok":
                continue
            key = int(row["opening_index"])
            paired.setdefault(key, {}).setdefault(row["participant"], {})[int(row["zq_player"])] = float(row["result"])
        changes = [(sum(pair["candidate"].values()) - sum(pair["off"].values())) / 2.0
                   for pair in paired.values()
                   if set(pair) == {"candidate", "off"}
                   and all(set(pair[name]) == {0, 1} for name in ("candidate", "off"))]
        if len(changes) == expected:
            mean = sum(changes) / len(changes)
            import random
            rng = random.Random(CONFIG["seed"])
            boot = sorted(sum(changes[rng.randrange(len(changes))] for _ in changes) / len(changes)
                          for _ in range(CONFIG["bootstrap"]))
            deltas[category] = {"complete_paired_openings": len(changes),
                                "delta_score_pct": 100 * mean,
                                "paired_bootstrap_95": [100 * boot[int(0.025 * len(boot))],
                                                         100 * boot[min(len(boot)-1, int(0.975 * len(boot)))]]}
        else:
            deltas[category] = {"complete_paired_openings": len(changes)}
    return participants, deltas


def build_report() -> tuple[str, dict]:
    root = Path(CONFIG["campaign_root"] if "campaign_root" in CONFIG else OUT)
    parity = read_json(Path(CONFIG["parity_manifest"]))
    # Queue status is deliberately read as data only; importing the queue module
    # could initialize its Windows locking/runtime machinery.
    queue_status = read_json(root / "queue" / "status.json")
    alias = read_json(Path(CONFIG["alias_manifest"]))
    natural = read_json(Path(CONFIG["natural_manifest"]))
    speed = timing_samples(read_json(Path(CONFIG["speed_manifest"])))
    lines = ["| Variant | Network | Parity | Speedup vs OFF | Speedup vs v1 | H2H 200 N | H2H 200 Center | Claustro 200 N | Claustro 200 Center | 3+2 H2H | 3+2 Claustro | 3+2 Titanium | Decision |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    deltas_report = {}
    for variant in CONFIG["variants"]:
        for network in CONFIG["networks"]:
            core_parity = parity_status(parity, network, variant)
            alias_cell = alias_status(alias, network, variant)
            natural_cell = natural_status(natural, network, variant)
            if alias_cell.startswith("FAIL"):
                parity_cell = "FAIL alias"
            elif natural_cell.startswith("FAIL"):
                parity_cell = "FAIL natural"
            elif core_parity.startswith("PASS") and alias_cell == "PASS" and natural_cell == "PASS":
                parity_cell = f"{core_parity} / alias PASS / natural PASS"
            elif core_parity.startswith("PASS"):
                parity_cell = natural_cell if natural_cell.startswith("PENDING") else alias_cell
            else:
                parity_cell = core_parity
            parity_pass = parity_cell.startswith("PASS")
            h2h = h2h_200(root, network, variant)
            claustro = external_200(root, network, variant, "claustrophobia")
            titanium = external_200(root, network, variant, "titanium")
            h2h_3, _ = three_plus_two(root, network, variant, "h2h")
            cl_3, cl_3_delta = three_plus_two(root, network, variant, "claustrophobia")
            ti_3, ti_3_delta = three_plus_two(root, network, variant, "titanium")
            # Keep shared OFF absolute results visible even when candidate
            # parity is pending or failed; only candidate comparisons are gated.
            cl_200_off = claustro.get("participants", {}).get("off", {})
            ti_200_off = titanium.get("participants", {}).get("off", {})
            cl_3_off = cl_3.get("off", {})
            ti_3_off = ti_3.get("off", {})
            if not parity_pass:
                h2h = {}
                h2h_3 = {}
                cl_3_delta = {}
                ti_3_delta = {}
                claustro = {"delta": {}}
                titanium = {"delta": {}}
            is_baseline = variant == "off"
            values = [
                score_cell(h2h.get("normal"), CONFIG["h2h_pairs"]["normal"]),
                score_cell(h2h.get("center"), CONFIG["h2h_pairs"]["center"]),
                (score_cell(cl_200_off.get("normal"), CONFIG["external_pairs_200ms"]["normal"])
                 if is_baseline else delta_cell(claustro.get("delta", {}).get("normal"), CONFIG["external_pairs_200ms"]["normal"])),
                (score_cell(cl_200_off.get("center"), CONFIG["external_pairs_200ms"]["center"])
                 if is_baseline else delta_cell(claustro.get("delta", {}).get("center"), CONFIG["external_pairs_200ms"]["center"])),
                f"N {score_cell(h2h_3.get('normal'), CONFIG['three_plus_two_pairs']['normal'])}; "
                f"C {score_cell(h2h_3.get('center'), CONFIG['three_plus_two_pairs']['center'])}",
                (f"N {score_cell(cl_3_off.get('normal'), CONFIG['three_plus_two_pairs']['normal'])}; "
                 f"C {score_cell(cl_3_off.get('center'), CONFIG['three_plus_two_pairs']['center'])}" if is_baseline else
                 f"N {delta_cell(cl_3_delta.get('normal'), CONFIG['three_plus_two_pairs']['normal'])}; "
                 f"C {delta_cell(cl_3_delta.get('center'), CONFIG['three_plus_two_pairs']['center'])}"),
                (f"N {score_cell(ti_3_off.get('normal'), CONFIG['three_plus_two_pairs']['normal'])}; "
                 f"C {score_cell(ti_3_off.get('center'), CONFIG['three_plus_two_pairs']['center'])}" if is_baseline else
                 f"N {delta_cell(ti_3_delta.get('normal'), CONFIG['three_plus_two_pairs']['normal'])}; "
                 f"C {delta_cell(ti_3_delta.get('center'), CONFIG['three_plus_two_pairs']['center'])}"),
            ]
            # Center Rush is the primary gate. Its outcome appears first in the
            # separate delta artifact and any completed negative result is explicit.
            center_values = []
            for summary, category, expected_pairs in (
                (h2h.get("center"), "h2h", CONFIG["h2h_pairs"]["center"]),
                (h2h_3.get("center"), "h2h", CONFIG["three_plus_two_pairs"]["center"]),
                (claustro.get("delta", {}).get("center"), "delta", CONFIG["external_pairs_200ms"]["center"]),
                (cl_3_delta.get("center"), "delta", CONFIG["three_plus_two_pairs"]["center"]),
                (ti_3_delta.get("center"), "delta", CONFIG["three_plus_two_pairs"]["center"])):
                if category == "h2h" and complete(summary, expected_pairs):
                    center_values.append(summary["score_pct"] - 50.0)
                elif category == "delta" and summary and summary.get("complete_paired_openings") == expected_pairs:
                    center_values.append(summary["delta_score_pct"])
            expected = [
                complete(h2h.get("normal"), CONFIG["h2h_pairs"]["normal"]),
                complete(h2h.get("center"), CONFIG["h2h_pairs"]["center"]),
                bool(claustro.get("delta", {}).get("normal", {}).get("complete_paired_openings") == CONFIG["external_pairs_200ms"]["normal"]),
                bool(claustro.get("delta", {}).get("center", {}).get("complete_paired_openings") == CONFIG["external_pairs_200ms"]["center"]),
                complete(h2h_3.get("normal"), CONFIG["three_plus_two_pairs"]["normal"]),
                complete(h2h_3.get("center"), CONFIG["three_plus_two_pairs"]["center"]),
                cl_3_delta.get("normal", {}).get("complete_paired_openings") == CONFIG["three_plus_two_pairs"]["normal"],
                cl_3_delta.get("center", {}).get("complete_paired_openings") == CONFIG["three_plus_two_pairs"]["center"],
                ti_3_delta.get("normal", {}).get("complete_paired_openings") == CONFIG["three_plus_two_pairs"]["normal"],
                ti_3_delta.get("center", {}).get("complete_paired_openings") == CONFIG["three_plus_two_pairs"]["center"],
            ]
            if is_baseline:
                off_complete = all(
                    complete(cl_200_off.get(category), CONFIG["external_pairs_200ms"][category])
                    and complete(ti_200_off.get(category), CONFIG["external_pairs_200ms"][category])
                    and complete(cl_3_off.get(category), CONFIG["three_plus_two_pairs"][category])
                    and complete(ti_3_off.get(category), CONFIG["three_plus_two_pairs"][category])
                    for category in ("normal", "center"))
                decision = "Baseline" if off_complete else "Baseline pending external"
            elif parity_cell.startswith("FAIL"):
                decision = "Parity fail / reference only"
            elif not parity_pass:
                decision = "Parity pending"
            elif variant not in CONFIG["queue_screening_variants"]:
                decision = "Speed-only; not selected for screening"
            elif queue_status:
                screening = queue_status.get("screening", {}).get(f"{network}/{variant}", {})
                if screening.get("h2h_pass") is False:
                    decision = f"Rejected H2H: {screening.get('h2h_reason', 'screening failed')}"
                elif screening.get("claustro_pass") is False:
                    decision = f"Rejected Claustro: {screening.get('claustro_reason', 'screening failed')}"
                elif network in queue_status.get("finalists", {}) and variant not in queue_status["finalists"][network]:
                    decision = "Not selected for H2H 3+2 (rank)"
                elif not screening.get("h2h_pass") or screening.get("claustro_pass") is not True:
                    decision = "Pending games"
                elif center_loss_ci(h2h.get("center"), CONFIG["h2h_pairs"]["center"]) or \
                        center_loss_ci(h2h_3.get("center"), CONFIG["three_plus_two_pairs"]["center"]) or \
                        delta_loss_ci(claustro.get("delta", {}).get("center"), CONFIG["external_pairs_200ms"]["center"]) or \
                        delta_loss_ci(cl_3_delta.get("center"), CONFIG["three_plus_two_pairs"]["center"]) or \
                        delta_loss_ci(ti_3_delta.get("center"), CONFIG["three_plus_two_pairs"]["center"]):
                    decision = "Center loss (95% CI)"
                elif not all(expected):
                    decision = "Pending games"
                else:
                    decision = "Complete for review"
            elif center_loss_ci(h2h.get("center"), CONFIG["h2h_pairs"]["center"]) or \
                    center_loss_ci(h2h_3.get("center"), CONFIG["three_plus_two_pairs"]["center"]) or \
                    delta_loss_ci(claustro.get("delta", {}).get("center"), CONFIG["external_pairs_200ms"]["center"]) or \
                    delta_loss_ci(cl_3_delta.get("center"), CONFIG["three_plus_two_pairs"]["center"]) or \
                    delta_loss_ci(ti_3_delta.get("center"), CONFIG["three_plus_two_pairs"]["center"]):
                decision = "Center loss (95% CI)"
            elif not all(expected):
                decision = "Pending games"
            else:
                decision = "Complete for review"
            lines.append("|" + "|".join([variant, network, parity_cell,
                speed_cell(speed, network, variant, "off"), speed_cell(speed, network, variant, "v1"),
                *values, decision]) + "|")
            deltas_report[f"{network}/{variant}"] = {
                "speed": speed_details(speed, network, variant),
                "speedup_vs_off": speed_cell(speed, network, variant, "off"),
                "speedup_vs_v1": speed_cell(speed, network, variant, "v1"),
                "speed_reference_only": bool(parity_cell.startswith("FAIL")),
                "strength_suppressed_due_to_parity": not parity_pass,
                "h2h_200ms_vs_main": {"normal": h2h.get("normal"), "center": h2h.get("center")},
                "claustrophobia_200ms_candidate_minus_main": claustro.get("delta", {}),
                "claustrophobia_200ms_off_absolute": cl_200_off,
                "titanium_200ms_candidate_minus_main": titanium.get("delta", {}),
                "titanium_200ms_off_absolute": ti_200_off,
                "h2h_3plus2_vs_main": h2h_3,
                "claustrophobia_3plus2_candidate_minus_main": cl_3_delta,
                "claustrophobia_3plus2_off_absolute": cl_3_off,
                "titanium_3plus2_candidate_minus_main": ti_3_delta,
                "titanium_3plus2_off_absolute": ti_3_off,
                "center_gate_deltas_pp": center_values,
                "decision": decision,
            }
    lines.extend(["", "External columns show the OFF absolute score on the OFF row and paired candidate-minus-OFF deltas on candidate rows. Speed measurements remain visible for parity-failed variants as reference-only performance data; game-strength fields are suppressed for those rows."])
    return "\n".join(lines) + "\n", deltas_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=Path(CONFIG["output_markdown"]))
    parser.add_argument("--deltas", type=Path, default=Path(CONFIG["output_deltas"]))
    args = parser.parse_args()
    markdown, deltas = build_report()
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.deltas.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown, encoding="utf-8")
    args.deltas.write_text(json.dumps(deltas, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.deltas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
