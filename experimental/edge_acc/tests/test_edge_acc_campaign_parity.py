import hashlib
from pathlib import Path

import pytest

from tools import run_edge_acc_campaign as campaign


def valid_entry(tmp_path: Path) -> tuple[dict, dict, Path]:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"test weights")
    source_hashes = {"src\\nnue.hpp": "header-hash"}
    command = ["g++", "-O3", "-DNDEBUG", "-std=c++17", "-pthread", "-Isrc",
               "-DZQ_EXP_EDGE_ACC_CACHE=1"]
    key = "n504/v1"
    manifest = {
        "source_sha256": source_hashes,
        "weights": {"n504": {"sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}},
        "builds": [{"network": "n504", "variant": "v1", "kind": "uci",
                    "returncode": 0, "command": command}],
        "large_tree": {key: {"status": "RAN", "comparison_to_off": {
            "status": "PASS", "actual_off": 10, "actual_candidate": 10, "mismatches": []}}},
        "accumulator": {key: {"status": "PASS", "stdout": "EDGE_ACC_PARITY_OK"}},
        "fixed_node": {key: {"comparison_to_off": {
            "status": "PASS", "positions": 80, "mismatches": []}}},
    }
    return manifest, {"src/nnue.hpp": "header-hash"}, weights


def test_parity_gate_normalizes_windows_manifest_paths(tmp_path: Path) -> None:
    manifest, expected_source, weights = valid_entry(tmp_path)
    approved = campaign.validate_parity_entry(manifest, "504", "v1", expected_source, weights)
    assert approved["large_tree"]["comparison_to_off"]["actual_candidate"] == 10


def test_unsafe_v2_cannot_enter_game_campaign() -> None:
    with pytest.raises(RuntimeError, match="speed-only reference"):
        campaign.validate_parity("504", "v2")


def valid_alias_manifest(weights: Path) -> dict:
    return {
        "finished_local": "2026-10-06T12:00:00Z",
        "source_sha256": {"src\\nnue.hpp": "header-hash"},
        "weights": {"504": {"sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}},
        "comparisons": [
            {"network": "504", "variant": variant, "phase": phase, "nodes": nodes,
             "status": "PASS", "mismatches": {}}
            for variant in ("v1", "v3")
            for phase in ("warm", "alias")
            for nodes in (1000, 4000)
        ],
    }


def test_alias_gate_checks_variant_network_and_all_cases(tmp_path: Path) -> None:
    manifest, expected_source, weights = valid_entry(tmp_path)
    alias = valid_alias_manifest(weights)
    approved = campaign.validate_alias_entry(alias, "504", "v1", expected_source, weights)
    assert len(approved["comparisons"]) == 4

    alias["comparisons"] = [row for row in alias["comparisons"]
                            if not (row["variant"] == "v1" and row["phase"] == "alias"
                                    and row["nodes"] == 4000)]
    with pytest.raises(RuntimeError, match="incomplete for 504/v1"):
        campaign.validate_alias_entry(alias, "504", "v1", expected_source, weights)
    assert len(campaign.validate_alias_entry(
        valid_alias_manifest(weights), "504", "v3", expected_source, weights)["comparisons"]) == 4


@pytest.mark.parametrize("mutation, message", [
    ("source", "source hashes"),
    ("weight", "weight hash"),
    ("row", "gate failed for 504/v1"),
])
def test_alias_gate_rejects_stale_or_failed_evidence(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    _, expected_source, weights = valid_entry(tmp_path)
    alias = valid_alias_manifest(weights)
    if mutation == "source":
        alias["source_sha256"]["src\\nnue.hpp"] = "stale-header"
    elif mutation == "weight":
        alias["weights"]["504"]["sha256"] = "stale-weight"
    else:
        next(row for row in alias["comparisons"]
             if row["variant"] == "v1" and row["phase"] == "alias" and row["nodes"] == 1000)["status"] = "FAIL"
    with pytest.raises(RuntimeError, match=message):
        campaign.validate_alias_entry(alias, "504", "v1", expected_source, weights)


def test_natural_gate_requires_each_budget_and_both_sibling_visits(tmp_path: Path) -> None:
    _, expected_source, weights = valid_entry(tmp_path)
    nodes = campaign.CONFIG["parity"]["natural_node_budgets"]
    manifest = {
        "finished_local": "2026-10-06T12:00:00Z",
        "source_sha256": {"src\\nnue.hpp": "header-hash"},
        "weights": {"504": {"sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}},
        "comparisons": [{"network": "504", "variant": "v3_node", "nodes": budget,
                         "status": "PASS", "mismatches": {}} for budget in nodes],
        "results": [{"network": "504", "variant": "v3_node", "nodes": budget,
                     "status": "RAN", "h63_n": 25, "v63_n": 20}
                    for budget in nodes],
    }
    approved = campaign.validate_natural_entry(manifest, "504", "v3_node", expected_source, weights)
    assert set(approved["branch_visits"]) == set(nodes)

    manifest["results"][0]["v63_n"] = 0
    with pytest.raises(RuntimeError, match="not both visited"):
        campaign.validate_natural_entry(manifest, "504", "v3_node", expected_source, weights)


@pytest.mark.parametrize("mutation, message", [
    ("source", "source hashes"),
    ("weight", "weight hash"),
    ("flags", "macro flags"),
])
def test_parity_gate_rejects_changed_provenance(tmp_path: Path, mutation: str, message: str) -> None:
    manifest, expected_source, weights = valid_entry(tmp_path)
    if mutation == "source":
        expected_source["src/nnue.hpp"] = "different-header-hash"
    elif mutation == "weight":
        manifest["weights"]["n504"]["sha256"] = "different-weight-hash"
    else:
        manifest["builds"][0]["command"][-1] = "-DZQ_EXP_EDGE_ACC_CACHE=0"
    with pytest.raises(RuntimeError, match=message):
        campaign.validate_parity_entry(manifest, "504", "v1", expected_source, weights)
